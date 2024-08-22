import torch
import math
import os
import cv2
from PIL import Image
import numpy as np

from modules.mesh_fusion.util import get_pinhole_intrinsics_from_fov
from modules.mesh_fusion.render import (
    features_to_world_space_mesh,
    render_mesh,
)
from utils.common_utils import (
    visualize_depth_numpy,
    save_rgbd,
)
import torch.nn.functional as F

import utils.functions as functions
import time
from modules.geo_predictors.PanoFusionDistancePredictor import PanoFusionDistancePredictor
from utils.camera_utils import *

from modules.equilib import equi2pers, cube2equi, equi2cube
from utils.warp_utils import transformation_from_parameters

class PanoWarp(torch.nn.Module):
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # renderer setting
        self.blur_radius = 0
        self.faces_per_pixel = 8
        self.fov = 90
        self.R, self.T = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]), torch.Tensor([[0., 0., 0.]])
        self.pano_width, self.pano_height = 1024 * 2, 512 * 2
        self.H, self.W = 512, 512
        self.device = "cuda:0"

        self.models_path = 'checkpoints'
        self.fix_structure = False

        # initialize global point-cloud / mesh structures
        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device)  # depth rendered from point cloud
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points
        self.vertices = torch.empty((3, 0), device=self.device, requires_grad=False)
        self.colors = torch.empty((3, 0), device=self.device, requires_grad=False)
        self.faces = torch.empty((3, 0), device=self.device, dtype=torch.long, requires_grad=False)
        self.pix_to_face = None

        # create exp dir
        timestamp = str(int(time.time()))
        self.setting = f"SDFT_pseudo_pairs"
        self.save_path = f'output/{self.setting}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.world_to_cam = torch.eye(4, dtype=torch.float32, device=self.device)
        self.K_44 = get_pinhole_intrinsics_from_fov(H=self.H, W=self.W, fov_in_degrees=self.fov).to(self.world_to_cam)
        self.K_b33 = self.K_44[:3,:3].unsqueeze(0)

    def empty_mesh(self):
        # initialize global point-cloud / mesh structures
        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device)  # depth rendered from point cloud
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points
        self.vertices = torch.empty((3, 0), device=self.device, requires_grad=False)
        self.colors = torch.empty((3, 0), device=self.device, requires_grad=False)
        self.faces = torch.empty((3, 0), device=self.device, dtype=torch.long, requires_grad=False)
        self.pix_to_face = None

    def project(self, world_to_cam):
        # project mesh into pose and render (rgb, depth, mask)
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf, self.mesh = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.H,
            W=self.W,
            fov_in_degrees=self.fov,
            RT=world_to_cam,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel
        )
        # mask rendered_image_tensor
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask

        # stable diffusion models want the mask and image as PIL images
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        self.inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        self.inpaint_mask_restore = self.inpaint_mask
        self.inpaint_mask_pil_restore = self.inpaint_mask_pil

        return rendered_image_tensor, rendered_image_pil

    def rgbd_to_mesh(self, rgb, depth, world_to_cam=None, mask=None, pix_to_face=None, using_distance_map=False):
        predicted_depth = depth.cuda()
        rgb = rgb.squeeze(0).cuda()
        if world_to_cam is None:
            world_to_cam = torch.eye(4, dtype=torch.float32)
        world_to_cam = world_to_cam.cuda()
        if pix_to_face is not None:
            self.pix_to_face = pix_to_face
        if mask is None:
            self.inpaint_mask = torch.ones_like(predicted_depth).bool()
        else:
            self.inpaint_mask = mask #[H,W]

        if self.inpaint_mask.sum() == 0:
            return
        
        vertices, faces, colors = features_to_world_space_mesh(
            colors=rgb,
            depth=predicted_depth,
            fov_in_degrees=self.fov,
            world_to_cam=world_to_cam,
            mask=self.inpaint_mask,
            pix_to_face=self.pix_to_face,
            faces=self.faces,
            vertices=self.vertices,
            using_distance_map=using_distance_map,
        )

        faces += self.vertices.shape[1] 

        self.vertices_restore = self.vertices.clone()
        self.colors_restore = self.colors.clone()
        self.faces_restore = self.faces.clone()

        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)


    def load_pano(self):
        image_path = f"input/input_panorama.png"
        image = Image.open(image_path)
        if image.size[0] < image.size[1]:
            image = image.transpose(Image.TRANSPOSE)
        image = functions.resize_image_with_aspect_ratio(image, new_width=self.pano_width)

        panorama_tensor = torch.tensor(np.array(image))[...,:3].permute(2,0,1).unsqueeze(0).float()/255

        pano_fusion_distance_predictor = PanoFusionDistancePredictor()
        depth = pano_fusion_distance_predictor.predict(panorama_tensor.squeeze(0).permute(1,2,0)) #input:HW3
        print(f"pano_fusion_distance...[{depth.min(), depth.mean(),depth.max()}]")
                
        return panorama_tensor, depth# panorama_tensor:BCHW, depth:HW

    def pano_to_perpective(self, pano_bchw, pitch, yaw):
        rots = {
            'roll': 0.,
            'pitch': pitch,  # rotate vertical
            'yaw': yaw,  # rotate horizontal
        }

        perspective = equi2pers(
            equi=pano_bchw.squeeze(0),
            rots=rots,
            height=self.H,
            width=self.W,
            fov_x=self.fov,
            mode="bilinear",
        ).unsqueeze(0) # BCHW

        return perspective

    def get_rand_ext(self, bs=1, range_scale=2):
        def rand_tensor(r, l):
            if r < 0:  
                return torch.zeros((l, 1, 1))
            rand = torch.rand((l, 1, 1))        
            sign = 2 * (torch.randn_like(rand) > 0).float() - 1
            return sign * (r / 2 + r / 2 * rand)

        trans_range={"x":0.4*range_scale, "y":-0.4*range_scale, "z":-0.4*range_scale, "a":-0.4*range_scale, "b":-0.4*range_scale, "c":-0.4*range_scale}
        x, y, z = trans_range['x'], trans_range['y'], trans_range['z']
        a, b, c = trans_range['a'], trans_range['b'], trans_range['c']
        cix = rand_tensor(x, bs)
        ciy = rand_tensor(y, bs)
        ciz = rand_tensor(z, bs)
        aix = rand_tensor(math.pi / a, bs)
        aiy = rand_tensor(math.pi / b, bs)
        aiz = rand_tensor(math.pi / c, bs)
        
        axisangle = torch.cat([aix, aiy, aiz], dim=-1)  # [b,1,3]
        translation = torch.cat([cix, ciy, ciz], dim=-1)
        
        cam_ext = transformation_from_parameters(axisangle, translation)  # [b,4,4]
        cam_ext_inv = torch.inverse(cam_ext)  # [b,4,4]
        return cam_ext, cam_ext_inv

    def get_pairs(self, view_rgb, view_depth, pose_cnt=2):
        all_poses = []
        
        for i in range(pose_cnt):
            cam_ext, cam_ext_inv = self.get_rand_ext()  # [b,4,4]
            cur_pose = cam_ext
            all_poses += [cur_pose]

        ref_depth = view_depth
        ref_img = view_rgb
        W, H = 512, 512

        inpaint_pairs = []  #(warp_back_image, warp_back_disp, warp_back_mask, ref_img, ref_depth)
        val_pairs = [] #(cam_ext, ref_img, warp_image, warp_disp, warp_mask, gt_img)

        for i, cur_pose in enumerate(all_poses[:]):
            print("-poses_idx:",i)
            cur_pose = all_poses[i]
            c2w = cur_pose
            cur_pose = torch.tensor(cur_pose.squeeze(0)).cuda()

            cam_int = self.K_b33.repeat(1, 1, 1)  # [b,3,3]

            #load cam_ext
            cam_ext = c2w
            cam_ext_inv = torch.inverse(cam_ext)
            cam_ext = cam_ext.repeat(1, 1, 1)[:,:-1,:]
            cam_ext_inv = cam_ext_inv.repeat(1, 1, 1)[:,:-1,:]

            rgbd = torch.cat([ref_img, ref_depth], dim=1).cuda()
            cam_int = cam_int.cuda()
            cam_ext = cam_ext.cuda()
            cam_ext_inv = cam_ext_inv.cuda()

            # warp to a random novel view
            self.rgbd_to_mesh(ref_img, ref_depth.squeeze(0).squeeze(0), self.world_to_cam)
            warp_image, _ = self.project(cur_pose)
            warp_disp = self.rendered_depth
            warp_mask = ~self.inpaint_mask
            self.empty_mesh()

            # warp back to the original view
            self.rgbd_to_mesh(warp_image[:3, ...].unsqueeze(0), warp_disp, cur_pose)
            warp_back_image, _ = self.project(self.world_to_cam)
            warp_back_disp = self.rendered_depth
            warp_back_mask = ~self.inpaint_mask            
            self.empty_mesh()

            # filter occlusion: warp_back_depth should not be smaller that ref_depth
            margin = 0.1
            occlusion_mask = ((warp_back_disp * ~self.inpaint_mask + margin) < 
                              (ref_depth.squeeze(0).squeeze(0) * ~self.inpaint_mask))
            warp_back_image *= ~occlusion_mask
            warp_back_mask *= ~occlusion_mask

            ref_depth_2 = ref_depth
            # all depth should be in [0~1]
            inpaint_pairs.append((ref_img, ref_depth_2, cur_pose,
                                    warp_image, warp_disp, warp_mask,
                                    warp_back_image, warp_back_disp, warp_back_mask))

        return inpaint_pairs

    def run(self):
        # load pano and project to tangent views
        panorama_tensor, init_depth = self.load_pano()
        
        cubemaps_pitch_yaw = [(0, 0), (0, 1/2 * np.pi), (0, 1 * np.pi), (0, 3/2 * np.pi),\
                                (1/2 * np.pi, 0), (-1/2 * np.pi, 0)]

        pitch_yaw_list = cubemaps_pitch_yaw
      
        view_rgb_depth_pairs = []
        for view_idx, (pitch, yaw) in enumerate(pitch_yaw_list):
            view_rgb = self.pano_to_perpective(panorama_tensor, pitch, yaw)
            view_depth = self.pano_to_perpective(init_depth.unsqueeze(0).unsqueeze(0), pitch, yaw)
            view_rgb_depth_pairs += [(view_rgb, view_depth)]

            view_rgb_pil = Image.fromarray((view_rgb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
            view_rgb_pil.save(f"{self.save_path}/view_rgb_{view_idx}.png")    

            
        # create pseudo <masked image, GT image> pairs using warp-back strategy
        video_frames = []
        for view_idx, (view_rgb, view_depth) in enumerate(view_rgb_depth_pairs):
            print(f"-view_idx:{view_idx}")
            pairs_per_view = 5
            inpaint_pairs = self.get_pairs(view_rgb, view_depth, pairs_per_view)
            for pair_idx, inpaint_pair in enumerate(inpaint_pairs):
                (ref_img, ref_depth, cur_pose,
                warp_rgb, warp_disp, warp_mask, 
                warp_back_image, warp_back_disp, warp_back_mask)  = inpaint_pair
                warp_back_mask = ~warp_back_mask

                ref_rgb_pil = Image.fromarray((ref_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
                ref_depth_pil = Image.fromarray(visualize_depth_numpy(warp_disp.squeeze(0).squeeze(0).cpu().detach().numpy())[0].astype(np.uint8))

                warp_rgb_pil = Image.fromarray((warp_rgb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
                warp_mask_pil = Image.fromarray(warp_mask.squeeze(0).squeeze(0).detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
            
                warp_back_rgb_pil = Image.fromarray((warp_back_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
                warp_back_mask_pil = Image.fromarray(warp_back_mask.squeeze(0).squeeze(0).detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

                ref_rgb_pil.save(f"{self.save_path}/ref_rgb_{view_idx}_{pair_idx}.png")
                warp_back_rgb_pil.save(f"{self.save_path}/warp_back_rgb_{view_idx}_{pair_idx}.png")
                warp_back_mask_pil.save(f"{self.save_path}/warp_back_mask_{view_idx}_{pair_idx}.png")


pipeline = PanoWarp()
pipeline.run()