#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import json
from typing import NamedTuple
from pathlib import Path

import imageio
import torch
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.gaussian_model import BasicPointCloud
from scene.cameras import MiniCam, Camera
from utils.graphics import getWorld2View2, focal2fov, fov2focal
from utils.graphics import getProjectionMatrix
from utils.trajectory import get_camerapaths
from utils.sh import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    preset_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam, mesh_pose in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}



def loadCamerasFromData(traindata, white_background):
    cameras = []

    # fovx = traindata["camera_angle_x"]
    frames = traindata["frames"]
    for idx, frame in enumerate(frames):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        fovx = np.array(frame["fovx"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = frame["image"] if "image" in frame else None
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        loaded_mask = np.ones_like(norm_data[:, :, 3:4])

        fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
        FovY = fovy 
        FovX = fovx

        image = torch.Tensor(arr).permute(2,0,1)
        loaded_mask = None 
        
        cameras.append((Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, 
                                gt_alpha_mask=loaded_mask, image_name='', uid=idx, data_device='cuda'), frame['mesh_pose']))
            
    return cameras


def loadCameraPreset(traindata, presetdata):
    cam_infos = {}
    ## camera setting (for H, W and focal)
    fovx = traindata["camera_angle_x"] * 0.75
    # W, H = traindata["frames"][0]["image"].size
    W, H = traindata["W"], traindata["H"]

    for camkey in presetdata:
        cam_infos[camkey] = []
        for idx, frame in enumerate(presetdata[camkey]["frames"]):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            fovy = focal2fov(fov2focal(fovx, W), H)
            FovY = fovy 
            FovX = fovx
        
            znear, zfar = 0.01, 100
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

            cam_infos[camkey].append(MiniCam(width=W, height=H, fovy=FovY, fovx=FovX, znear=znear, zfar=zfar,
                                          world_view_transform=world_view_transform, full_proj_transform=full_proj_transform))
            
    return cam_infos


def readDataInfo(traindata, white_background):
    print("Reading Training Transforms")
    
    train_cameras = loadCamerasFromData(traindata, white_background)
    preset_minicams = loadCameraPreset(traindata, presetdata=get_camerapaths())
    
    nerf_normalization = getNerfppNorm(train_cameras)

    pcd = BasicPointCloud(points=traindata['pcd_points'].T, colors=traindata['pcd_colors'], normals=None)

    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cameras,
                           test_cameras=[],
                           preset_cameras=preset_minicams,
                           nerf_normalization=nerf_normalization,
                           ply_path='')
    return scene_info
