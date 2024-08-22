from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def rot_x_world_to_cam(degree=45):
    angle_rad = np.deg2rad(degree) 
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotation_x = torch.Tensor([[1, 0, 0, 0],
                        [0, cos_theta, -sin_theta, 0],
                        [0, sin_theta, cos_theta, 0],
                        [0, 0, 0, 1]])
    return rotation_x

def rot_y_world_to_cam(degree=45):
    angle_rad = np.deg2rad(degree) 
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotation_y = torch.Tensor([[cos_theta, 0, sin_theta, 0],
                        [0, 1, 0, 0],
                        [-sin_theta, 0, cos_theta, 0],
                        [0, 0, 0, 1]])
    return rotation_y

def rot_z_world_to_cam(degree=45):
    angle_rad = np.deg2rad(degree) 
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotation_z = torch.Tensor([[cos_theta, sin_theta, 0, 0],
                            [-sin_theta, cos_theta, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    return rotation_z

def resize_image_with_aspect_ratio(image, new_width=2048):
    original_width, original_height = image.size
    new_height = int((original_height / original_width) * new_width)
    resized_image = image.resize((new_width, new_height))
    return resized_image
    
def add_padding(image, top_pad=0, bottom_pad=0, left_pad=0, right_pad=0):
    padded_image = ImageOps.expand(image, (left_pad, top_pad, right_pad, bottom_pad))
    return padded_image


def get_cubemap_views_world_to_cam():
    world_to_cam_list = []

    # horizontal
    rot_angle = 270 # 'F' in equilib
    world_to_cam = rot_y_world_to_cam(rot_angle)
    world_to_cam_list.append(world_to_cam)

    world_to_cam = torch.eye(4, dtype=torch.float32)
    world_to_cam_list.append(world_to_cam)   

    rot_angle = 90
    world_to_cam = rot_y_world_to_cam(rot_angle)
    world_to_cam_list.append(world_to_cam)

    rot_angle = 180
    world_to_cam = rot_y_world_to_cam(rot_angle)
    world_to_cam_list.append(world_to_cam)

    # ceiling
    world_to_cam = rot_x_world_to_cam(90)  @ rot_y_world_to_cam(270)
    world_to_cam_list.append(world_to_cam)

    # floor
    world_to_cam = rot_x_world_to_cam(-90) @ rot_y_world_to_cam(270)
    world_to_cam_list.append(world_to_cam)
    return world_to_cam_list


def pil_to_tensor(pil):
    return torch.Tensor(np.asarray(pil)).permute(2,0,1).unsqueeze(0).float()/255 #BCHW

def tensor_to_pil(tensor): # BCHW
    tensor = (tensor.detach().cpu().numpy()*255).astype(np.uint8)
    pil = Image.fromarray(tensor.squeeze(0).transpose(1, 2, 0))
    return pil


def normalize(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor


def depth_to_distance(depth_image, fx=256, fy=256, cx=256, cy=256):
    # Calculate grid of x and y coordinates
    grid_x, grid_y = torch.meshgrid(torch.arange(depth_image.size(2)), torch.arange(depth_image.size(1)))
    
    # Convert to float and move to the same device as depth_image
    grid_x = grid_x.float().to(depth_image.device)
    grid_y = grid_y.float().to(depth_image.device)
    
    # Calculate distance
    distance_image = depth_image * torch.sqrt(((grid_x - cx) / fx) ** 2 + ((grid_y - cy) / fy) ** 2 + 1)
    return distance_image


def distance_to_depth(depth_image, fx=256, fy=256, cx=256, cy=256):
    # Calculate grid of x and y coordinates
    grid_x, grid_y = torch.meshgrid(torch.arange(depth_image.size(2)), torch.arange(depth_image.size(1)))
    # Convert to float and move to the same device as depth_image
    grid_x = grid_x.float().to(depth_image.device)
    grid_y = grid_y.float().to(depth_image.device)
    # Calculate distance
    distance_image = depth_image / torch.sqrt(((grid_x - cx) / fx) ** 2 + ((grid_y - cy) / fy) ** 2 + 1)
    return distance_image


def colorize_single_channel_image(image, color_map=cv2.COLORMAP_JET):
    '''
    return numpy data
    '''
    image = image.squeeze()
    assert len(image.shape) == 2

    image = image * 255
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    image = image.astype(np.uint8)

    image = cv2.applyColorMap(image, color_map)

    return image


def write_video(out_path, images, fps=30, library='imageio'):
    assert out_path[-3:] == 'mp4'
    if library == 'imageio':
        import imageio
        writer = imageio.get_writer(out_path, fps=fps)
        for image in images:
            writer.append_data(image)
        writer.close()
    else:
        # Use OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps,
            (images[0].shape[1], images[0].shape[0]))
        for image in images:
            writer.write(image)
        writer.release()


def write_image(out_path, image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    assert (len(image.shape) == 3 and image.shape[-1] in [1, 3]) or len(image.shape) == 2
    if len(image.shape) == 3:
        image = image[:,:,::-1].copy()
    assert(cv2.imwrite(out_path, image))
