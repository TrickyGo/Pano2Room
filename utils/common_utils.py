import cv2 as cv
import torch
import numpy as np
import os
from struct import unpack
from os.path import join as pjoin
import cv2
from PIL import Image

def read_dpt(dpt_file_path):
    """read depth map from *.dpt file.
    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    # depth_data = depth_data/depth_data.max()           ########## lwqwgc
    fid.close()

    return depth_data


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
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(out_path, fourcc, fps,
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
    assert(cv.imwrite(out_path, image))


def read_image(in_path, squeeze=True, to_torch=True, channel_first=False, factor=1):
    img = cv.imread(in_path)[:,:,::-1].copy()
    if factor != 1:
        h, w, _ = img.shape
        h, w = h // factor, w // factor
        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
    if squeeze:
        img = img.astype(np.float32) / 255.
    if to_torch:
        img = torch.from_numpy(img)
    if channel_first:
        assert to_torch
        img = img.permute(2, 0, 1)

    return img


def colorize_single_channel_image(image, color_map=cv.COLORMAP_JET):
    '''
    return numpy data
    '''
    image = image.squeeze()
    assert len(image.shape) == 2

    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    image = image.astype(np.uint8)

    image = cv.applyColorMap(image, color_map)

    return image


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_TURBO):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        if (x > 0).any():
            mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
            ma = np.max(x)
        else:
            mi = 0.0
            ma = 0.0
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]

def save_rgbd(image, depth, prefix, idx, outdir):
    filename = f"{idx:04}_{prefix}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    dst = Image.new('RGB', (image.width + depth.width, image.height))
    dst.paste(image, (0, 0))
    dst.paste(depth, (image.width, 0))
    dst.save(file_out)
    return dst, file_with_ext