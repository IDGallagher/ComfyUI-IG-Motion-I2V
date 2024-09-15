import hashlib
import os
from typing import Iterable
import re

import numpy as np
import torch

def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions: Iterable=None):
    directory = directory.strip()
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files

# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    h = hashlib.sha256()
    b = bytearray(10*1024*1024) # read 10 megabytes at a time
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        i = 0
        # don't hash entire file, only portions of it if requested
        while n := f.readinto(mv):
            if i%hash_every_n == 0:
                h.update(mv[:n])
            i += 1
    return h.hexdigest()

def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Replace only whole words to avoid partial replacements
        new_key = re.sub(r'\bquery\b', 'to_q', new_key)
        new_key = re.sub(r'\bkey\b', 'to_k', new_key)
        new_key = re.sub(r'\bvalue\b', 'to_v', new_key)
        new_key = re.sub(r'\bproj_attn\b', 'to_out.0', new_key)
        new_state_dict[new_key] = value
    return new_state_dict

def print_loading_issues(missing_keys, unexpected_keys):
    if missing_keys:
        print(f"Missing keys when loading state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys when loading state_dict: {unexpected_keys}")


# The following functions are adapted from the implementation by Tom Runia
# under the MIT License. Original code can be found at:
# https://github.com/tomrunia/OpticalFlow_Visualization

def make_colorwheel(device):
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    Returns:
        torch.Tensor: Color wheel tensor of shape [ncols, 3], values in [0, 255]
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3), device=device)

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY, device=device) / RY)
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG, device=device) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = torch.floor(255 * torch.arange(0, GC, device=device) / GC)
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - torch.floor(255 * torch.arange(0, CB, device=device) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = torch.floor(255 * torch.arange(0, BM, device=device) / BM)
    col += BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - torch.floor(255 * torch.arange(0, MR, device=device) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel  # Shape: [ncols, 3]

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to flow components u and v.
    Args:
        u (torch.Tensor): Horizontal flow of shape [N_frames, H, W]
        v (torch.Tensor): Vertical flow of shape [N_frames, H, W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        torch.Tensor: Flow visualization images of shape [N_frames, H, W, 3], values in [0,1]
    """
    device = u.device
    n_frames, H, W = u.shape
    flow_image = torch.zeros(n_frames, H, W, 3, device=device)
    colorwheel = make_colorwheel(device)  # Shape: [ncols, 3]
    ncols = colorwheel.shape[0]

    rad = torch.sqrt(u ** 2 + v ** 2)
    a = torch.atan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1 = k1 % ncols  # Wrap around
    f = fk - k0.float()

    for i in range(3):  # For R, G, B channels
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col = torch.where(idx, 1 - rad * (1 - col), col * 0.75)

        if convert_to_bgr:
            ch_idx = 2 - i
        else:
            ch_idx = i
        flow_image[..., ch_idx] = col

    return flow_image.clamp(0, 1)  # Shape: [N_frames, H, W, 3]

def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Converts flow UV images to RGB images.
    Args:
        flow_uv (torch.Tensor): Flow UV images of shape [N_frames, H, W, 2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        torch.Tensor: RGB images of shape [N_frames, H, W, 3], values in [0,1]
    """
    assert flow_uv.dim() == 4 and flow_uv.size(3) == 2, 'input flow must have shape [N_frames, H, W, 2]'
    device = flow_uv.device
    if clip_flow is not None:
        flow_uv = torch.clamp(flow_uv, max=clip_flow)

    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    rad = torch.sqrt(u ** 2 + v ** 2)
    rad_max = rad.max()
    epsilon = 1e-10
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_uv_to_colors(u, v, convert_to_bgr)