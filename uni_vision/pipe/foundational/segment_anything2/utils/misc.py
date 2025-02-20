# https://github.com/facebookresearch/sam2/blob/main/sam2/utils/misc.py


import os
import warnings
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm



def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention to Ampere (8.0)  or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
            
        
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # availble on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2]) 
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Considering upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be better).",
                category=UserWarning,
                stacklevel=2
            )
        
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
        
    else:
        old_gpu = True
        use_flash_attn = True
        math_kernel_on = True
    return old_gpu, use_flash_attn, math_kernel_on



def get_connected_components(mask):
    """Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W)

    Args:
    - mask : A binary mask tensor of shape (N, 1, H, W) where 1 is the foreground and 0 is bg
        
    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels for
              foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected components
              
    """
            
    from sam2 import _C 
    
    return  _C.get_connected_components(mask.to(torch.uint8).contiguous())

def mask_to_box(masks: torch.Tensor):
    """
    Compute bounding box given an input mask.

    Args:
    - masks: [B, 1, H, W]  masks, dtype = torch.Tensor
    
    Returns:
    - box_coords: [b, 1, 4], contains (x, y) coordinates of top left and bottom right corners, dtype = torch.Tensor
    """
    B, _, h, w = masks.shape
    device  = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim = -1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim = -1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim = -1)
    
    return bbox_coords

def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8: # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size # the original video size
    
    return img, video_height, video_width


    