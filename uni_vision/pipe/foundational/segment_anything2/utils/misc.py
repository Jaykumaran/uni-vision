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



def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size # the original video size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of videos frames to be loaded asynchronously without blocking session start.
    """
    
    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device
        
        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)
        
        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc = "frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e
        
        
        
        self.thread = Thread(target=_load_frames, daemon= True)
        self.thread.start()
        
    
    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception
        
        img = self.images[index]
        if img is not None:
            return img
        
        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img
        return img
        
    def __len__(self):
        return len(self.images)


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean = (0.485, 0.456, 0.406),
    img_std = (0.229, 0.224, 0.225),
    async_loading_frames = False,
    compute_device = torch.device("cuda")
):
    """ 
    Load the video frames in the video path. The frames are resized to image_size as in the model
    and are loaded to GPU if offload_video_to_cpu = False. This is used by the demo,
    """
    
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4"]
    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path = video_path,
            image_size = image_size,
            offload_video_to_cpu = offload_video_to_cpu,
            img_mean = img_mean,
            img_std = img_std,
            compute_device = compute_device
        )
        
    elif is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            video_path = video_path,
            image_size = image_size,
            offload_video_to_cpu = offload_video_to_cpu,
            img_mean = img_mean,
            img_std = img_std,
            compute_device = compute_device
        )
        
    else:
        raise NotImplementedError(
            "Only MP4 Video and JPEG Folder are supported at this moment."
        )
        
def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean = (0.485, 0.456, 0.406),
    img_std = (0.229, 0.224, 0.225),
    async_loading_frames = False,
    compute_device = torch.device("cuda")
):
    
    """ 
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format)
    
    The frames are resized to image_size x image_size and are loaded to GPU if `offload_video_to_cpu` is `False` 
    and to CPU if `offload_video_to_cpu` is `True`.
    
    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError(
            "Only JPEG frames are supporgted at this moment. For video files, you may use"
            "ffmpeg (https://ffmpeg.org/) to extract frames into folder of JPEG files, such as \n"
            "```\n"
            "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
            "```\n"
            "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
            "ffmpeg to start the JPEG file from 00000.jpg"
        )
    
    
    frames_names = [
        p
        for p in os.listdir(jpg_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    
    frames_names.sort(key = lambda p: int(os.path.splitext(p)[0]))
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype = torch.float32)[:, None, None]
    
    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc = "frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    
    # normalize by mean and std
    images -= img_mean                                                 
    images /= img_std     
    return images, video_height, video_width
         
         
                
            
                 
        

    