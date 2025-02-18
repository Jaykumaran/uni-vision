# https://github.com/facebookresearch/sam2/blob/main/sam2/utils/transforms.py  

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2Transforms(nn.Module):
    def __init__(self,
                 resolution, 
                 mask_threshold,
                 max_hole_area = 0.0, 
                 max_sprinkle_area = 0.0):
        
        """  
        Transforms for SAM2
        """
        
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std)
            )
        )
        
        
    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)
    
    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim = 0)
        
        return img_batch
    
    
    def transform_coords(
        self, coords: torch.Tensor, normalize = False, orig_hw = False
    ) -> torch.Tensor:
        
        """ 
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        
        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 Model
        """
        
        
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 0] = coords[..., 1] / h
        
        coords = coords * self.resolution # unnormalize coords
        return coords
    
    
    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """ 
        Perform PostProcessing on output masks
        """
        
        from sam2.utils.misc import get_connected_components
        
        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1) # Flatten as 1-channel image
        
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in the background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # As of missing holes are in bg by just adding a positive score it can be considered as fg
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)  # condition, input, others --> wherever condition is not met it fills the others value
            
            if self.max_sprinkle_area >  0:
               labels, areas = get_connected_components(
                   mask_flat > self.mask_threshold
               )
               
               is_hole = (labels > 0)  & (areas <= self.max_hole_area)
               is_hole = is_hole.reshape_as(masks)
               # As of islands are in fg by just adding a negative score it can be considered as bg
               # We will holes with a small negative mask score (10.0) to change them to background
              
        
        except Exception as e:
            
             # Skip the post-processing step if the CUDA kernel fails
             warnings.warn(
                 f"{e}\n\n Skipping the post-processing step due to the error above. You can"
                 "still use SAM2 and it's OK to ignore the error above, although some post-processing"
                 "functionality may be limited (which doesn't affect the results in most cases; see)"
                 "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                 category = UserWarning,
                 stacklevel=2
             )
             
             masks = input_masks
        
        masks = F.interpolate(masks, orig_hw, mode = "bilinear", align_corners=False)
        return masks
            
        
        
    
        
    
    
    