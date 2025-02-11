# http://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py

import logging

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from PIL.Image import Image
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
from sam2.build_sam import build_sam2_hf


class SAM2ImagePredictor:
    def __init__(self, 
                 sam_model: SAM2Base,
                 mask_threshold = 0.0,
                 max_hole_area = 0,
                 max_sprinkle_area = 0,
                 **kwargs
                 ):
        """Uses SAM-2 to calculate image embedding for an image, and then allow repeated, efficient mask prediction given prompts.

        Args:
            sam_model (SAM2Base): The model to use for Mask Prediction
            mask_threshold (float): The threshold to use when converting mask logits.
            max_hole_area (int): If max_hole_area > 0, we will fill small holes in up to the max
                                 area of max_hole_area in low_res_masks.
            max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles upto the maximum area of 
                                    max_sprinkle_area in low_res_masks.
        """
        
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution = self.image_size,
            mask_threshold = mask_threshold,
            max_hole_area = max_hole_area,
            max_sprinkle_area = max_sprinkle_area
        )
        
        
        # Predictor state
        self.is_image_ste = False
        self._features = None
        self._origh_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False
        
        # Predictor config
        self.mask_threshold = mask_threshold
        
        # Spatial dim for backbone feature maps
        self._backbone_feature_sizes = [
            (256, 256),
            (128, 128),
            (64, 64)
        ]
        
    @classmethod    # directly operates on class level rather than on a instance.
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        
        """
        Load a pretrained model from HuggingFace Hub
        
        Returns:
            (SAM2ImagePredictor): The loaded model
        
        """
        
        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)
    
    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image]
    ) -> None:
        
        """
        Calculates the image embeddings for the provided image, allowing masks to be predicted with the predict method.
        
        Arguments:
            image (np.ndarray or PIL Image): in RGB format. The Image should be in HWC format if np.ndarray or WHC format if PIL Image
                                                            with pixel values in [0, 255]
            image_format(str): 'BGR' or 'RGB'
        """
        
        self.reset_predictor()
        
        # Transform image to model compatibility
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._origh_hw = [image.shape[:2]]
        
        else:
            raise NotImplementedError("Image not Supported")
    
        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)
        
        assert(
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_features, _, _ = self.model._prepare_backbone_features(backbone_out)
        
        # Add no_mem_embed, which is added to the lowest resolution feature map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_features[-1] = vision_features[-1] + self.model.no_mem_embed
            
            
        feats = [
            feature.permute(1, 2, 0).view(1, -1, *feature_size)
            for feature, feature_size in zip(vision_features[::-1], self._backbone_feature_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed")

        
        
        
        
        
        