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
        self.is_image_set = False
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

        
    @torch.no_grad()    
    def set_image_batch(
        self,
        image_list: List[np.ndarray],
        ) -> None:
        
        """
          Calculates the image embeddings for the provided image batch, allowing masks to be predicted with the predict_batch method.
        
        Arguments:
            image (List[np.ndarray]): in RGB format. The Image should be in HWC format if np.ndarray or WHC format if PIL Image
                                                            with pixel values in [0, 255]
        """
        
        self.reset_predictor()
        assert isinstance(image_list, list) # to check whether its a list type
        self._origh_hw = []
        
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape HWC"
            self._origh_hw.append(image.shape[:2])
            
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert(
            len(img_batch.shape) == 4 and img_batch.shape[1] ==3 
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing the image embedddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_features, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest resolution feature map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_features[-1] = vision_features[-1] + self.model.no_mem_embed
        
        features = [
            feature.permute(1, 2, 0).view(batch_size, -1, *feature_size)
            for feature, feature_size in zip(vision_features[::-1], self._bb_feature_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": features[-1], "high_res_features": features[:-1]}
        self.is_image_set = True
        self._is_batch = True
        logging.info("Image embedding computed")
    
    
    
    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords = True
         ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        
        """
        This function is similar to predict(...), however it is used in batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        
        
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction"
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        
        for img_idx in range(num_images):
            
            # Transforms input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx = img_idx
            )
            
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits = return_logits,
                img_idx = img_idx
            )
            
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(all_low_res_masks)
            
        return all_masks, all_ious, all_low_res_masks
        
    
        
        


    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Predicts masks for the given input prompts, using the currently set image
        
        Arguments:
            point_coords (np.ndarray or None): A Nx2 array of point prompts to the model.Each point is in (X,Y) in pixels.
            
            point_labels (np.ndarray or None): A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a bg point.
            
            box (np.ndarray or None): A length 4 array given a box prompt to the model, XYXY format.
            
            mask_input (np.ndarray): A low resolution mask input to the model, typically coming from  previous iteration. Has form 1xHxW where for SAM, H=W=256.
            
            multimask_output (bool): If true, the model will return three masks. For ambiguous input prompts (such as single click), this often will produce 
                                     better masks than a single prediction. If only a single mask is needed, the model's predicted quality score can be used to select 
                                     the best mask. For non-ambiguous prompts, such as multiple input prompts, multimask_output = False can give better results.
            return_logits (bool): If true, returns un-thresholded mask logits instead of binary mask.
            
            normalize_coords (bool): If true, the point coordinates will be normalized to the range [0, 1] and point_coords is expected to be w.r.t image dimensions
            
    
        Returns:
        
            (np.ndarray): The output masks in CxHxW format, where C is the number of masks, and (H, W) is the original image size.
            
            (np.ndarray): An array of length C containing the model's predictions for the quality of each mask. # IoU
            
            (np.ndarray): An array of shape CxHxW, where C is the number of masks and H=W=256. These low resolution logits can be passed to a subsequent iteration
                          as mask input.
            
        """
        
        if not self._is_image_set:      
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction"
            )
        
        # Transform input prompts
        
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )
        
        
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output, 
            return_logits = return_logits
        )
        
        
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            
        return masks_np, iou_predictions_np, low_res_masks
        
        
        def _prep_prompts(
            self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx = -1
        ):
            unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
            if point_coords is not None:
                assert(
                    point_labels is not None # check if point labels i.e. 0, 1 is provided when point coords are passed
                ), "point_labels must be supplied if point_coords is applied."
                
                point_coords = torch.as_tensor(
                    point_coords, dtype = torch.float, device = self.device
                )
                
                unnorm_coords = self._transforms.transform_coords(
                    point_coords, normalize = normalize_coords, orig_hw = self._orig_hw[img_idx]
                )
                
                labels = torch.as_tensor(
                    point_labels, dtype=torch.int, device=self.device
                )
                
                if len(unnorm_coords.shape) == 2:
                    unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
                    
            
            if box is not None:
                box = torch.as_tensor(box, dtype = torch.float, device= self.device)
                unnorm_box = self._transforms.transform_boxes(
                    box, normalize = normalize_coords, orig_hw = self._orig_hw[img_idx]
                ) # Bx2x2
            
            if mask_logits is not None:
                mask_input = torch.as_tensor(
                    mask_logits, dtype=torch.float, device=self.device
                )
                
                if len(mask_input.shape) == 3:
                    mask_input = mask_input[None, :, : , :]
            
            return mask_input, unnorm_coords, labels, unnorm_box