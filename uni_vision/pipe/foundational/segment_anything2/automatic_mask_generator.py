# https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area # type: igonre

from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import(
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    unxcrop_points,
)



class SAM2AutomaticMaskGenerator:

    def __init__(
        self,
        model: SAM2Base,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_ofset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlay_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        multimask_output: bool = True,
        **kwargs
    ):
        """
        Usig a SAM2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM2 with a HieraL backbone

        Args:
            model (SAM2Base): The SAM2 Model used for mask prediction
            points_per_side (int, None): The number of points to be sampled
                                         along one side of the image. The total number of 
                                         points is points_per_side **2. If None, 'point_grids' 
                                         must provide explicit point sampling.
            points_per_batch (int): Set the number of points run simultaneously by the
                                    model. Higher number may be faster but uses more GPU memory.
            pred_iou_thresh (float): A filtering threshold in [0, 1], using the stability of the mask
                                     under changes to the cutoff used to binarize the model's mask predictions.
            stability_score_thresh (float): A filtering threshold [0, 1], using the stability of the mask under changes
                                            to the cutoff used to binarize the model's mask predictions.
            stability_score_ofset (float): The amount to shift the cutoff when calculated the stability score.
            
            mask_threshold (float): Threshold for binarizing the mask logits
            box_nms_thresh (float): The box IoU cutoff used by non-maximal suppression
                                    to filter duplicate masks.
            crop_n_layers (int):  If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run,
                                  where each layer has 2**i_layer number of image crops.
            crop_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
            crop_overlay_ratio (float): Sets the degree to which crops overlap..
            crop_n_points_downscale_factor (int) : The number of points-per-side sampled in layer n is scaled down by 
                                                    crop_n_points_downscale_factor**n.
            point_grids (List[np.ndarray or None): A list over explicit grids of points used for sampling, normalized to [0, 1]. 
                                                    The nth gid in the list is used in the nth crop layer. Exclusive with points_per_side.
            min_mask_region_area (int): If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller
                                        than min_mask_region_area. Requires opencv
            output_mode (str):   The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle' requires pycocotools
            use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
            multimask_output (bool): Whether to output multimask at each point of the grid.
        """
        
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided"
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        
        else:
            raise ValueError("Can't have both point_per_side and point_grid be None")
        
        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle"
        ], f"Unknown output mode {output_mode}"
        
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils # type: ignore # noqa: F401

            except ImportError as e:
                print("Please install pycocotools")
                raise e
            
            
        self.processor = SAM2ImagePredictor(
            model,
            max_hole_area = min_mask_region_area,
            max_sprinkle_area = min_mask_region_area
        )
                
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_ofset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlay_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output
        
        
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        Load a pretrained model from Hugging Face Hub

        Args:
            model_id (str): The Hugging Face repositry ID.
            **kwargs: Additional arguments to pass to the model constructor.
        Returns:
            SAM2AutomaticMaskGenerator: The loaded model
        """
        
        from sam2.build_sam import build_sam2_hf
        
        sam_model  = build_sam2_hf(model_id, **kwargs)
        
        return cls(sam_model, **kwargs)
    

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image

        Args:
            image (np.ndarray): Image in HWC uint8 format

        Returns:
            List[Dict[str, Any]]: A list over records for masks. 
            Each record is a dict containing following keys:
                segmentation (dict(str, any) or np.ndarray): The mask. If output_mode = 'binary_mask',
                                                            is an array of shape HW. Otherwise its a 
                                                             dictionary containing RLE.
                bbox (list(float)): The box around the mask, in XYWH format.
                area (int): The area in pixels of the mask.
                predicted_iou (float): The model's own prediction of the mask's
                                        quality. This is filtered by the pred_iou_thresh parameter.
                point_coords (list(list(float))): The point coordinates input to the model to 
                                                  generate the mask.
                stability_score (float): A measure of the mask's quality. This is filtered on using
                                         the stability_score_thresh parameter.
                                         
                crop_box (list(float)): The crop of the image used to generate the mask, given in XYWH format.
        """
        
        # Generate masks
        mask_data = self._generate_masks(image)
        
        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
              coco_encode_rle(rle)  for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentatons"]  = [rle_to_mask(rle) for rle in mask_data["rles"]]
        
        else:
            mask_data["segmentations"] == mask_data["rles"]
            
        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                
            }