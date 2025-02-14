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
    uncrop_boxes_xyxy,
    uncrop_masks,
    unxcrop_points
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
            min_mask_region_area (int): If >0, postpreocessing will be applied to remove disconnected regions and holes in masks with area smaller
                                        than min_mask_region_area. Requires opencv
            output_mode (str, optional): 
            use_m2m (bool, optional): 
            multimask_output (bool, optional):
        """