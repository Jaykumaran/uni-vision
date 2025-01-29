import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from vlms.models.molmo.molmo_utils import overlay_points_on_image

from datetime import datetime

#Ref: github.com/sovit-123/SAM_Molmo_Whisper/


# ************ TIME ****************
now = datetime.now()
current_time = now.strftime("%H_%M_%S")


# *********************** SEGMENT WITH POINTS ******************************
def segment_with_points(image, points, show_pts = True, sam_model_id = "facebook/sam2.1-hiera-large",):
    
    points = np.array(points)
    
    point_lables = np.ones(len(points), dtype = np.uint8)

    predictor = SAM2ImagePredictor.from_pretrained(sam_model_id)
    
    with torch.inference_mode():
        predictor.set_image(image)
        
        masks, scores, logits = predictor.predict(
            point_coords= points,
            point_labels= point_lables,
            multimask_output=False #return three masks -> two partial and full mask of the instance
        )
        
        
    sam_output = show_mask(image, masks, scores, borders=True)
    
    r = int(image.shape[0] * 0.0007)  #size of the point
    
    if show_pts is not None:
        final_image = overlay_points_on_image(
            sam_output, points, radius=r, color = (0, 0, 255, 0.6)
        )
        
        dpi = plt.rcParams['figure.dpi']
        
        figsize = image.shape[1] / dpi, image.shape[0] / dpi
        plt.figure(figsize=figsize)
        plt.imshow(final_image)
        
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"sam_molmo_outputs/molmo_sam_pt_{current_time}.jpg", 
                    bbox_inches = 'tight', pad_inches = 0)
        plt.show()





#****************** SINGLE MASK **************************
def show_mask(mask, base_image, color = (1.0, 40/255, 50/255, 0.6), borders = True):
    
    """
    Returns an image with overlaid on the base_image
    
    Params:
    - mask: 2D numpy array of shape (H,W), where non-zero values indicate the mask.
    - base_image: 3D numpy array of shape (H,W,3) the original image.
    - colors: Tuple of (R, G, B, A) where A is the alpha transparency
    - borders: Boolean indicating whether to draw contours around the mask.
    
    Returns:
        blended: 3D numpy array of shape (H, W, 3), the image with mask overlay
    """
    
    #Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    h, w = mask.shape
    
    #Extract RGB and alpha components
    overlay_color = np.array(color[:3], dtype = np.float32) #RGB without opacity
    alpha = color[3] #opacity
    
    #Normalize base image to [0,1]
    base_image_norm = base_image.astype(np.float32) / 255.0
    
    #Create an empty overlay image
    overlay = np.zeros_like(base_image_norm)
    
    #Assign the overlay color to the masked regions
    #Using the 2D Mask to index the first two dimensions
    overlay[mask == 1] = overlay_color
    
    #Blend the overlay with the base image only where the mask is present
    blended  = base_image_norm.copy()
    #apply color tp only areas that has 1s with alpha transparency
    blended[mask == 1] = (
        alpha * overlay[mask == 1] + (1 - alpha) * base_image_norm(mask == 1)
    )
    
    #Convert back to [0, 255] uint8
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    #If borders are flagged True, draw contours on the blended image
    if borders:
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Draw contours in blue
        cv2.drawContours(blended, contours , -1,(255, 0, 0), 2) 
        
    return blended


# *********** Show all / multiple masks *****************
    
def show_masks(image, masks, scores, borders = True):
    
    """
    Overlays the top mask (based on highest score)
    
    Parameters:
    - masks: List or array of masks
    - scores: Lisy or array of scores corresponding to each mask
    - borders: default = True, sraw a contour to the mask outline.
    """
    
    sort_idxs = np.argsort(scores)[::-1] #descending order - high prob score to low
    sorted_masks = masks[sort_idxs]
    scores = scores[sort_idxs]
    
    
    if len(masks) == 0:
        print("No valid masks found")
        return image
    
    top_mask = sorted_masks[0]
    segment_image = show_mask(top_mask, image, borders = True)
    
    
    return segment_image



    
    