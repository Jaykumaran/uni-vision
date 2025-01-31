import torch
import torch.nn.functional as F
from typing import Tuple



def mean_iou(predictions: torch.tensor, ground_truths: torch.tensor, num_classes: int = None, dims: Tuple = (1,2)):
    """
    Args:
    
    predctions : Pred from model with or without softmax
    
    dims : Dims correspond of image height and width
    
    Returns:
    A scalar tensor representing the Classwise Mean IOU Metric
    
    """
    
    #Convert single channels gt masks into one hot encoded vector
    #Shape: [B, H, W] --> [B,H, W, num_classes]
    ground_truths = F.one_hot(ground_truths, num_classes=num_classes)
    
    #Convert unnormalized predictions into one hot encoded across channels
    #Shape: [B, H, W] --> [B, H, W, num_classes]
    predictions = F.one_hot(predictions, num_classes=num_classes)
    
    #Intersection: |G ∩ P| Shape: [B, num_classes]
    intersection = (predictions * ground_truths).sum(dims = dims)
    
    #Summation: |G| + |P| Shape: [B, num_classes]
    summation = (predictions.sum(dims = dims))
    
    #Union. Shape: [B, num_classes]
    union = summation - intersection
    
    #IoU Shape: [B, num_classes]
    iou = intersection / union
    
    iou = torch.nan_to_num_(iou, nan = 0.0)
    
    #Shape: [batch_size, ]
    num_classes_present = torch.count_nonzero(summation, dim = 1)
    
    #IoU per image
    #Average over the total number of classes present in the gt and pred
    #Shape: [batch_size, ]
    iou = iou.sum(dim = 1) / num_classes_present
    
    #Compute mean over remaininh axes (batch and classes)
    #Shape: Scalar
    iou_mean = iou.mean()
    
    return iou_mean



#Dice + CE Loss

def dice_coef_loss(predictions: torch.tensor, ground_truth: torch.tensor, num_classes: int = None, dims = (1, 2), smooth = 1e-8)-> torch.tensor:
    
    """
    returns: a scalar tensor
    """
    
    #Shape: [B, H, W] --> [B, H, W, num_classes]
    ground_truth_oh = F.one_hot(ground_truth, num_classes=num_classes)
    
    
    #********* Only for Dice loss logic *****************
    #After applying softmax num_classes will be at dim = 1
    #Shape: [B, H, W] --> [B, H, W, num_classes]
    prediction_norm = F.softmax(predictions, dim = 1).permute(0, 2, 3, 1)
    
    
    #Intersection: |G ∩ P| Shape: [B, num_classes]
    intersection = (prediction_norm * ground_truth).sum(dim = dims)
    
    summation = prediction_norm.sum(dim=dims) + ground_truth.sum(dim =dims)
    
    #Dice Shape: [B, num_classes] 
    #Smoothing factor to avoid zero div error
    dice = (2.0 * intersection + smooth) / summation
    
    #Compute mean over the remaining axes (batch and classes)
    dice_mean = dice.mean()
    
    #*****************************************************
    
    #CE Loss
    CE = F.cross_entropy(predictions, ground_truth)
    
    return (1.0 - dice_mean) + CE
    
    
    
    
    
    
