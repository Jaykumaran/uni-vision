import torch
import torch.nn.functional as F



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
