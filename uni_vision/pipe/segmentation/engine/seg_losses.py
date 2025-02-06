import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple







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
    
    
    
    
    


### **************** U^2Net **********###


### ***************** ISNet **********###
bce_loss = nn.BCELoss(size_average=True)

def multi_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0
    
    for i in range(0, len(preds)):
        
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            tmp_target = F.interpolate(target, size = preds[i].size()[2:], mode = "bilinear", align_corners=True)
            loss = loss + bce_loss(preds[i], tmp_target)
        else: #pred and tar are of same size
            loss = loss + bce_loss(preds[i], target)
        
        if(i==0):
            loss0 = loss #initial loss
        
    return loss0, loss


fea_loss = nn.MSELoss(size_average=True) # average over all loss values, i.e. loss of each samples is averaged ??
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)

#- - - 

def multi_loss_fusion_kl(preds, target, dfs, fs, mode = "MSE"):
    
    loss0 = 0.0
    loss  =0.0
    
    for i in range(0, len(preds)):
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            tmp_target = F.interpolate(target, size = preds[i].size()[2:], mode = "bilinear", align_corners=True)
            loss = loss + bce_loss(preds[i], tmp_target)
        else: # pred and tar are of same size
            loss = loss + bce_loss(preds[i], target)
        
        if(i==0):
            loss0 = loss #initial loss
    
    for i in range(0, len(preds)):
        if (mode == 'MSE'):
            loss = loss + fea_loss(dfs[i], fs[i]) # add the mse loss of features as additional constraints
        elif (mode == 'KL'):
            loss = loss + kl_loss(F.log_softmax(dfs[i], dim = 1), F.softmax(fs[i], dim = 1))
        elif (mode == 'MAE'):
            loss = loss + l1_loss(dfs[i], fs[i])
        elif (mode == "SmoothL1"):
            loss = loss + smooth_l1_loss(dfs[i], fs[i])
            
    return loss0, loss
        
    
    
