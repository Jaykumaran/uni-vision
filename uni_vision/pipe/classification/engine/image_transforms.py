from typing import List

import torch
from torchvision import transforms




def common_transforms(mean: List  = [0.485, 0.456, 0.406] , std: List = [0.229, 0.224, 0.225] , img_size = (224,224)):
    """Performs Naive Transformations

    Args:
        mean (List, optional): _description_. Defaults to [0.485, 0.456, 0.406].
        std (List, optional): _description_. Defaults to [0.229, 0.224, 0.225].
        img_size (tuple, optional): _description_. Defaults to (224,224).
    """
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std) #by default imagenet mean and std
    ])
    return preprocess


def denormalize(image_tensor, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
    
    
    """
    Denormalizes a tensor with a given mean and std.
    Args:
        image_tensor (torch.Tensor): The normalized image tensor (C, H, W)
        mean (list or tuple) : The mean used for normalization 
        std (list or tuple): The standard deviation used for normalization
    """
    
    mean = torch.tensor(mean , dtype = image_tensor.dtype, device = image_tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype = image_tensor.dtype, device = image_tensor.device).view(-1, 1, 1)
    
    return image_tensor * std + mean
    