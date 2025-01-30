import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List



#Support Multi Class and Binary Segmentation
class CustomSegDataset(Dataset):
    
    def __init__(self, *, image_size, num_classes, image_paths:List, mask_paths : List = None, transforms = None, is_train = False):
        
        self.img_size = image_size
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        
        self.is_train = is_train
        self.transforms = transforms
        
    
    def __len__(self):
        return len(self.image_paths)

    
    def load_file(self, file_path, interpolation = cv2.INTER_NEAREST, mask = False):
        
        if mask:
            file = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        else:
            file = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) #BGR 2 RGB
            
        file = cv2.resize(file, self.img_size, interpolation = interpolation)
    
        return file
    
    def __getitem__(self, idx):
        
        #Get image and mask path
        image_path = self.image_paths[idx]
        
        #Load image
        image = self.load_file(image_path, interpolation=cv2.INTER_CUBIC)
        
        mask = None
        if self.mask_paths is not None:
            #Get mask path
            
            mask_path = self.mask_paths[idx]
            
            mask = self.load_file(mask_path, interpolation=cv2.INTER_NEAREST, mask = True)
            
            #Process the mask correctly
            if self.num_classes == 1:
                mask = (mask > 0).astype(np.float32) #Binary mask (0 or 1)
            else:
                mask = mask.astype(np.int64) #Multi class mask (integer labels)
            
            
            #Apply transformations
            if self.transforms:
                augmented = self.transforms(image = image, mask = mask if mask is not None else None)
                image  = augmented['image']
                mask  = augmented['mask'] if "mask" in augmented else mask
            
            #Convert to tensors
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) #C, H, W
            if mask is not None:
                mask = torch.tensor(mask, dtype = torch.float32 if self.num_classes == 1 else torch.long)
            
            return image if mask is None else (image, mask)