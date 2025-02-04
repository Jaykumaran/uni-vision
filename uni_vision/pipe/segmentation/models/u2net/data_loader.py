import torch
import numpy as np

from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from typing import List, Dict


class Transform:
    def __init__(self, output_size: int, mean : List = [0.485, 0.456, 0.406], std: List = [0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.Resize((output_size, output_size)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
        
        self.to_tensor = transforms.ToTensor()
        
    def __call__(self, sample: Dict):
        imidx, image, mask = sample["imidx"], sample["image"], sample["mask"]
        
        image = self.transform(image) #normalization 3 channel image
        mask = self.to_tensor(mask) #Convert to (1, H, W) without normalization
        
        return {"imidx": imidx, "image": image, "label": mask}



class SalObjDataset(Dataset):
    def __init__(self , img_name_list: List, mask_name_list: List, transform: transforms = None):
        
        self.image_name_list = img_name_list
        self.mask_name_list = mask_name_list
        self.transform = transform
        
    
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert('RGB')
        imidx = np.array([idx])
        
        if len(self.mask_name_list) == 0:
            mask = np.zeros((image.size[1], image.size[0]))
        else:
            mask = Image.open(self.mask_name_list[idx]).convert('L')
        
        sample = {"imidx": imidx, "image": image, "label": mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        

#Usage:
# transform = Transform(output_size=224)
# dataset = SalObjDataset(img_paths, masks_paths, transform=transform)