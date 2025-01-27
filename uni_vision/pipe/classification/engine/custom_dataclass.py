import os
import pandas as pd
from PIL import Image

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from image_transforms import common_transforms

#Usage:
# ********* CSV ************
#dataset_csv = CustomDatasetClass(input_path="data.csv", root_dir="images", transform=ToTensor())
# ******** ImageFolder ***********
#dataset_folder = CustomDatasetClass(input_path="data_folder", transform=ToTensor())


class CustomDatasetClass(Dataset):
    """Supports either a csv file or root dir of the dataset folder"""
    
    def __init__(self, input_path, root_dir = None, transform = None, img_size = (224,224)):
        
        if transform is not None:
            self.transform = transform(img_size)
        else:
            self.transform = common_transforms(img_size)
        
        if os.path.isfile(input_path) and input_path.endswith('.csv'):
            self.data_frame = pd.read_csv(input_path)
            self.root_dir = root_dir
                
            #Ensure root dir is specified when using a CSV File
            if not self.root_dir:
                raise ValueError("root_dir must be specified when input path is a CSV File")
            
            #class_idx mapping from the CSV File
            self.classes = self.data_frame['class'].unique()
            self.cls_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
        elif os.path.isdir(input_path):
            self.image_folder = datasets.ImageFolder(root = input_path, transform = transform)
            self.classes = self.image_folder.classes
            self.cls_to_idx = self.image_folder.class_to_idx
        
        else:
            raise ValueError("input path must be a valid CSV or a  path to the dataset root folder")
        
   
    def __len__(self):
        #Use ImageFolder's length if a folder is provided
        if hasattr(self, 'image_folder'):
            return len(self.image_folder)
        #Otherwise dataframe from CSV
        return len(self.data_frame)
    
    @property
    def __num_classes__(self):
        return len(self.classes)
    
    @property
    def __classes__(self):
        return self.classes
    
    
    def __getitem__(self, idx):
        
        #If using ImageFolder
        if hasattr(self, 'image_folder'):
            return self.image_folder[idx]
        
        #If using CSV File
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        
        img_name = self.data_frame.iloc[idx, 0]
        img_path  = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB') #remove alpha channel
        
        class_str = self.data_frame.iloc[idx, 1]
        label = self.cls_to_idx[class_str]
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
        return image, label
        