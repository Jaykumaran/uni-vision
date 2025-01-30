import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2

class CustomSegDataset(Dataset):
    
    def __init__(self, *, image_size, num_classes, image_paths, mask_paths = None, transforms = None, is_train = False):
        
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
            file = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BAYER_BGR2RGB) #BGR 2 RGB
        file = cv2.resize(file, self.img_size, interpolation = interpolation)
    
        return file
    
    def __getitem__(self, idx):
        
        #Get image and mask path
        image_path = self.image_paths[idx]
        
        #Load image
        image = self.load_file(image_path, interpolation=cv2.INTER_CUBIC)
        
        if self.mask_paths is not None:
            #Get mask path
            
            mask_path = self.mask_paths[idx]
            
            mask = self.load_file(mask_path, interpolation=cv2.INTER_NEAREST, mask = True)
            
            #*