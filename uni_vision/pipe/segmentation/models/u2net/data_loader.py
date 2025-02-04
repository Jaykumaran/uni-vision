#Source: https://github.com/xuebinqin/U-2-Net/blob/master/data_loader.py 

from __future__ import print_function, division

import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class Rescale:
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
        
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['lable']
        
        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            
            if h > w: #maintain aspect ratio
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size
        
        new_h , new_w = int(new_h), int(new_w)

        #normalize
        image = transform.resize(image, (self.output_size, self.output_size), mode = 'constant')
        label = transform.resize(label, (self.output_size, self.output_size), mode = 'constant', order = 0, preserve_range=True)
        
        return {'imidx': imidx, image: 'image', 'label': label}                
                
                
                
                
        
        
    
