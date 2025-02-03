import os
import requests
from zipfile import ZipFile

from PIL import Image
import numpy as np
from torchinfo import summary

import torch
import torchvision.transforms as T

import torchvision.transforms.functional as F

import matplotlib.pyplot as plt




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



U2NET_MODEL_URL = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
U2NETP_MODEL_URL = "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"


U2NET_MODEL_PATH = "u2net.pth"
U2NETP_MODEL_PATH = "u2netp.pth"


from u2net import U2NET, U2NETP


u2net = U2NET(in_ch = 3, out_ch = 1)
u2netp = U2NETP(in_ch = 3, out_ch = 1)

def load_model(model, model_path, device):
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model = model.to(device)
    
    return model
    

u2net = load_model(model = u2net, model_path=U2NET_MODEL_PATH, device = DEVICE)
u2netp = load_model(model = u2netp, model_path=U2NETP_MODEL_PATH, device = DEVICE)


resize_shape = (320, 320)

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

transforms = T.Compose([
    T.ToTensor(), 
    T.Normalize(mean = MEAN, std = STD)
])


def prepare_batch(image_dir, resize, transforms, device):
    
    image_batch = []
    for image_file in os.listdir(image_dir):
        image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
        image_resize = image.resize(resize, resample=Image.BILINEAR)
        
        image_trans