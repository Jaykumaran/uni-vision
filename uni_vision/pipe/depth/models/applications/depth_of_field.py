import cv2
import numpy as np

class DepthField:
    
    def __init__(self, img_path : str, raw_depth_path: str, focal_range= 0.1):
        
        self.img_path = img_path
        self.raw_depth_path = raw_depth_path
        self.focal_range = focal_range # # Range around focal depth to remain sharp
    
    def apply_dof(focal_depth):
        