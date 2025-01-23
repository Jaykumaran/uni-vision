import open3d as o3d
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



class depthToPointCloud:

    def __init__(self, img_path : str, raw_depth_path: str, f_px: int, plyfile_name: str):
    
        
        self.rgb_image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        self.raw_depth = cv2.imread(raw_depth_path, cv2.IMREAD_UNCHANGED)
        
        self.plyfile_name = plyfile_name
        
        self.width, height  = self.rgb_image.shape[:2]
        
        
        #Camera intrinsics
        
        f_px = 470.4 #in mm
        f_py = f_px #square pixels of camera sensor
    
    def register_cloud(self):
        
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        z = np.array(self.raw_depth)
        
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis = -1).reshape(-1, 3)
        colors = np.array(self.rgb_image).reshape(-1, 3) / 255.0
        
        out_dir = "registered_clouds"
        os.makedirs(out_dir, exist_ok=True)
        
        pcd = o3d.geometry.PointCloud()
        
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
                
        o3d.io.write_point_cloud(f'{os.path.join(out_dir, self.plyfile_name)}', pcd)
        

    def visualize(self):