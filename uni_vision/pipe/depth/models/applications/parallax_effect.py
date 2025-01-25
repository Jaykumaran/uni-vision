import os
import cv2
import numpy as np


#Usage: parallax = ParallaxEffect(img_path, depth_map_path, fps, frame_count, displacement)
# parallax.run()

class ParallaxEffect:
    def __init__(self, img_path:str, raw_depth_path: str, fps, frame_count, displacement):
        
        self.img_path = img_path
        self.raw_depth_path = raw_depth_path
        self.fps = fps
        self.frame_count = frame_count
        self.displacement = displacement
        
        self.rgb_image = cv2.imread(self.img_path)
        self.depth_map = cv2.imread(self.raw_depth_path, cv2.IMREAD_GRAYSCALE)
        
        if self.rgb_image is None:
            raise FileNotFoundError(f"Image not found at {self.img_path}")
        if self.depth_map is None:
            raise FileNotFoundError(f"Depth map not found at {self.raw_depth_path}")
        
    
    def parallax(self):
        depth_map = cv2.normalize(self.depth_map.astype('float32'),dst = None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX)
        
        h, w, _ = self.rgb_image.shape
        save_path = os.splitext(os.path.basename(self.rgb_image))[0]
        out = cv2.VideoWriter(f'{save_path}_parallax.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
        
        #Create a named window to avoid erros
        cv2.namedWindow('Parallax Effect', cv2.WINDOW_NORMAL)
        
        #Generate frames
        for t in range(self.frame_count):
            # Calculate x and y displacement to create a smooth looping effect
            dx = self.displacement * np.sin(2 * np.pi * t / self.frame_count)
            dy = self.displacement * np.cos(2 * np.pi * t / self.frame_count)
            
            # Warp the image based on the depth map and displacement
            flow_x = self.depth_map * dx
            flow_y = self.depth_map * dy
            
            #Debug: Check flow ranges
            if t == 0:
                print(f"Flox X Range: {flow_x.min()} to {flow_y.max()}")
                print(f"Flow Y Range: {flow_y.min()} to {flow_y.max()}")
                    
            #Create a meshgrid for remapping
            x,y  = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + flow_x).astype(np.float32)
            map_y = (y + flow_y).astype(np.float32)
        
            #Remap the image
            warped_image = cv2.remap(self.rgb_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            
            # Debug: Ensure warped image is valid
            if t == 0:
                print(f"Warped Image Shape: {warped_image.shape}")
                print(f"Warped Image Dtype: {warped_image.dtype}")
            
            #Write the frame to the video
            out.write(warped_image)
            
            #Show the effect frame by frame
            cv2.imshow("Parallax Effect", warped_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()
        cv2.destroyAllWindows()
    
    def run(self):
        self.parallax()
        
        
        
                  
        
        