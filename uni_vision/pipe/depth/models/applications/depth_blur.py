import os
import cv2
import numpy as np


#Usage: depthblur = DepthBlur(img_path, depth_map_path, focus_margin)
# depthblur.run()

class DepthBlur:
    
    def __init__(self, img_path : str, raw_depth_path: str, focus_margin: int = 0.2):
        
        self.img_path = img_path
        self.raw_depth_path = raw_depth_path
        
        self.rgb_image = cv2.imread(self.img_path)
        self.depth_map = cv2.imread(self.raw_depth_path, cv2.IMREAD_GRAYSCALE)
        self.focus_margin = focus_margin
        
    
    def blur(self):
        
        if self.rgb_image is None:
            raise FileNotFoundError(f"Image not found at {self.img_path}")
        if self.depth_map is None:
            raise FileNotFoundError(f"Depth map not found at {self.raw_depth_path}")
        
        # Normalize depth map to range [0, 1]
        depth_map_normalized = cv2.normalize(self.depth_map.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        
        # Convert normalized depth map back to uint8 (0-255 range)
        depth_map_uint8  = (depth_map_normalized * 255).astype(np.uint8)
        
        #Automatically infer focus range
        min_depth = np.min(depth_map_normalized)
        
        focus_near = int(min_depth * 255)
        focus_far = int((min_depth + self.focus_margin) * 255)
        
        # Debug: Print focus range
        print(f"Focus range: {focus_near} to {focus_far}")
        
        # Create a binary mask for the focus region
        focus_mask = cv2.inRange(depth_map_uint8, focus_near, focus_far)
        
        #Apply Gaussian Blur to the entire image
        blurred_image = cv2.GaussianBlur(self.rgb_image, ksize=(51,51), sigmaX=0)
        
        # Convert focus mask to 3 channels for blending
        focus_mask_color = cv2.merge([focus_mask, focus_mask, focus_mask])
        
        # Blend images: Keep original where mask is white, blur otherwise
        result = np.where(focus_mask_color == 255, self.rgb_image, blurred_image)
        
        return result
        
        
    def run(self):
        
        result = self.blur()
        save_path = os.path.splitext(os.path.basename(self.img_path))[0]
        
        cv2.imshow("Depth Blur Effect", result)
        cv2.imwrite(f'{save_path}_blur.jpg', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        