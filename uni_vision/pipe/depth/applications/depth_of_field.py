import cv2
import numpy as np



#Usage: dof_effect = DepthOfFieldEffect(img_path, depth_map_path, focal_range)
# dof_effect.run()


class DepthOfFieldEffect:
    
    def __init__(self, img_path : str, raw_depth_path: str, focal_range= 0.1, screen_width: int = 1920, screen_height : int = 1080):
        
        self.rgb_image = cv2.imread(img_path)
        self.depth_map = cv2.imread(raw_depth_path, cv2.IMREAD_GRAYSCALE)
        self.focal_range = focal_range # # Range around focal depth to remain sharp
        
        #Get screen size and resize window to fit
        self.screen_width = screen_width
        self.screen_height = screen_height
    
    def resize(self):

        depth_map_normalized = cv2.normalize(self.depth_map.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        
        scale = min(self.screen_width / self.rgb_image.shape[1], self.screen_height / self.rgb_image.shape[0])
        self.new_width = int(self.rgb_image.shape[1] * scale)
        self.new_height = int(self.rgb_image.shape[0] * scale)
        
        self.rgb_image = cv2.resize(self.rgb_image, (self.new_width, self.new_height))
        self.depth_map_normalized = cv2.resize(depth_map_normalized, (self.new_width, self.new_height))
        
      
    def apply_dof(self, focal_depth):
        

        focal_range = 0.1 ## Range around focal depth to remain sharp
    
        
        #Create smooth focus weights
        sharpness_weights = np.exp(-(self.depth_map_normalized - focal_depth) ** 2) / (2 * focal_range ** 2)
        
        #Apply Gaussian blur to the background
        blurred_image = cv2.GaussianBlur(self.rgb_image, ksize=(51,51), sigmaX=0) #By setting standard dev. as 0 , OpenCV automatically calculates optimal value based on ksize
        
        # Blend the original image and blurred image using sharpness weights
        sharpness_weights_3d = np.expand_dims(sharpness_weights, axis=2) #Add a channel for blending
        
        dof_image = sharpness_weights_3d * self.rgb_image + (1 - sharpness_weights_3d * blurred_image)
        dof_image = np.clip(dof_image, 0 , 255).astype(np.uint8)
        
        return dof_image
    
    # Callback function for the trackbar
    def on_trackbar(self, value):
        # Convert slider value (0-100) to focal depth (0.0-1.0)
        focal_depth = value / 100.0
        dof_image = self.apply_dof(focal_depth)
        cv2.imshow("Depth of Field Effect", dof_image)
        
    
    def run(self):
        
        # Create a window and resize it to fit the screen
        cv2.namedWindow("Depth of Field Effect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth of Field Effect", self.new_width, self.new_height)
        
        
        # Create a trackbar (slider) at the bottom of the window
        cv2.createTrackbar(trackbarName="Focal Plane", windowName="Depth of Field Effect", value = 50, count=100, onChange= self.on_trackbar) #Default at middle (50)
        
        #Show initial DOF Effect
        initial_dof_image = self.apply_dof(0.5) #Start with focal depth at 0.5
        cv2.imshow("Depth of Field", initial_dof_image)
        
        #Wait until user closes the Window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


