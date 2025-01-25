#Should be model agnostic
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  
import tempfile
import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

#Usage: metric_depth = metricDepthVisualize(img_path, depth_map_path, focal_px)
# metric_depth.run()


class metricDepthVisualize:
    
    def __init__(self, img_path: str, depth_map_path: str, focal_px: int, max_size = 1536):
        
        self.img_path = img_path
        self.depth_map_path = depth_map_path
        self.focal_px = focal_px
        self.max_size = max_size
        
        self.rgb_image = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        self.depth_map = cv2.imread(self.depth_map_path, cv2.IMREAD_GRAYSCALE)
        

        self.rgb_image = cv2.resize(self.rgb_image, (self.max_size, self.max_size))
        self.depth_map = cv2.resize(self.depth_map, (self.max_size, self.max_size))
        
        # Ensure depth is a 2D numpy array
        if self.depth_map.ndim !=2:
            self.depth_map = self.depth_map.squeeze()
        
        self.inverse_depth = 1.0 / (self.depth_map + 1e-6)
        
        
    def show_distance_OpenCVWindow(self):
        """Displays the OpenCV window with constant distance display."""
        
        # Clip and normalize the inverse depth for visualization
        self.inverse_depth = np.clip(self.inverse_depth, a_min = 1e-6, a_max = 10)
        normalized_image = cv2.normalize(src = self.inverse_depth, dst = None, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        
        color_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_INFERNO)
        
        #Initialize mouse position
        mouse_position = [0, 0]
        
        def update_mouse_position(event, x, y, flags, param):
            "Update the mouse position on mouse events"
            if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
                mouse_position[0], mouse_position[1] = x, y
                
            
        cv2.namedWindow("Interactive Depth Viewer")
        cv2.setMouseCallback("Interactive Depth Viewer", update_mouse_position)
        
        while True:
            display_image = color_map.copy()
            
            #Get the current mouse position
            x, y = mouse_position
            
            cv2.circle(display_image, (x,y), radius=5, color = (255,255,255), thickness=-1)
            # Ensure the position is within the image bounds
            if 0 <= x < self.inverse_depth.shape[1] and 0 <= y < self.inverse_depth.shape[0]:
                inv_depth = self.inverse_depth[y, x]
                z = 1.0 / inv_depth if inv_depth > 1e-6 else float('inf')
                cv2.putText(img = display_image,
                            text= f"Distance: {z:.2f} m",
                            org = (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color = (255,255,255),
                            thickness=2, lineType=cv2.LINE_AA)
                
            cv2.imshow("Interactive Depth Viewer", display_image)
            
            #Exit if 'Esc' is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
        cv2.destroyAllWindows()
        
    def run(self):
        
        self.show_distance_OpenCVWindow()
        
                 

        
        
        
        
    
        
        
        
        
