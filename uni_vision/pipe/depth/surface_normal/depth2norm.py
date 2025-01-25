import cv2
import numpy as np



def estimate_normal(depth: np.ndarray, kernel_size: int = 7):
    
    
    grad_x = cv2.Sobel(depth.astype(np.float32), ddepth=cv2.CV_32F, dx = 1, dy = 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth.astype(np.float32), ddepth= cv2.CV_32F, dx = 0, dy = 1, ksize=kernel_size)
    
    z = np.full(grad_x.shape, 1)
    
    #stack depth wise
    normals = np.dstack((-grad_x, -grad_y, z))
    
    #magnitude of normals
    normals_mag = np.linalg.norm(normals, axis= 2 , keepdims=True)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        normals_normalized = normals / (normals_mag + 1e-5) #qe-5 to avoid zero div errors
    
    #normalize between - 1 to 1
    normals_normalized = np.nan_to_num(normals_normalized, nan = -1, posinf=-1, neginf=-1)
    #normalize from [-1, 1] to [0, 1]
    normal_from_depth = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)

    return normal_from_depth