import numpy as np
import cv2

def rev_id2color(id2color: dict):

    rev_id2color = {value: key for key, value in id2color.items()}
    
    return rev_id2color



#Visualization Utilities
def rgb2grayscale(rgb_arr, color_map = rev_id2color, bg_cls_idx = 0):
    
    #Collapse H,W dimensions
    reshaped_rgb_arr  = rgb_arr.reshape((-1,3))
    
    '''Get an array of all the unique pixels along with the "inverse" array.
    of the same shape as the original array filled with indices to the unique array.
    Each value in the "inverse" array points to the unique pixel at that location
    in the input array.
    '''
    
    unique_pixels, inverse = np.unique(reshaped_rgb_arr, axis=0, return_inverse=True)
    
    #If unique pixel not found in color map, class ID of background pixel is used
    grayscale_map = np.array([color_map.get(tuple(pixel), bg_cls_idx) for pixel in unique_pixels])[inverse]
    
    return grayscale_map.reshape(rgb_arr.shape[:2])
    


def num2rgb(num_arr, colormap : dict = None):
    
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
    
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]
    
    return np.float32(output) / 255.0 #normalized



def image_overlay(image, segmented_image):
    
    alpha = 1.0 #Og image transparency
    beta = 0.7 #seg map 
    gamma  = 0.0 
    
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
    
    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, dst = image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return np.clip(image, 0.0, 1.0)




