import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch




def denormalize(image_tensor: torch.tensor, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
    
    for c in range(3):
        image_tensor[:, c, :, :].mul_(std[c]).add_(mean[c])
    
    return torch.clamp(image_tensor, min = 0.0, max = 1.0)




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




def overlayed_groundtruth(*, images, masks, color_mask = False, color_map = id2color):
    
    title = ['GT Image', 'GT Mask', 'Overlayed Mask']
    
    for idx in range(images.shape[0]):
        
        image = images[idx]
        grayscale_gt_mask = masks[idx]
        
        plt.figure(figsize = (12, 4))
        
        #Create RGB Seg map from Grayscale seg map
        rgb_gt_mask = num2rgb(grayscale_gt_mask, colormap=color_map)
        
        overlayed_image = image_overlay(image, rgb_gt_mask)
        
        plt.subplot(1, 3, 1)
        plt.title(title[0])
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(title[1])
        if color_mask:
            plt.imshow(rgb_gt_mask)
        
        else:
            plt.imshow(grayscale_gt_mask)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(title[2])
        plt.imshow(overlayed_image)
        plt.axis('off')
        
        plt.show()
        plt.close()
        
    return