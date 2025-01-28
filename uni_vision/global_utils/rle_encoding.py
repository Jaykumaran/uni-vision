import numpy as np

#Usually used in most of Kaggle Competition

# ******** RUN LENGTH ENCODING (RLE) ******************

#Segmentation
def segmentation_rle_encode(mask):
    
    #Flatten the mask into a 1D array
    pixels = mask.flatten()
    
    #Add a 0 at the beginning and end to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])
    #Find indices where pixel values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    #Find indices where pixel values change
    runs[1::2] -= runs[::2]
    #Return the RLE as a string
    return ' '.join(str(x) for x in runs) # " ".join(map(str, runs))
    
    
    

#For eg:
"""Given a mask:

    mask = [
        [0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]
    
    pixels.flatten()
    
    1. pixels = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    #Add 0 at begin and end
    2. pixels = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 ,0]
    
    #Find indices of transitions
    For eg: Compare pixels[1:] and pixels[:-1]
    Transition from 0 to 1 at index 3 - start of run
    Transition from 1 to 0 at index 7 - end of run
    runs = [4, 8, 14, 17, 22, 26]
    
    #Calculate run lengths
    Subtract start indices from end indices
    runs[1::2] -= runs[::2]
    Start indices: runs[::2] = [4, 14, 22]
    End indices: runs[1::2] = [8, 17, 26]
    Subtract runs[1::2] = [9 -4, 17 - 14, 26 - 22] = [4, 3, 4]
    runs  = [4, 4, 14, 3, 22, 4]
    #Indicates that,
    At index 4, there are 4 consecutive 1s
    At index 14, there are 3 consecutive 1s
    At index 22, there are 4 consecutive 1s
"""