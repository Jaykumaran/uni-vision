import cv2
from typing import Tuple




#**************** Detection ****************************
def draw_bounding_boxes(image, boxes, color: Tuple = (0,255,255), thickness: int = 3):
    boxes = boxes.cpu().numpy()
    
    for box, category_id in boxes:
        
        x, y, w, h = box
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        

        ann_img = cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color = color,
            thickness = thickness
        )
        
        ann_img = cv2.putText(
            ann_img.copy(),
            str(category_id),
            (xmin, ymin),
            font = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color = color,
            thickness  = thickness,
            lineType = cv2.LINE_AA  
        )
        
        return ann_img

#**************** Segmentation ****************************
