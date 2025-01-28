from global_utils.hf_model_utils import load_causal_model
from transformers import GenerationConfig
import torch
import cv2
import re



def molmo_answer(model_id, quant_config, query_text, input_image,max_tokens =2048, device: torch.tensor.device = "cuda"):
    
    
    model, processor = load_causal_model(model_id=model_id, quant_config=quant_config, device=device)
    inputs = processor.process(images = input_image, text = query_text)
    
    #move inputs to the correct device and create a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k,v in inputs.items()}
    
    
    output = model.generate_from_batch(
        inputs, GenerationConfig(max_new_tokens = max_tokens, stop_strings = "<|endoftext|>", tokenizer = processor.tokenizer)
    )
    
    #Only get the generated tokesn; decode them to text 
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens = True)
   
    return generated_text



def overlay_pointson_image(image, points ,radius = 5, color = (255, 0, 0)):
    
    """Overlay points and a label on the image

    Args:
    image: in RGB Format -->BGR -> RGB format
    points: List of points with coordinates [(x1, y1), (x2, y2), ...]  
    """
    
    pink_color = (158, 89, 243)
    
    for (x,y) in points:
        
        
        #Draw points as circle with outline for highlighting
        outline = cv2.circle(image, (int(x), int(y)), radius=radius+1, 
                             color = pink_color, thickness=2, lineType=cv2.LINE_AA) #Check for Color channel format BGR/RGB
        image_with_points = cv2.circle(outline, (int(x), int(y)),
                                       radius=radius, color = color, 
                                       thickness= -1, lineType=cv2.LINE_AA)
        
        save_image = image_with_points.copy()
        image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("Overlayed Points", save_image)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    

def molmo_coords(image, generated_text):
    
    h, w, _ = image.shape #OpenCV/Array image
    
    if "</point" in generated_text:
        #regex
        matches = re.findall(r'(?:x(?:\d*)="([\d.]+)"\s*y(?:\d*)="([\d.]+)")', generated_text)
        
        if len(matches) > 1:
            cordinates = [(int(float(x_val)/ 100 * w), int(float(y_val)/100*h)) for x_val, y_val in matches]
            
        else:
            coordinates = [(int(float(x_val)/100 * w), int(float(y_val) / 100 *h)) for x_val, y_val in matches]
            
        
    else:
        print("There are no points obtained from the regex pattern")
    
    return coordinates
        