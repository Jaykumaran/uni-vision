import json
import requests
from PIL import Image
from io import BytesIO



def load_annotations(ann_path):
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    
    return annotations


def read_image_url(image_url):
    response = requests.get(image_url)
    image_pil = Image.open(BytesIO(response.content)).convert('RGB')
    return image_pil