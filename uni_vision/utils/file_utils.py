import json




def load_annotations(ann_path):
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    
    return annotations