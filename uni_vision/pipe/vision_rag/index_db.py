from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor



def vision_index(model_hf_name, file, ds):
    
    
    model = ColPali.from_pretrained(
        
        model_hf_name
    )
    processor = ColPaliProcessor.from_pretrained(model_hf_name)

    
    images = []
    for f in file:
        images.extend(convert_from_path(f))
    
    #run inference
    dataloader = DataLoader(
        images,
        batch_size = 4,
        shuffle = False,
        collate_fn = lambda x: processor.process_images(x)
    )
    
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to("cuda") for k,v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
            
        ds.extend(list(torch.unbind(embeddings_doc.to("cuda"))))
    
    return f"Uploaded and converted {len(images)} pages", ds, images
            
    
    