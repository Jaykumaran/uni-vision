from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor


def vision_search(model_hf_name : str, queries: str, ds, images):
        
    model = ColPali.from_pretrained(
        
        model_hf_name
    )
    processor = ColPaliProcessor.from_pretrained(model_hf_name)

    
    qs = []
    with torch.no_grad():
        batch_queries = processor.process_queries([queries]).to("cuda")
        batch_queries = {k: v.to("cuda") for k, v in batch_queries.items()}
        queries_embeddings = model(**batch_queries)
        qs.extend(list(torch.unbind(queries_embeddings.to("cuda"))))
        
        
    scores = processor.score_multi_vector(qs,ds)
    best_page_idx = int(scores.argmax(axis = 1).item())
    return f"The most relevant page is {best_page_idx}", images[best_page_idx]
    