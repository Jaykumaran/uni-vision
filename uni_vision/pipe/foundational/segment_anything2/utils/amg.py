# https://github.com/facebookresearch/sam2/blob/main/sam2/utils/amg.py

import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import numpy as np
import torch



class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """
    
    
    def __init__(self, **kwargs):
        for v in kwargs.values:
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
            
        self._stats = dict(**kwargs)
        
    
    def __setitem__(self, key:str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats[key] = item
        
    
    def __delitem__(self, key: str)  -> None:
        del self._stats[key]
    
    def __getitem__(self, key:str) -> Any:
        return self._stats[key]
    
    
    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()
    
    
    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")
            
    
    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim = 0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")
            
    
    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
               self._stats[k] = v.float().detach().cpu().numpy()

def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    
    """
    Filter masks at the edge of a crop, but not at edge of a image
    """
    
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device = boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float32, device = boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol = atol, rtol = 0) # check absolute tolerance to be considered equal
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol = atol, rtol=0)
    # used to refine edge areas 
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge) # ~ negates
    return torch.any(near_crop_edge, dim = 1)


def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh



def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args  # check all inputs to have
    ), "Batched iteration must have same number of inputs in a batch"
    n_batches = len(args[0] // batch_size + int(len(args[0]) % batch_size != 0)) # drop = False,
    
    for b in range(n_batches):
        yield(arg[b * batch_size   : (b + 1) * batch_size] for arg in args) # extracts a batch from each input with slicing
        
        batches = [
            [arg[b * batch_size :  (b + 1) * batch_size] for arg in args]
            for b in range(n_batches)
        ]
        

def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """  
    Encodes masks to an uncompressed RLE, in the format expected by pycoco tools.
    """
    
    # Swap in (h, w) fortran order and flatten h, w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(start_dim=1) #b, h*w
    
    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1] # bitwise XOR
    # Eg: Tensor: [0, 0, 1, 1, 1, 0, 0, 1] ; XOR: 0 1 0 0 1 0 1
    change_indices = diff.nonzero() # non zero extracts indices where non zero occurs. i.e., 1
    
    # Encode run length
    out = []
    for i in range(b): # batch_size
        cur_idxs = change_indices[change_indices[:, 0] == i, 1j]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype = cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device = cur_idxs.device),
            ]
        )     
        
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    
    return out

def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """ Compute a binary mask from an uncompressed RLE."""
    # rle.items() = size, counts
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False # to switch between fg and bg
    for count in rle["counts"]:
        mask[idx: idx + count] = parity
        idx += count    
        parity ^= True
    
    mask = mask.reshape(w, h)
    return mask.transpose() # Put in C order

def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])




    

