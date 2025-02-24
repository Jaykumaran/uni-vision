# https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py


import logging
import os
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import sam2
from huggingface_hub import hf_hub_download



#check for circular import error based on parent dir
if os.path.isdir(os.path.join(sam2.__path[0], "sam2")):
    raise RuntimeError("You are currently running python from parent dir of sam2 repo"
                       "i.e. the directory where sam2 is cloned into..."
                       "To avoid circular imports run from another directory")
    
    

HF_MODEL_ID_TO_FILENAMES = {    
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.py",
    ),
     "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_tiny.py",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_tiny.py",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_tiny.py",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.py",
    ),
     "facebook/sam2-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.py",
    ),
      "facebook/sam2-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
       "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
    
                            
}



def build_sam2(
    config_file,
    ckpth_path = None,
    device = "cuda",
    mode = "eval",
    hydra_overrides_extra = [],
    apply_post_processing = True,
    **kwargs
):
    if apply_post_processing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability = true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability_delta = 0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability_thresh = 0.98",
            
        ]
        
        # Read config and init model
        cfg = compose(config_name = config_file, overrides = hydra_overrides_extra)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model,  _recursive = True)
        _load_checkpoint(model, ckpth_path)
        model = model.to(device)
        if mode == "eval":
            model.eval()
            return model
        

def  build_sam2_video_predictor(
    config_file,
    ckpth_path = None,
    device = "cuda",
    mode = "eval",
    hydra_overrides_extra = [],
    apply_postprocessing = True,
    vos_optimized = True,
    **kwargs
):
    
    hydra_overrides = [
        "++model._target_ = sam2.sam2_video_predictor.SAM2VideoPredictor"
    ]
    
    if vos_optimized:
        hydra_overrides = [
            "++model._target_ = sam2.samw_video_predictor.SAM2VideoPredictorvos",
            "++model.compile_image_encoder = True" # set sam2_base to handle this
        ]
        
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
        # dynamically fall back to multi-mask if the single mask is not stable
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability = true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability_delta = 0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability_thresh = 0.98",
        
        # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the masks
        # are exactly as what users see from clicking
        "++model.binarize_mask_from_pts_for_mem_enc = true" ,
        # fill small holes in the low-res masks to 'fill hole area' (before resizing them to the original video resolution),
        "++model.fill_hole_area = 8",
    ]
        
    hydra_overrides.extend(hydra_overrides_extra)
    
    
    # Read config and init model
    cfg = compose(
        config_name=config_file, overrides=hydra_overrides
    )
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive = True)
    _load_checkpoint(model, ckpth_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model



def _hf_download(model_id):
    
    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path

def build_sam2_hf(model_id, **kwargs):
    config_name, ckth_path  = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpth_path=ckth_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckth_path  = _hf_download(model_id)
    return build_sam2_video_predictor(config_file=config_name, ckpth_path=ckth_path, **kwargs)



def _load_checkpoint(model, ckpth_path):
    if ckpth_path is not None:
        sd  = torch.load(ckpth_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd) # check for keys
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint successfully")
        
