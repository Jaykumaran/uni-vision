from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    BitsAndBytesConfig,
)
import torch



def load_causal_model(
    model_id,
    quant_config: BitsAndBytesConfig = None,
    dtype = "auto",
    device: torch.tensor.device = "cuda",
):
    
   processor = AutoProcessor.from_pretrained(
       model_id,
       trust_remote_code = True,
       torch_dtype = "auto",
       device_map = device
   ) 
   
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       trust_remote_code = True,
       torch_dtype = dtype,
       device_map = device,
       quantization_config = quant_config
   )
   
   
   return model, processor