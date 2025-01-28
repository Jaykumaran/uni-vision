from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    root_log_dir : str
    root_checkpoint_dir: str
    log_dir: str
    checkpoint_dir: str
    
    
    batch_size : int = 32,
    img_size : tuple = (224, 224),
    total_epochs : int = 50
    init_learning_rate : float = 1e-4,
    weight_decay : float = None,
    log_interval : int = 5,
    test_interval : int = 1,
    device : str = "cuda",
    scheduler : torch.optim.lr_scheduler = None,
    num_workers : int = 2