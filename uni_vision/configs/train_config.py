import os
from dataclasses import dataclass, field
import torch

@dataclass
class TrainingConfig:
    root_log_dir : str  = "../train_logs"
    root_checkpoint_dir: str = "../checkpoints"
    
    log_dir: str = field(init=False) # automatically generate `log_dir` and `checkpoint_dir` based on root paths
    checkpoint_dir: str = field(init = False)
    
    
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
    
    
    def __post_init__(self):
        
        self.log_dir = os.path.join(self.root_log_dir, "experiment_logs")
        self.checkpoint_dir = os.path.join(self.root_checkpoint_dir, "model_checpoints")