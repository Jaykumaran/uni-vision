from dataclasses import dataclass


@dataclass
class TrainingConfig:
    root_log_dir : str
    root_checkpoint_dir: str
    log_dir: str
    checkpoint_dir: str
    