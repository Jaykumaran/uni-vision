from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    IMG_HEIGHT : int = 224,
    IMG_WIDTH : int = 224,
    NUM_CLASSES : int = None,
    DATA_ROOT: str = None
    
    