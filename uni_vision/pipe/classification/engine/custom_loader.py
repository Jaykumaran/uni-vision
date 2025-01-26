from torch.utils.data import DataLoader
from custom_dataclass import CustomDatasetClass
 





def loader(custom_dataclass, batch_size: int, shuffle = False, num_workers: int = 2):
    
    loader = DataLoader(custom_dataclass, 
                        batch_size = batch_size, 
                        shuffle=shuffle,
                        num_workers= num_workers
                        )
    
    
    return loader