from torch.utils.tensorboard import SummaryWriter
import os
from uni_vision.configs.train_config import TrainingConfig

def setup_logdir(training_config: TrainingConfig):
    
    #Ensure roor dir exists
    os.makedirs(training_config.root_log_dir, exist_ok = True)
    os.makedirs(training_config.root_checkpoint_dir, exist_ok = True)
    
    
    #Check for existing versions folders
    if os.path.isdir(training_config.root_log_dir):
        #Get all folder numbers in the root_dir
        folder_numbers  = [
                            int(folder.replace("version_", ""))
                            for folder in os.listdir(training_config.root_log_dir)
                            if folder.startswith("version_") and folder.replace("version_", "").isdigit()
                        ]
        
        if folder_numbers:
            last_version_number = max(folder_numbers)
            
            #New version name
            version_name = f"version_{last_version_number + 1}"
        else:
            version_name = "version_0"
    else:
        version_name = "version_0"
        
    
    #Update training config default dir 
    training_config.log_dir = os.path.join(training_config.root_log_dir, version_name)
    training_config.checkpoint_dir = os.path.join(training_config.root_checkpoint_dir, version_name)
    
    os.makedirs(training_config.log_dir, exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    print(f"Logging at: {training_config.log_dir}")
    print(f"Model checkpoint at: {training_config.checkpoint_dir}")
    
    
    return training_config, version_name