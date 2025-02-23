import os
import time
import copy
import random
from typing import Union, Optional
import numpy as np
import torch.utils.tensorboard
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from uni_vision.logger_tb.tb_log import setup_logdir
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from uni_vision.configs.train_config import TrainingConfig
from ..utils.plot_metrics import plot_metrics
from ..utils.plot_predictions import plot_predictions


bold = f"\033[1m"
reset = f"\033[0m"



class Trainer:
    
    def __init__(self,  train_config: TrainingConfig, model: nn.Module , train_loader : DataLoader, val_loader: DataLoader, 
                 optimizer : optim.Optimizer, total_epochs: int, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None): 
    
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_epochs = total_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        #Setup log dir and model ckpt versioning
        self.train_config, _ = setup_logdir(train_config)

        #Tensorboard Writer
        self.tb_writer = SummaryWriter(
            log_dir= train_config.log_dir,
            comment= "Univision ~ Up and Running"
        )
        
    
    def train_one_epoch(self, epoch_idx: int, device: Union[str, int, torch.device]  = "cuda" , class_weight: Optional[torch.tensor ]= None):
        
        if isinstance(device, int):
            device = torch.device(f"cuda: {device}")
        elif isinstance(device, str):
            device = torch.device(device)
        
        
        self.model.train()
        num_classes = self.train_loader.dataset.__num_classes__
        self.scaler = torch.amp.GradScaler()

        acc_metric = MulticlassAccuracy(num_classes = num_classes, average = "micro")
        mean_metric = MeanMetric()
        
        status = f"Train:\t{bold} Epoch: {epoch_idx}/{self.total_epochs}{reset}"
        prog_bar = tqdm(self.train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        prog_bar.set_description(status)
        
        for (batch_data, target) in prog_bar:
            batch_data, target = batch_data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            
            output = self.model(batch_data)
            
            if class_weight is not None:
                loss = F.cross_entropy(output, target, weight = class_weight)
            else:
                loss = F.cross_entropy(output, target)
            
            self.scaler.scale(loss).backward()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            
            #batch loss
            mean_metric.update(loss.cpu().item(), weight = batch_data.shape[0])

            #get prob score using softmax
            prob = F.softmax(output, dim = 1) #along the out_channels dim
            
            #Get the index of max probability
            pred_idx = prob.detach().argmax(dim = 1)
            
            #Batch accuracy
            acc_metric.update(pred_idx.cpu(), target.cpu())
            
            #Update progress bar description
            step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():.4f}"
            prog_bar.set_description(step_status)
            
    
        #Per epoch
        train_loss = mean_metric.compute()
        train_acc = acc_metric.compute()
        
        prog_bar.close()
        
        return train_loss, train_acc
    
    
    def validate(self, epoch_idx: int, device: Union[str, int, torch.device]  = "cuda"):
        
        if isinstance(device, int):
            device = torch.device(f"cuda: {device}")
        elif isinstance(device, str):
            device = torch.device(device)
            
        num_classes = self.val_loader.dataset.__num_classes__
        acc_metric = MulticlassAccuracy(num_classes = num_classes, average='micro')
        mean_metric = MeanMetric()
        
        status = f"Valid:\t{bold} Epoch: {epoch_idx}/{self.total_epochs}{reset}"
        prog_bar = tqdm(self.val_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        prog_bar.set_description(status)
        
        for batch_data, targets in prog_bar:
            
            batch_data, targets = batch_data.to(device), targets.to(device)
            
            self.model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=str(device)):
                
                  outputs = self.model(batch_data)
            
            prob = F.softmax(outputs, dim = 1)
            
            val_loss = F.cross_entropy(outputs, targets).item()
            
            pred_idx = prob.detach().argmax(dim = 1)
            
            mean_metric.update(val_loss, weight = batch_data.shape[0])
            
            acc_metric.update(pred_idx.cpu(), targets.cpu())
            
            #Update prog bar description
            step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():.4f}"
            prog_bar.set_description(step_status)
            
        val_loss = mean_metric.compute()
        val_acc = acc_metric.compute()
        
        prog_bar.close()
        
        
        return val_loss, val_acc
            
        
        
    def run(self,
        DEVICE: torch.device | str, 
    ) -> dict:
    
        DEVICE = torch.device(DEVICE) if isinstance(DEVICE, str) else DEVICE
        
        best_loss = torch.tensor(np.inf) #largest possible value from there descrease as it has to be minimum
        best_weights = None
        
        best_acc = torch.tensor(-np.inf)
        best_epoch = None # Start with the first epoch
        
        
        #epoch train/val loss
        epoch_train_loss, epoch_val_loss = [], []
        
        #epoch train/test accuracy
        epoch_train_acc, epoch_val_acc = [] , []
         
        
        #training time measurement
        t_begin = time.time()
        
        for epoch in range(self.total_epochs):
            
            #Training
            train_loss, train_acc = self.train_one_epoch( epoch_idx = epoch + 1, device=DEVICE)
            
            val_loss, val_acc = self.validate(epoch_idx = epoch + 1, device = DEVICE)
            
            train_loss, train_acc = float(train_loss), float(train_acc)
            val_loss, val_acc = float(val_loss), float(val_acc)
            
            
            train_loss_stat = f"{bold}Train Loss: {train_loss:.4f}{reset}"
            train_acc_stat =  f"{bold}Train Acc: {train_acc:.4f}{reset}"
            
            val_loss_stat = f"{bold}Val Loss  : {val_loss:.4f}{reset}"
            val_acc_stat =  f"{bold}Val Acc  : {val_acc:.4f}{reset}"
            
            print(f"\n{train_loss_stat:<30}{train_acc_stat}")
            print(f"{val_loss_stat:<30}{val_acc_stat}")
            
            
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
            
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)  
            
            self.tb_writer.add_scalars('Loss/train-val', {'train': train_loss,
                                                          'validation':val_loss}, epoch)
            
            self.tb_writer.add_scalars('Accuracy/train-val', {'train': train_acc, 
                                                              'validation': val_acc}, epoch)
            
            
            #(Adjust LR)
            if self.scheduler:
               #For schedulers like 'ReduceLROnPlateau', step with val loss
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                   self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch + 1
                
                print(f"Model Improved ... Saving Model ...💾 ...", end = "")
                best_weights = copy.deepcopy(self.model.state_dict())   
                
                checkpoint_path = os.path.join(
                    self.train_config.checkpoint_dir,
                    f"best_epoch_{epoch + 1}_loss_{val_loss:.4f}.pth"
                )
                  
                torch.save({
                    "model_state_dict": best_weights,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch + 1,  
                }, checkpoint_path)
                print(f"...: Saved to {checkpoint_path}✅ ...\n")
            
            print(f"{'=' * 72}\n")
                
        print(f"Total time: {(time.time() - t_begin):.2f}s, Best Val Loss: {best_loss:.3f}, Best Val Acc: {best_acc:.2f} ; At Epoch: {best_epoch}")    

  
               
        #Load best weights   
        self.model.load_state_dict(best_weights)
        
        history = dict(
            train_loss = epoch_train_loss,
            train_acc = epoch_train_acc,
            valid_loss = epoch_val_loss,
            valid_acc = epoch_val_acc,
        )
        
        
        plot_metrics(
            self.train_config,
            [epoch_train_acc, epoch_val_acc],
            ylabel = "Accuracy",
            ylim = [0.0, 1.1],
            metric_name = ["Training Accuracy", "Validation Accuracy"],
            color = ["b", "g"] ,#'b'
            num_epochs = self.total_epochs,
            save_name='accuracy_plot'
        )
        
        plot_metrics(
            self.train_config,
            [epoch_train_loss, epoch_val_loss],
            ylabel = "Loss",
            ylim = [0.0, 2.0],
            metric_name = ["Training Loss", "Validation Loss"],
            color = ["r", "y"],
            num_epochs=self.total_epochs,
            save_name='loss_curve_plot'
        )
        
        plot_predictions(self.train_config, 
                         model = self.model,
                         data_loader=self.val_loader, 
                         class_names=self.val_loader.
                         dataset.__classes__, 
                         mode = "all",
                         num_samples = 10)    
            
        return history
            
            
            
            
        
        
        
        
       
        

