import os
import time
import copy
import random
from typing import Union
import numpy as np
import torch.utils.tensorboard
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger_tb.tb_log import setup_logdir
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from torchmetrics.clasification import MulticlassAccuracy


bold = f"\033[1m"
reset = f"\033[0m"



class Train:
    
    def __init__(self,  model: nn.Module , train_loader : torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
                 optimizer : torch.optim.Optimizer, total_epochs: int, scheduler: torch.optim.lr_scheduler = None): 
    
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_epochs = total_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        
        
    
    def train_one_epoch(self, train_config: dataclass, epoch_idx: int, device: Union["str", int]  = "cuda", class_weight: int = None):
        
        self.model.train()
        
        #Set checkpoint and logging dir
        train_config, current_version_name = setup_logdir(train_config)
        
        tb_writer = SummaryWriter(
            log_dir=train_config.log_dir,
            comment= "Univision ~ Up and Running"
        )


        acc_metric = MulticlassAccuracy(num_classes = self.train_loader.dataset.__num_classes__, average = "micro")
        mean_metric = MeanMetric()
        
        status = f"Train:\t{bold} Epoch: {epoch_idx} / {self.total_epochs}{reset}"
        prog_bar = tqdm(self.train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        prog_bar.set_description(status)
        
        for (batch_data, target) in enumerate(prog_bar):
            batch_data, target = batch_data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            
            output = self.model(batch_data)
            
            if class_weight is not None:
                loss = F.cross_entropy(output, target, weight = class_weight)
            else:
                loss = F.cross_entropy(output, target)
            
            loss.backward()
            
            self.optimizer.step()
            
            if self.scheduler:
               self.scheduler.step()
            
            batch_loss = mean_metric(loss.item(), weight = batch_data.shape[0])

            #get prob score using softmax
            prob = F.softmax(output, dim = 1) #along the out_channels dim
            
            #Get the index of max probability
            pred_idx = prob.detach().argmax(dim = 1)
            
            #Batch accuracy
            batch_acc = acc_metric(pred_idx.cpu(), target.cpu())
            
            #Update progress bar description
            step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():4f}"
            prog_bar.set_description(step_status)
            
    
        #Per epoch
        train_loss = mean_metric.compute()
        train_acc = acc_metric.compute()
        
        prog_bar.close()
        
        return train_loss, train_acc
    
    
    def validate(self, train_config: dataclass, epoch_idx: int, device: str  = "cuda"):
        
        self.model.eval()
        
        acc_metric = MulticlassAccuracy(num_classes = self.val_loader.dataset.__num_classes__)
        mean_metric = MeanMetric()
        
        status = f"Valid:\t{bold}Epoch: {epoch_idx}/ {self.total_epochs}{reset}"
        prog_bar = tqdm(self.val_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        prog_bar.set_descroption(status)
        
        for batch_data, targets in prog_bar:
            
            batch_data, targets = batch_data.to(device), targets.to(device)
            
            outputs = self.model(batch_data)
            
            prob = F.softmax(outputs, dim = 1)
            
            val_loss = F.cross_entropy(outputs, targets).item()
            
            pred_idx = prob.detach().argmax(dim = 1)
            
            batch_loss = mean_metric(val_loss, weight = self.batch_data.shape[0])
            
            batch_acc = acc_metric(pred_idx.cpu(), targets.cpu())
            
            #Update prog bar description
            step_status = status + f"\tLoss: {mean_metric.compute():.4f}, Acc: {acc_metric.compute():.4f}"
            prog_bar.set_description(step_status)
            
        val_loss = mean_metric.compute()
        val_acc = acc_metric.compute()
        
        prog_bar.close()
        
        
        return val_loss, val_acc
            
        
        
    def run(self,
        DEVICE: torch.device, 
        model: nn.Module, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Union[optim.Adam, optim.SGD],
        scheduler: optim.lr_scheduler,
        ckpt_dir: str,
        summary_writer: torch.utils.tensorboard.writer.SummaryWriter,
        dataset_config: dataclass,
        train_config : dataclass
    ) -> dict:
        
        
        best_loss = torch.tensor(np.inf) #largest possible value from there descrease as it has to be minimum
        
        #epoch train/val loss
        epoch_train_loss = []
        epoch_val_loss = []
        
        #epoch train/test accuracy
        epoch_train_acc = []
        epoch_val_acc = []
        
        #training time measurement
        t_begin = time.time()
        
        for epoch in range(self.total_epochs):
            
            train_loss, train_acc = self.train_one_epoch(train_config, epoch_idx = epoch + 1, device=DEVICE)
            
            val_loss, val_acc = self.validate(train_config, epoch_idx = epoch + 1, device = DEVICE)
            
            train_loss_stat = f"{bold}Train Loss: {train_loss:.4f}{reset}"
            train_acc_stat = f"{bold}Train Acc: {train_acc:.4f}{reset}"
            
            val_loss_stat = f"{bold}Val Loss: {val_loss:.4f}{reset}"
            val_acc_stat = f"{bold}Val Acc: {val_acc:.4f}{reset}"
            
            print(f"\n{train_loss_stat:<30}{train_acc_stat}")
            print(f"{val_loss_stat:<30}{val_acc_stat}")
            
            
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
            
            epoch_val_loss.append(val_loss)
            epoch_val_acc.loss.append(val_acc)  
            
            summary_writer.add_scalars('Loss/train-val', {'train': train_loss,
                                                          'validation':val_acc}, epoch)
            
            summary_writer.add_scalars('Accuracy/train-val', {'train': train_acc, 
                                                              'validation': val_acc}, epoch)
            
            
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"\Model Improved ... Saving Model ...", end = "")
                best_weights = copy.deepcopy(model.state_dict())     
                torch.save(model.state_dict(), os.path.join(train_config.ckpt_dir, "best.pth")) 
                print("Done.\n")
                
        print(f"Total time: {(time.time() - t_begin):.2f}s, Best Loss: {best_loss:.3f}")    


        self.load_state_dict(best_weights)
        
        history = dict(
            model = model, 
            train_loss = epoch_train_loss,
            train_acc = epoch_train_acc,
            valid_loss = epoch_val_loss,
            valid_acc = epoch_val_acc,
            train_config = train_config,
            dataset_config = dataset_config
        )
        
        
        return history
            
            
            
            
        
        
        
        
       
        

