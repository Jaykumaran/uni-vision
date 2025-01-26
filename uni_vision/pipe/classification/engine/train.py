import random

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger_tb.tb_log import setup_logdir
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
        
        
        
    
    def train_one_epoch(self, train_config: dataclass, epoch_idx: int, device: str  = "cuda", class_weight: int = None):
        
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
    
    
    def validate():
        
        
        
    def run():
        
        

