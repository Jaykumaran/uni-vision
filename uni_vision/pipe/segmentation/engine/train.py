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
from logger_tb.tb_log import setup_logdir
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from configs.train_config import TrainingConfig
from utils.plot_metrics import plot_metrics
from utils.plot_predictions import plot_predictions

from uni_vision.pipe.segmentation.engine.seg_losses import dice_coef_loss, mean_iou


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
        
    
    def train_one_epoch(self, epoch_idx: int, device: Union[str, int]  = "cuda" , class_weight: Optional[torch.tensor ]= None):
        
        self.model.train()
        num_classes = self.train_loader.dataset.__num_classes__
        metric_record = MeanMetric()
        loss_record = MeanMetric()
        acc_record = MulticlassAccuracy(num_classes = num_classes, average = "micro")
       
        
        status = f"Train:\t{bold} Epoch: {epoch_idx} / {self.total_epochs}{reset}"
        prog_bar = tqdm(self.train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        prog_bar.set_description(status)
        
        for (batch_data, target) in enumerate(prog_bar):
            batch_data, target = batch_data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_data) #logits
            
            #Have to include focal loss
            loss = dice_coef_loss(outputs, target, num_classes = num_classes)
        
            loss.backward()
            
            self.optimizer.step()
                    
            #Get the index of max probability
            pred_idx = outputs.detach().argmax(dim = 1)
            
            mean_iou_metric = mean_iou(pred_idx, target, num_classes = num_classes)
            
            acc_record.update(pred_idx.cpu(), target.cpu())
            
            loss_record.update(loss.detach().cpu(), weight = batch_data.shape[0])
            metric_record.update(mean_iou_metric.cpu())
            
            #Update progress bar description
            step_status = status + f"\tLoss: {loss_record.compute():.4f}, IoU: {metric_record.compute():4f}, Acc: {acc_record.compute():4f}"
            prog_bar.set_description(step_status)
            
    
        #Per epoch
        train_loss = loss_record.compute()
        train_iou = metric_record.compute()
        train_acc = acc_record.compute()
        
        prog_bar.close()
        
        return train_loss, train_iou, train_acc
    
    
    def validate(self, epoch_idx: int, device: str  = "cuda"):
        
        self.model.eval()
        
        num_classes = self.train_loader.dataset.__num_classes__
        metric_record = MeanMetric()
        loss_record = MeanMetric()
        acc_record = MulticlassAccuracy(num_classes = num_classes, average = "micro")
        
        
        status = f"Valid:\t{bold}Epoch: {epoch_idx}/ {self.total_epochs}{reset}"
        prog_bar = tqdm(self.val_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        prog_bar.set_description(status)
        
        for batch_data, target in prog_bar:
            
            batch_data, target = batch_data.to(device), target.to(device)
            
            with torch.no_grad():
                outputs = self.model(batch_data)
                    
            val_loss = dice_coef_loss(outputs, target, num_classes=num_classes)
            
            pred_idx = outputs.argmax(dim = 1)
            
            #Calculate Segmentation specific metric (Dice or IoU)
            
            mean_iou_metric = mean_iou(pred_idx, target, num_classes=num_classes)
            
            acc_record.update(pred_idx.cpu(), target.cpu())
            loss_record.update(val_loss.cpu(), weight= batch_data.shape[0])
            
            metric_record.update(mean_iou_metric.cpu(), weight = batch_data.shape[0])
            
            
            #Update prog bar description
            step_status = status + f"\tLoss: {loss_record.compute():.4f}, IoU: {metric_record.compute():4f}, Acc: {acc_record.compute():4f}"
            prog_bar.set_description(step_status)
            
        val_loss = loss_record.compute()
        val_iou = metric_record.compute()
        val_acc = acc_record.compute()
        
        prog_bar.close()
        
        
        return val_loss, val_iou,  val_acc
            
        
        
    def run(self,
        DEVICE: torch.device, 
    ) -> dict:
        
        
        best_loss = torch.tensor(np.inf) #largest possible value from there descrease as it has to be minimum
        best_weights = None
        
        
        #epoch train/val loss
        epoch_train_loss, epoch_val_loss = [], []
        
        #epoch train/test accuracy
        epoch_train_acc, epoch_val_acc = [] , []
         
        
        #training time measurement
        t_begin = time.time()
        
        for epoch in range(self.total_epochs):
            
            #Training
            train_loss, train_acc = self.train_one_epoch(self.train_config, epoch_idx = epoch + 1, device=DEVICE)
            
            val_loss, val_acc = self.validate(self.train_config, epoch_idx = epoch + 1, device = DEVICE)
            
            
            train_loss_stat = f"{bold}Train Loss: {train_loss:.4f}{reset}"
            train_acc_stat = f"{bold}Train Acc: {train_acc:.4f}{reset}"
            
            val_loss_stat = f"{bold}Val Loss: {val_loss:.4f}{reset}"
            val_acc_stat = f"{bold}Val Acc: {val_acc:.4f}{reset}"
            
            print(f"\n{train_loss_stat:<30}{train_acc_stat}")
            print(f"{val_loss_stat:<30}{val_acc_stat}")
            
            
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
            
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)  
            
            self.tb_writer.add_scalars('Loss/train-val', {'train': train_loss,
                                                          'validation':val_acc}, epoch)
            
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
                print(f"\Model Improved ... Saving Model ...", end = "")
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
                print(F"Model Improved: Saved to {checkpoint_path}✅.\n")
                
        print(f"Total time: {(time.time() - t_begin):.2f}s, Best Loss: {best_loss:.3f}")    

  
               
        #Load best weights   
        self.model.load_state_dict(best_weights)
        
        history = dict(
            train_loss = epoch_train_loss,
            train_acc = epoch_train_acc,
            valid_loss = epoch_val_loss,
            valid_acc = epoch_val_acc,
        )
        
        
        plot_metrics(
            [train_acc, val_acc],
            ylabel = "Accuracy",
            ylim = [0.0, 1.1],
            metric_name = ["Training Accuracy", "Validation Accuracy"],
            color = ["b", "g"] ,#'b'
            num_epochs = self.total_epochs,
            save_name='accuracy_plot'
        )
        
        plot_metrics(
            [train_loss, val_loss],
            ylabel = "Loss",
            ylim = [0.0, 2.0],
            metric_name = ["Training Loss", "Validation Loss"],
            color = ["r", "y"],
            num_epochs=self.total_epochs,
            save_name='loss_curve_plot'
        )
        
        plot_predictions(model = self.model,data_loader=self.val_loader, class_names=self.val_loader.__classes__, mode = "correct", num_samples = 10)
        
        return history
            
            
            
            
        
        
        
        
       
        

