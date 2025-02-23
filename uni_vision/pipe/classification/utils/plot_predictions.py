import torch
import torch.nn as nn
from typing import List
from ..engine.image_transforms import denormalize
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from uni_vision.configs.train_config import TrainingConfig



def visualize_image_predictions(
    train_config: TrainingConfig,
    image_batch: torch.tensor,
    preds_batch: torch.tensor,
    target_batch: torch.tensor,
    class_names: List[int],
    total_samples: int,
    mode: str
):
    
    num_cols = min(total_samples, 5) # Max 5 columns
    num_rows = math.ceil(total_samples / num_cols)

    title_color = {
        "correct": "g",
        "incorrect": "r",
        "all": "blue" # def
    }[mode]
        
    
    font_format = {
        "family": "sans-serif",
        'size': 16
    }
    
    # fig = plt.figure(figsize = (15, 10), layout = 'constrained')
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize = (num_cols * 3, num_rows * 3))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    axes = axes.flatten() if num_rows > 1 else [axes]
    
    for i, (image, preds, target) in enumerate(zip(image_batch, preds_batch, target_batch)):
        
        if i >= total_samples:
            break
            
        image_np = (image.numpy()*255).astype(np.uint8)
        
        # Set subplot
        ax = axes[i]
        ax.imshow(image_np)
        ax.axis('off')
         
        title = f"GT: {class_names[int(target)]}, Pred: {class_names[int(preds[1])]}"
        ax.set_title(title, fontdict = font_format, color = title_color)

    
    # Hide any extra axes (if total samples is not a perfect multiple of num_cols)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    
    plt.savefig(f"{os.path.join(train_config.checkpoint_dir, 'prediction_canvas')}.jpg")
    plt.show()


def prediction_batch(model: nn.Module, batch_inputs: torch.tensor):
    
    model.eval()
    
    with torch.no_grad():
        batch_ops = model(batch_inputs)
    
    batch_probs = batch_ops.softmax(dim = 1)
    batch_confs , batch_cls_ids = batch_probs.max(dim = 1) # get max, still arg max not applied. 
    
    return torch.stack([batch_confs.cpu(), batch_cls_ids.cpu()], dim = 1)

def plot_predictions(
    train_config: TrainingConfig,

    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device = "cpu",
    mean: torch.tensor = torch.tensor([0.485, 0.456, 0.406]),
    std: torch.tensor = torch.tensor([0.229, 0.224, 0.225]),
    mode : str = "correct",
    num_samples: int = 10,
):
    model = model.eval().to(device)
    
    images_to_plot = []
    preds_to_plot = []
    targets_to_plot = []
    
    count_num_preds = 0
    
    for i, (img_batch, target_batch) in enumerate(data_loader):
        
        batch_images = img_batch.to(device)
        
        pred_batches = prediction_batch(model, batch_images)
        
        if mode == "correct":
            keep_ids = pred_batches[:, 1] == target_batch
        
        elif mode == "incorrect":
            keep_ids = pred_batches[:, 1] != target_batch
        
        elif mode == "all":
            keep_ids = torch.ones_like(target_batch, dtype=torch.bool) # Select all samples from the batch
        else:
            raise ValueError("mode should be either: correct or incorrect")
        
        count_num_preds += keep_ids.sum()
        
        image_batch_denorm = denormalize(img_batch).clamp(min=0, max = 1)
        
        #Reshape images to (B,H,W,C) to be plotted
        image_batch_denorm = image_batch_denorm.permute(0, 2, 3, 1)
        
        images_to_plot.append(image_batch_denorm[keep_ids])
        preds_to_plot.append(pred_batches[keep_ids])
        targets_to_plot.append(target_batch[keep_ids])
        
        if count_num_preds > num_samples:
            break
    
    #Concatenate all the images, predictions and targets list
    images_to_plot = torch.cat(images_to_plot)
    preds_to_plot = torch.cat(preds_to_plot)
    targets_to_plot = torch.cat(targets_to_plot)
    
    visualize_image_predictions(
        train_config= train_config,
        image_batch = images_to_plot,
        preds_batch=preds_to_plot,
        target_batch=targets_to_plot,
        class_names=class_names,
        total_samples=num_samples,
        mode=mode,

    )
    
    
    return 