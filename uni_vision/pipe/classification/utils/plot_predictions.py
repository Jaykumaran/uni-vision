import torch
import torch.nn as nn
from typing import List
from engine.image_transforms import denormalize
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from configs.train_config import TrainingConfig



def visualize_image_predictions(
    image_batch: torch.tensor,
    preds_batch: torch.tensor,
    target_batch: torch.tensor,
    class_names: List[int],
    total_samples: int,
    mode: str
):
    
    num_cols = 5
    num_rows = math.ceil(total_samples / num_cols)

    if mode == "correct":
        title_color = "g"
    elif mode ==  "incorrect":
        title_color = "r"
    
    font_format = {
        "family": "sans-serif",
        'size': 16
    }
    
    fig = plt.figure(figsize = (24, 12), layout = 'constrained')
    
    for i, (image, preds, target) in enumerate(zip(image_batch, preds_batch, target_batch)):
        
        if i >= total_samples:
            break
            
        image_np = (image.numpy()*255).astype(np.uint8)
         
        ax = plt.subplot(num_cols, num_cols, i + 1)
        title = f"Tar: {class_names[int(target)]}, Pred: {class_names[int(preds[1])]}"
        title += f'({float(preds[0]):.2f})'
        title_obj = plt.title(title, fontdict=font_format)

        plt.setp(title_obj, color = title_color)
        
        plt.axis('off')
        
        plt.imshow(image_np)
    

    plt.show()
    plt.savefig(f"{os.path.join(TrainingConfig.checkpoint_dir, "prediction_canvas")}.jpg")

    return






def prediction_batch(model: nn.Module, batch_inputs: torch.tensor):
    
    model.eval()
    
    with torch.no_grad():
        batch_ops = model(batch_inputs)
    
    batch_probs = batch_ops.softmax(dim = 1)
    batch_confs , batch_cls_ids = batch_ops.max(dim = 1)
    
    return torch.stack([batch_confs.cpu(), batch_cls_ids.cpu()], dim = 1)

def plot_predictions(
    model: nn.Module,
    data_loader: torch.utils.DataLoader,
    class_names: List[str],
    device = "cpu",
    mean: torch.tensor = torch.tensor([0.485, 0.456, 0.406]),
    std: torch.tensor = torch.tensor([0.229, 0.224, 0.225]),
    mode : str = "correct",
    num_samples: int = 10,
):
    model = model.eval.to(device)
    
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
    targets_to_plot = torch.cat(preds_to_plot)
    
    visualize_image_predictions(
        image_batch = images_to_plot,
        preds_batch=preds_to_plot,
        target_batch=targets_to_plot,
        class_names=class_names,
        total_samples=num_samples,
        mode=mode,

    )
    
    
    return 