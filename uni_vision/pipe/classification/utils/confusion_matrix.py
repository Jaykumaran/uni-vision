import matplotlib.pyplot as plt
# from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd



def get_confusion_matrix(model, val_loader, save_path : str = None):
    
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    #Collect true labels and predictions
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1) #Get the predicted class index
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    #Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #Define class labels
    class_names = [f'({i:02d}) {cls_name}' for i, cls_name in enumerate(val_loader.dataset.classes)]
    
    #Save the confusion matrix to a Dataframe
    if save_path:
        cm_df = pd.DataFrame(cm, index = class_names, columns = class_names)
        
        cm_df.to_csv(save_path)
        print(f"Confusion matrix saved to: {save_path}")
        
    return cm, class_names


def display_confusion_matrix(model, val_loader, title = "Confusion Matrix", save_path: str = None):
    
    cm, class_names = get_confusion_matrix(model, val_loader, save_path)
    
    #Plot the confusion matrix
    fig = plt.figure(figsize = (15, 12))
    ax = fig.add_subplot(nrows = 1, ncols = 1 , index = 1)
    sns.heatmap(cm, annot = True, fmt="d", cmap = "coolwarm", xticklabels=class_names, yticklabels=class_names)
    ax.set_title(title)
    ax.set_ylabel("Actual labels")
    ax.set_xlabel("Predicted labels")
    ax.set_xticklabels(class_names, rotation = 90)
    ax.set_yticklabels(class_names, rotation = 90)
    return fig
        
    