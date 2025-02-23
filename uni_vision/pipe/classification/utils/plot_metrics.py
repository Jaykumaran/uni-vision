import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import os

plt.rcParams['figure.figsize'] = (12,9)
plt.style.use('ggplot')

bold = f"\033[1m" ; reset = f"\033[1m"


def plot_metrics(
    train_config,
    metrics,
    num_epochs,
    title = None,
    ylabel = None,
    ylim = None,
    metric_name = None,
    color = None, 
    save_name = None 
):
    
    fig, ax = plt.subplots(figsize = (6,4))
    
    for idx, metric in enumerate(metrics):
        ax.plot(metric, color = color[idx], label = metric_name[idx])
    
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title if title else ylabel)
    plt.xlim([0, num_epochs - 1]) #will start from 0 in plotting
    plt.ylim(ylim)
    
    #tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(base = 5)) #interval between ticks
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(base = 1))
    
    plt.grid(True, linestyle = "--", alpha = 0.7)
    plt.legend(metric_name)
    plt.savefig(f"{os.path.join(train_config.checkpoint_dir, save_name)}.jpg")
    plt.show(block = False)
    plt.close()
    
    
    
    






