'''
Reproducing the results of subtask 3 with regular plot
'''
import os
import sys
# Add the folder path to the sys.path
sys.path.append("/lustre/home/1900017859/Grokking")
sys.path.append("/lustre/home/1900017859/Grokking/grok")
from grok.training import train
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = 'results'
    training_pct_values = np.linspace(0.2, 0.8, 7)
    # Set the figure for optimizer plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # Change the batch size parameter
    batch_size_values = [64, 512, 4096]
    for i, batch_size in enumerate(batch_size_values):
        max_val_acc_list = []
        for training_pct in training_pct_values:
            train_loss, train_acc, val_loss, val_acc, steps = train("transformer", device, training_pct, batch_size, 500000, 0, 0.05, 0.001, "AdamW")
            max_val_acc = np.max(val_acc)
            max_val_acc_list.append(max_val_acc)
        axs[0, i].plot(training_pct_values, max_val_acc_list)
        axs[0, i].set_title(f'AdamW, Batch size: {batch_size}')

    # Change the start lr parameter
    start_lr_values = [3e-3, 3e-4]
    for i, start_lr in enumerate(start_lr_values):
        max_val_acc_list = []
        for training_pct in training_pct_values:
            train_loss, train_acc, val_loss, val_acc, steps = train("transformer", device, training_pct, 512, 500000, 0, 0.05, start_lr, "AdamW")
            max_val_acc = np.max(val_acc)
            max_val_acc_list.append(max_val_acc)
        axs[1, i].plot(training_pct_values, max_val_acc_list)
        axs[1, i].set_title(f'AdamW, Start lr: {start_lr}')
    
    # Change the optimizer type parameter
    max_val_acc_list = []
    for training_pct in training_pct_values:
        train_loss, train_acc, val_loss, val_acc, steps = train("transformer", device, training_pct, 512, 500000, 0, 0.05, 0.001, "SGD")
        max_val_acc = np.max(val_acc)
        max_val_acc_list.append(max_val_acc)
    axs[1, 2].plot(training_pct_values, max_val_acc_list)
    axs[1, 2].set_title(f'SGD')

    # fig.text(0.5, 0.04, 'Training data fraction', ha='center', va='center', fontsize=12)
    # fig.text(0.04, 0.5, 'Best validation accuracy', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "subtask_3_optimizer.png"))
    plt.show()

if __name__ == "__main__":
    main()