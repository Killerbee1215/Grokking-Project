'''
Reproducing the results of subtask 3 with regular plot
'''
import os
import sys
# Add the folder path to the sys.path
sys.path.append("D:\Grokking")
sys.path.append("D:\Grokking\grok")
from grok.training import train
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = 'results'
    training_pct_values = np.linspace(0.2, 0.8, 7)
    # Set the figure for regular plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # Change the weight decay parameter
    weight_decay_values = [0.01, 0.1, 1]
    for i, weight_decay in enumerate(weight_decay_values):
        max_val_acc_list = []
        for training_pct in training_pct_values:
            train_loss, train_acc, val_loss, val_acc, steps = train("transformer", device, training_pct, 512, 500000, 0, weight_decay, 0.001, "AdamW")
            max_val_acc = np.max(val_acc)
            max_val_acc_list.append(max_val_acc)
        axs[0, i].plot(training_pct_values, max_val_acc_list)
        axs[0, i].set_title(f'AdamW, Weight Decay: {weight_decay}')

    # Change the dropout parameter
    dropout_values = [0, 0.1, 0.5]
    for i, dropout in enumerate(dropout_values):
        max_val_acc_list = []
        for training_pct in training_pct_values:
            train_loss, train_acc, val_loss, val_acc, steps = train("transformer", device, training_pct, 512, 500000, dropout, 0.05, 0.001, "AdamW")
            max_val_acc = np.max(val_acc)
            max_val_acc_list.append(max_val_acc)
        axs[1, i].plot(training_pct_values, max_val_acc_list)
        axs[1, i].set_title(f'AdamW, Dropout: {dropout}')

    # fig.text(0.5, 0.04, 'Training data fraction', ha='center', va='center', fontsize=12)
    # fig.text(0.04, 0.5, 'Best validation accuracy', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "subtask_3_regular.png"))
    plt.show()

if __name__ == "__main__":
    main()