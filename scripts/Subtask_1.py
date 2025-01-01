'''
Reproducing the results of subtask 1
'''
import os
import sys
# Add the folder path to the sys.path
sys.path.append("D:\Grokking")
sys.path.append("D:\Grokking\grok")
from grok.training import train
import torch
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = 'results'
    train_loss_1, train_acc_1, val_loss_1, val_acc_1, steps_1 = train("transformer", device, 0.3, 512, 100, 0, 0.05, 0.001, "AdamW")
    train_loss_2, train_acc_2, val_loss_2, val_acc_2, steps_2 = train("transformer", device, 0.5, 512, 100, 0, 0.05, 0.001, "AdamW")
    train_loss_3, train_acc_3, val_loss_3, val_acc_3, steps_3 = train("transformer", device, 0.7, 512, 100, 0, 0.05, 0.001, "AdamW")
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(steps_1, train_acc_1, label='Train Acc')
    axs[0].plot(steps_1, val_acc_1, label='Val Acc')
    axs[0].set_title('Accuracy of 30% training data')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xscale('log')
    axs[0].legend()

    axs[1].plot(steps_2, train_acc_2, label='Train Acc')
    axs[1].plot(steps_2, val_acc_2, label='Val Acc')
    axs[1].set_title('Accuracy of 50% training data')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xscale('log')
    axs[1].legend()

    axs[2].plot(steps_3, train_acc_3, label='Train Acc')
    axs[2].plot(steps_3, val_acc_3, label='Val Acc')
    axs[2].set_title('Accuracy of 70% training data')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('Accuracy')
    axs[2].set_xscale('log')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "subtask_1.png"))
    plt.show()

if __name__ == "__main__":
    main()