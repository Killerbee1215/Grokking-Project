'''
Reproducing the results of subtask 2
'''
import os
import sys
# Add the folder path to the sys.path
sys.path.append("D:\Grokking")
sys.path.append("D:\Grokking\grok")
from grok.training import train_k
import torch
import matplotlib.pyplot as plt

def plot_curves(steps, train_loss, train_acc, val_loss, val_acc, weight_norms, title_prefix, filename, save_folder='results'):

    # 绘制准确率曲线
    plt.plot(steps, train_acc, label='Training Accuracy')
    plt.plot(steps, val_acc, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.title(f'{title_prefix} Accuracy')
    plt.savefig(os.path.join(save_folder, f'{filename}_accuracy.png'))
    plt.show() 
    plt.clf()

    # 绘制损失曲线
    plt.plot(steps, train_loss, label='Training Loss')
    plt.plot(steps, val_loss, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.legend()
    plt.title(f'{title_prefix} Loss')
    plt.savefig(os.path.join(save_folder, f'{filename}_accuracy.png'))
    plt.show()
    plt.clf()

    # 绘制权重曲线
    plt.plot(steps, weight_norms, label='Weight Norm')
    plt.xlabel('Steps')
    plt.ylabel('Weight Norm')
    plt.xscale('log')
    plt.legend()
    plt.title(f'Weigh Norm')
    plt.savefig(os.path.join(save_folder, f'{filename}_accuracy.png'))
    plt.show() 
    plt.clf()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = 'results'
    #k=2,dropout=0
    train_loss, train_acc, val_loss, val_acc, steps, weight_norms = train_k("transformer", device,training_fraction=0.5, batch_size=1024, max_steps=50000, dropout=0, weight_decay=0.2, start_lr=0.001, optimizer_type="AdamW", num_operands=2, modulus=23)
    plot_curves(steps, train_loss, train_acc, val_loss, val_acc, weight_norms, 'Training and Validation_p=23_k=2', 'p=23_k=2',save_folder)
    #k=2,dropout=0.1
    train_loss, train_acc, val_loss, val_acc, steps, weight_norms = train_k("transformer", device,training_fraction=0.5, batch_size=1024, max_steps=50000, dropout=0.1, weight_decay=0.2, start_lr=0.001, optimizer_type="AdamW", num_operands=2, modulus=23)
    plot_curves(steps, train_loss, train_acc, val_loss, val_acc, weight_norms, 'Training and Validation_p=23_k=2_dropout', 'p=23_k=2_dropout',save_folder)
    
    #k=3,dropout=0
    train_loss, train_acc, val_loss, val_acc, steps, weight_norms = train_k("transformer", device,training_fraction=0.3, batch_size=1024, max_steps=50000, dropout=0, weight_decay=0.1, start_lr=0.001, optimizer_type="AdamW", num_operands=3, modulus=23)
    plot_curves(steps, train_loss, train_acc, val_loss, val_acc, weight_norms, 'Training and Validation_p=23_k=3', 'p=23_k=3',save_folder)
    #k=3,dropout=0.1
    train_loss, train_acc, val_loss, val_acc, steps, weight_norms = train_k("transformer", device,training_fraction=0.5, batch_size=1024, max_steps=50000, dropout=0.1, weight_decay=0.1, start_lr=0.001, optimizer_type="AdamW", num_operands=3, modulus=23)
    plot_curves(steps, train_loss, train_acc, val_loss, val_acc, weight_norms, 'Training and Validation_p=23_k=3_dropout', 'p=23_k=3_dropout',save_folder)

    #k=4,dropout=0
    train_loss, train_acc, val_loss, val_acc, steps, weight_norms = train_k("transformer", device,training_fraction=0.2, batch_size=2048, max_steps=50000, dropout=0, weight_decay=0.003, start_lr=0.001, optimizer_type="AdamW", num_operands=4, modulus=23)
    plot_curves(steps, train_loss, train_acc, val_loss, val_acc, weight_norms, 'Training and Validation_p=23_k=2', 'p=23_k=2',save_folder)
    #k=4,dropout=0.1
    train_loss, train_acc, val_loss, val_acc, steps, weight_norms = train_k("transformer", device,training_fraction=0.2, batch_size=2048, max_steps=50000, dropout=0.1, weight_decay=0.003, start_lr=0.001, optimizer_type="AdamW", num_operands=4, modulus=23)
    plot_curves(steps, train_loss, train_acc, val_loss, val_acc, weight_norms, 'Training and Validation_p=23_k=2_dropout', 'p=23_k=2_dropout',save_folder)

if __name__ == "__main__":
    main()