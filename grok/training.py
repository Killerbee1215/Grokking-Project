import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import prepare_loader
from data import prepare_loader_k
import os
from argparse import ArgumentParser
from model import Transformer, MLP, LSTMModel

def train(model_type, device, training_fraction, batch_size, max_steps, dropout, weight_decay, start_lr, optimizer_type):
    if model_type == "transformer":
        model = Transformer(num_layers=2, dim_model=128, num_heads=4, num_tokens=99, seq_len=4, dropout=dropout).to(device)
    elif model_type == "MLP":
        model = MLP(dim_model=128, num_tokens=99, seq_len=4).to(device)
    elif model_type == "LSTM":
        model = LSTMModel(num_layers=2, dim_model=128, hidden_dim=512, num_tokens=99, seq_len=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9,0.98),lr=start_lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay)

    train_loader, val_loader = prepare_loader(training_fraction, batch_size)
    n_epochs = max_steps // len(train_loader)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    steps = []

    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_train_loss, epoch_train_accuracy = 0, 0
        for i, tr_batch in enumerate(train_loader):
            tr_batch = [t.to(device) for t in tr_batch]
            inputs, targets = tr_batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            accuracy = np.mean(np.argmax(outputs.cpu().detach().numpy(), axis=1) == targets.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            # Store loss and accuracy for training
            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy

        # Average the metrics for the epoch
        train_losses.append(epoch_train_loss / len(train_loader))
        train_accuracies.append(epoch_train_accuracy / len(train_loader))
        steps.append(epoch * len(train_loader))

        model.eval()
        epoch_val_loss, epoch_val_accuracy = 0, 0
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                val_batch = [t.to(device) for t in val_batch]
                inputs, targets = val_batch
                outputs = model(inputs)
                loss = criterion(outputs, targets.long())
                accuracy = np.mean(np.argmax(outputs.cpu().detach().numpy(), axis=1) == targets.cpu().detach().numpy())

                # Store loss and accuracy for validation
                epoch_val_loss += loss.item()
                epoch_val_accuracy += accuracy

        # Average the metrics for the validation set
        val_losses.append(epoch_val_loss / len(val_loader))
        val_accuracies.append(epoch_val_accuracy / len(val_loader))
        print(f"Epoch {epoch}, train loss {train_losses[-1]}, train accuracy {train_accuracies[-1]}")
        print(f"Epoch {epoch}, val loss {val_losses[-1]}, val accuracy {val_accuracies[-1]}")
    return train_losses, train_accuracies, val_losses, val_accuracies, steps

def train_k(model_type, device, training_fraction, batch_size, max_steps, dropout, weight_decay, start_lr, optimizer_type, num_operands=2, modulus=97):
    

    num_tokens = modulus + 2  # modulus 个数字 + op_add + op_eq
    seq_len = num_operands * 2   # 例如，k=2 时：[x1, op_add, x2, op_eq]

    if model_type == "transformer":
        model = Transformer(num_layers=2, dim_model=128, num_heads=4, num_tokens=num_tokens, seq_len=seq_len, dropout=dropout).to(device)
    elif model_type == "MLP":
        model = MLP(dim_model=128, num_tokens=num_tokens, seq_len=seq_len).to(device)
    elif model_type == "LSTM":
        model = LSTMModel(num_layers=2, dim_model=128, hidden_dim=512, num_tokens=num_tokens, seq_len=seq_len).to(device)
    

    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=start_lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, weight_decay=weight_decay)

    #通过prepare_loader_k函数生成k个数的数据
    train_loader, val_loader = prepare_loader_k(training_frac=training_fraction, batch_size=batch_size, num_operands=num_operands, modulus=modulus)

    n_epochs = max_steps // len(train_loader)
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    steps = []
    #补充对weight_norms的记录
    weight_norms=[]


    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_train_loss, epoch_train_accuracy = 0, 0
        for i, tr_batch in enumerate(train_loader):
            tr_batch = [t.to(device) for t in tr_batch]
            inputs, targets = tr_batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            accuracy = np.mean(np.argmax(outputs.cpu().detach().numpy(), axis=1) == targets.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy
        
        # 计算 weight norm
        with torch.no_grad():
            total_weight_norm = 0
            for name, param in model.named_parameters():
                if 'weight' in name: 
                    total_weight_norm += torch.norm(param).item()
            weight_norms.append(total_weight_norm)

        train_losses.append(epoch_train_loss / len(train_loader))
        train_accuracies.append(epoch_train_accuracy / len(train_loader))
        steps.append(epoch * len(train_loader))

        model.eval()
        epoch_val_loss, epoch_val_accuracy = 0, 0
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                val_batch = [t.to(device) for t in val_batch]
                inputs, targets = val_batch
                outputs = model(inputs)
                loss = criterion(outputs, targets.long())
                accuracy = np.mean(np.argmax(outputs.cpu().detach().numpy(), axis=1) == targets.cpu().detach().numpy())

                epoch_val_loss += loss.item()
                epoch_val_accuracy += accuracy

        val_losses.append(epoch_val_loss / len(val_loader))
        val_accuracies.append(epoch_val_accuracy / len(val_loader))
        print(f"Epoch {epoch}, train loss {train_losses[-1]}, train accuracy {train_accuracies[-1]}")
        print(f"Epoch {epoch}, val loss {val_losses[-1]}, val accuracy {val_accuracies[-1]}")

    #增加了对weight_norms的输出
    return train_losses, train_accuracies, val_losses, val_accuracies, steps, weight_norms


def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--start_lr", type=float, default=0.001)
    parser.add_argument("--optimizer_type", type=str, default="AdamW")

    #新加的参数
    parser.add_argument("--num_operands", type=int, default=2)
    parser.add_argument("--modulus", type=int, default=97)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #使用train_k
    train_loss, train_acc, val_loss, val_acc, steps ,weight_norms= train_k(args.model, device, args.training_fraction, args.batch_size, args.max_steps, args.dropout, args.weight_decay, args.start_lr, args.optimizer_type, args.num_operands, args.modulus)

    plt.plot(steps, train_acc, label='Training Accuracy')
    plt.plot(steps, val_acc, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

if __name__ == "__main__":
    main()
