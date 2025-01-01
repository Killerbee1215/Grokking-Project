import torch
from math import ceil
import itertools
from torch.utils.data import DataLoader, TensorDataset, random_split

MODULUS = 97
NUMS = list(range(MODULUS))

def make_data():
    op_add, op_eq = MODULUS, MODULUS + 1
    tuples = itertools.product(NUMS, repeat=2) 
    eqs = []
    for a, b in tuples:
        c = (a + b) % MODULUS
        eq = [a, op_add, b, op_eq, c]
        eqs.append(eq)
    eqs = torch.tensor(eqs, dtype=torch.int64)
    return eqs


def prepare_loader(training_frac = 0.5, batch_size = 512):
    data = make_data()
    dataset = TensorDataset(data[:,:-1], data[:,-1])

    train_size = int(training_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = min(batch_size, ceil(len(dataset) / 2))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
