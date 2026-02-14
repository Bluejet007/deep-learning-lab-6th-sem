import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import MyDL

dev = T.device('cuda' if T.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 5
lr = 0.03

train_set = FashionMNIST('./data', transform=ToTensor(), download=True) 
train_load = DataLoader(train_set, batch_size, False)
test_set = FashionMNIST('./data', False, ToTensor(), download=True) 
test_load = DataLoader(test_set, batch_size, False)

mod: nn.Module  = None
try:
    mod = T.load('./models/Q1.pt', weights_only=False)
except FileNotFoundError:
    mod = MyDL.CNNClassifier()
mod.to(dev)

print('State dict:')
for param in mod.state_dict().keys():
    print(f'{param}: {mod.state_dict()[param].size()}')

loss_fn = nn.CrossEntropyLoss()
opter = T.optim.SGD(mod.parameters(), lr=lr)

trainer = MyDL.Trainer(mod, opter, loss_fn, dev)
trainer.fit(train_load, epochs)
T.save(mod, './models/Q1.pt')

cm = trainer.conf_mat(test_load)
total = cm.sum().item()
corr = cm.diag().sum().item()
acc = 100 * corr / total

print(f'Accuracy: {acc}%')