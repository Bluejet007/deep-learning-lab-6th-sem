import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import MyDL

f_name = '.'.join(__file__.split('.')[:-1])
mod_path = f'{f_name}.pth'

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
    mod = T.load(mod_path, weights_only=False)
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
T.save(mod, mod_path)

cm = MyDL.conf_mat(trainer.mod, test_load)
total = cm.sum().item()
corr = cm.diag().sum().item()
acc = 100 * corr / total

print(f'Accuracy: {acc}%')