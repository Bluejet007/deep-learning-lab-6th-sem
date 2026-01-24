import torch as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import matplotlib.pyplot as plt

x = T.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
y = T.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])
lr = 0.001
eps = 3

class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

class RegressionModel(nn.Module):
    def __init__(self, in_f: int):
        super().__init__()
        self.lin = nn.Linear(in_f, 1)

    def forward(self, x):
        return self.lin.forward(x)

    def err(self, y, p):
        return (y - p) ** 2

myData = MyDataset(x, y)
mod = RegressionModel(1)
oper = SGD(mod.parameters(), lr=lr)
loss_ep = []

for _ in range(eps):
    loss = 0.0

    for inv, outv in DataLoader(myData, shuffle=True):
        p = mod(inv)
        err = mod.err(outv, p)
        err.backward()
        oper.step()
        oper.zero_grad()
        loss += err

    loss /= len(x)
    loss_ep.append(loss.item())


print(f'w = {mod.lin.weight.item()}')
print(f'b = {mod.lin.bias.item()}')
plt.plot(loss_ep)
plt.title('MSE vs Epochs')
plt.show()