import torch as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import matplotlib.pyplot as plt

x = T.tensor([
    [3.0, 8.0],
    [4.0, 5.0],
    [5.0, 7.0],
    [6.0, 3.0],
    [2.0, 1.0]
              ])
y = T.tensor([-3.7, 3.5, 2.5, 11.5, 5.7])
lr = 0.001
eps = 5

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
        self.w = nn.Parameter(T.rand([in_f], requires_grad=True))
        self.b = nn.Parameter(T.rand([1], requires_grad=True))

    def forward(self, x):
        return self.w @ x.T + self.b

    def err(self, y, p):
        return (y - p) ** 2

myData = MyDataset(x, y)
mod = RegressionModel(2)
oper = SGD(mod.parameters(), lr=lr)
loss_ep = []

for _ in range(eps):
    loss = 0.0

    for inv, outv in DataLoader(myData):
        p = mod(inv)
        err = mod.err(outv, p)
        err.backward()
        oper.step()
        oper.zero_grad()
        loss += err

    loss /= len(x)
    loss_ep.append(loss.item())


print(f'w = {mod.w.detach().numpy()}')
print(f'b = {mod.b.item()}')
plt.plot(loss_ep)
plt.title('MSE vs Epochs')
# plt.show() # No graph needed for lab book

print(f'x1 = 3, x2 = 2')
print(f'p = {mod(T.tensor([3.0, 2.0])).item()}')