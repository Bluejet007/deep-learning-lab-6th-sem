import torch as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

x = T.tensor([1.0, 5.0, 10.0, 10.0, 25.0])
y = T.tensor([0, 0, 0, 0, 0, 1, 1, 1,])
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
        return (1 + T.exp(-(self.w @ x.T + self.b))) ** -1

    def err(self, y, p):
        p = T.clamp(p, 1e-7, 1 - 1e-7)
        return -(y * T.log(p) + (1 - y) * T.log(1 - p))

myData = MyDataset(x, y)
mod = RegressionModel(1)
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