import torch as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

T.manual_seed(7)
X = T.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]]) 
Y = T.tensor([0.0, 1.0, 1.0, 0.0])
dev = T.device('cuda' if T.cuda.is_available() else 'cpu')
batch = 1
eps = 1000
lr = 0.05

class MyDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].to(dev), self.Y[idx].to(dev)

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(2, 2, dtype=T.float32)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(2, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: T.Tensor):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.act2(x)


full_data = MyDataset(X, Y)
data_load = DataLoader(full_data, batch_size=batch, shuffle=True)
mod = XORModel().to(dev)
print(mod)
loss_fn = nn.BCELoss()
opter = T.optim.SGD(mod.parameters(), lr=lr)

def one_epoch():
    total = 0.0
    for x, y in data_load:
        opter.zero_grad()
        p = mod(x)

        loss = loss_fn(p.flatten(), y)
        loss.backward()
        opter.step()

        total += loss.item()

    return total / (len(data_load) * batch)

print()
loss_l = []
for e in range(eps):
    mod.train(True)

    avg_loss = one_epoch()
    loss_l.append(avg_loss)

    if e % (eps // 10) == 0:
        print(f'EP {e}/{eps}, loss: {avg_loss}')

print()
for param in mod.named_parameters():
    print(param)

plt.plot(loss_l)
plt.title('Log-loss vs Epochs')
plt.show()