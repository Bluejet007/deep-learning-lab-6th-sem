import torch as T
import torchvision as TV
from torch.utils.data import DataLoader
from torch import nn

dev = T.device('cuda' if T.cuda.is_available() else 'cpu')
transform = TV.transforms.Compose([
    TV.transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray to a float tensor
    TV.transforms.Normalize((0.1307,), (0.3081,)) # Normalizes the data (optional but recommended)
])
train_data = TV.datasets.MNIST('./data', download=True, transform=transform)
test_data = TV.datasets.MNIST('./data', train=False, download=True, transform=transform)

for raw, label in DataLoader(train_data, shuffle=True):
    inp = raw.flatten()
    print(inp.shape)
    print(inp)

class MINISTModel(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.lin1 = nn.Linear(784, 16, dtype=T.float32)
        self.act1 = nn.ReLU()
        self.lin1 = nn.Linear(16, 8, dtype=T.float32)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(8, 10)
        self.act2 = nn.Softmax()

    def forward(self, x: T.Tensor):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return self.act2(x)

mod = MINISTModel().to(dev)
loss_fn = nn.CrossEntropyLoss()
opter = T.optim.SGD(mod.parameters(), lr=0.03)