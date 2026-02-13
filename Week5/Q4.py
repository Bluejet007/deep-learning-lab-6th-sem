import torch as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

dev = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('Using device:', dev)

batch_size = 64
epochs = 5
lr = 0.03

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST('./data', transform=transform, download=True)
test_dataset = MNIST('./data', False, transform, download=True)

train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(test_dataset, batch_size, True)

class CNNClassifier(nn.Module):
    def __init__(self, red=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64 // red, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64 // red, 128 // red, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128 // red, 64 // red, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class Trainer:
    def __init__(self, model, optimiser, criterion, device):
        self.mod = model.to(device)
        self.opter = optimiser
        self.loss_fn = criterion
        self.dev = device

    def one_epoch(self, loader):
        self.mod.train()
        running_loss = 0.0

        for x, y in loader:
            x = x.to(self.dev, non_blocking=True)
            y = y.to(self.dev, non_blocking=True)

            self.opter.zero_grad()
            logits = self.mod(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.opter.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def fit(self, loader, eps):
        loss_l = []
        for e in range(eps):
            avg_loss = self.one_epoch(loader)
            loss_l.append(avg_loss)

            print(f'Epoch [{e+1}/{eps}], loss = {avg_loss}')

        return loss_l
    
    def conf_mat(self, loader, num_classes=10):
        self.mod.eval()
        cm = T.zeros(num_classes, num_classes, dtype=T.int64)

        with T.no_grad():
            for x, y in loader:
                x = x.to(self.dev, non_blocking=True)
                y = y.to(self.dev, non_blocking=True)

                logits = self.mod(x)
                preds = T.argmax(logits, dim=1)

                for t, p in zip(y.view(-1), preds.view(-1)):
                    cm[t.long(), p.long()] += 1

        return cm

reds = [1, 2, 4]
cm = None
for red in reds:
    mod = CNNClassifier()
    loss_fn = nn.CrossEntropyLoss()
    opter = T.optim.SGD(mod.parameters(), lr=lr)

    trainer = Trainer(mod, opter, loss_fn, dev)
    losses = trainer.fit(train_loader, epochs)
    cm = trainer.conf_mat(test_loader)
    plt.plot(losses)

print(cm)
plt.legend(reds)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')
plt.show()