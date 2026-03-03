import torch as T
from torch.utils.data import DataLoader, random_split
from torch import nn
import torchvision.transforms as transf
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10
import MyDL

f_name = '.'.join(__file__.split('.')[:-1])
mod_path = f'{f_name}.pth'
check_path = f'{f_name}_ch.pt'

dev = T.device("cuda" if T.cuda.is_available() else "cpu")
b_size = 64
epochs = 8
lr = 0.02
trans = transf.Compose([
    transf.Resize(224),
    transf.ToTensor()
])

train_set, _ = random_split(CIFAR10('./data', transform=trans, download=True), (1000, 49000))
train_load = DataLoader(train_set, b_size, True)
test_set, _= random_split(CIFAR10('./data', False, trans, download=True), (300, 9700))
test_load = DataLoader(test_set, b_size)
train_loss = None
test_loss = []

mod = None

try:
    mod = T.load(mod_path, weights_only=False)
    mod = mod.to(dev)
    print('Found saved model')
except FileNotFoundError:
    mod = resnet18(weights=ResNet18_Weights.DEFAULT)
    mod = mod.to(dev)
    print('Downloaded model:')
    print(mod)

    mod.fc = nn.Linear(mod.fc.in_features, 10)

loss_fn = nn.CrossEntropyLoss()
opter = T.optim.SGD(mod.parameters(), lr=lr)

trainer = MyDL.Trainer(mod, opter, loss_fn, dev)
train_loss, test_loss = trainer.fit(train_load, epochs, check_path, test_load)
T.save(mod, mod_path)
MyDL.graph_loss(train_loss, test_loss)
    
cm = MyDL.conf_mat(mod, test_load)
total = cm.sum().item()
corr = cm.diag().sum().item()
acc = 100 * corr / total

print(f'Accuracy: {acc}%')