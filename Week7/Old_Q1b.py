import torch as T
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from torchvision.models import AlexNet_Weights, alexnet
import MyDL

f_name = '.'.join(__file__.split('.')[:-1])
mod_path = f'{f_name}.pth'
check_path = f'{f_name}_ch.pt'

dev = T.device("cuda" if T.cuda.is_available() else "cpu")
b_size = 64
epochs = 3
lr = 0.02
trans = transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])])

train_set = MyDL.AlexDataset(trans)
train_load = DataLoader(train_set, b_size, True)
test_set = MyDL.AlexDataset(trans, 'validation')
test_load = DataLoader(test_set, 64)

mod = None

try:
    mod = T.load(mod_path, weights_only=False)
    mod = mod.to(dev)
    print('Found saved model')
except:
    mod = alexnet(weights=AlexNet_Weights.DEFAULT)
    mod = mod.to(dev)
    print('Downloaded model:')
    print(mod)

    for param in mod.features.parameters():
        param.requires_grad = False

    num_ftrs = mod.classifier[6].in_features
    mod.classifier[6] = nn.Linear(num_ftrs, 2)

    loss_fn = nn.CrossEntropyLoss()
    opter = T.optim.SGD(mod.parameters(), lr=lr)

    trainer = MyDL.Trainer(mod, opter, loss_fn, dev)
    trainer.fit(train_load, epochs, check_path)
    T.save(mod, mod_path)

cm = MyDL.conf_mat(mod, test_load, 2)
total = cm.sum().item()
corr = cm.diag().sum().item()
acc = 100 * corr / total

print(f'Accuracy: {acc}%')