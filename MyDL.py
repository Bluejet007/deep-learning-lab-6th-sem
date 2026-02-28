import torch as T
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
import kagglehub
import glob
from PIL import Image

# Week 6
def conf_mat(mod: nn.Module, loader: DataLoader, num_classes: int=10) -> T.Tensor:
    mod.eval()
    dev = next(mod.parameters()).device
    cm = T.zeros(num_classes, num_classes, dtype=T.int64)

    with T.no_grad():
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            logits = mod(x)
            preds = T.argmax(logits, dim=1)

            for t, p in zip(y.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    return cm

# Week 3
class MyDataset(Dataset):
    def __init__(self, x: T.Tensor, y: T.Tensor):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index)  -> tuple[T.Tensor, T.Tensor]:
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return len(self.x)

class RegressionModel(nn.Module):
    def __init__(self, in_f: int):
        super().__init__()
        self.w = nn.Parameter(T.rand([in_f], requires_grad=True))
        self.b = nn.Parameter(T.rand([1], requires_grad=True))

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.w @ x.T + self.b

    def err(self, y: T.Tensor | float | int, p: T.Tensor | float | int) -> T.Tensor | float | int:
        return (y - p) ** 2


# Week 5
class CNNClassifier(nn.Module):
    def __init__(self, red: int=1):
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

    def forward(self, x) -> T.Tensor:
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class Trainer:
    def __init__(self, model: nn.Module, optimiser: T.optim.Optimizer, criterion: nn.Module, device: T.device):
        self.mod = model.to(device)
        self.opter = optimiser
        self.loss_fn = criterion
        self.dev = device

    def one_epoch(self, loader: DataLoader) -> T.Tensor:
        self.mod.train()
        loss = 0.0

        for x, y in loader:
            x = x.to(self.dev, non_blocking=True)
            y = y.to(self.dev, non_blocking=True)

            self.opter.zero_grad()
            logits = self.mod(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.opter.step()

            loss += loss.item()

        return loss / len(loader)

    def fit(self, loader: DataLoader, eps: int, check_path: str | None= None) -> list[T.Tensor]:
        loss_l = []
        start = 0
        
        if check_path is not None and os.path.exists(check_path):
            check = T.load(check_path)
            start = check['l_epoch'] + 1
            self.mod.load_state_dict(check['model_state'])
            self.opter.load_state_dict(check['opter_state'])
            print(f'Loading checkpoint after epoch {start}')

        for e in range(start, eps):
            avg_loss = self.one_epoch(loader)
            loss_l.append(avg_loss)

            print(f'Epoch [{e+1}/{eps}] loss: {avg_loss}')

            if check_path is not None:
                check = {
                    'l_loss': avg_loss,
                    'l_epoch': e,
                    'model_state': self.mod.state_dict(),
                    'opter_state': self.opter.state_dict()
                }
                T.save(check, check_path)

        return loss_l
    
# Week 7
class AlexDataset(Dataset):
    def __init__(self, transform, str='train'):
        print("Trying to fetch dataset from Kaggle....")
        path = kagglehub.dataset_download("birajsth/cats-and-dogs-filtered")
        print("Dataset found at: ", path)
        self.imgs_path = f'{path}/cats_and_dogs_filtered/{str}/'
        file_list = glob.glob(self.imgs_path + '*')
        self.data = []
        for cls_path in file_list:
            class_name = cls_path.split("/")[-1]
            for img_pth in glob.glob(cls_path + '/*.jpg'):
                self.data.append([img_pth, class_name])
        
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, class_name = self.data[index]
        img = Image.open(img_path).convert("RGB")
        label = self.class_map[class_name]

        if self.transform:
            img = self.transform(img)

        return img, label