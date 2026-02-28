import torch as T
from torch.utils.data import Dataset, DataLoader
# from torchmetrics.classification import MulticlassConfusionMatrix
import kagglehub
import glob
import torch.nn as nn
from torchvision import transforms
from torchvision.models import AlexNet_Weights, alexnet
from PIL import Image
import MyDL

f_name = '.'.join(__file__.split('.')[:-1])
mod_path = f'{f_name}.pth'

dev = T.device("cuda" if T.cuda.is_available() else "cpu")
b_size = 64
epochs = 3
lr = 0.02
trans = transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])])

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


train_set = AlexDataset(trans)
train_load = DataLoader(train_set, b_size, True)
test_set = AlexDataset(trans, 'validation')
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
    trainer.fit(train_load, epochs)
    T.save(mod, mod_path)

cm = MyDL.conf_mat(mod, test_load, 2)
total = cm.sum().item()
corr = cm.diag().sum().item()
acc = 100 * corr / total

print(f'Accuracy: {acc}%')