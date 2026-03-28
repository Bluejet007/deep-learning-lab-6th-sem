import glob
import torch as T
from torch.utils.data import DataLoader, random_split
from torch import nn
import torchvision.transforms as transf
from unicodedata import normalize
from string import ascii_lowercase
import MyDL

mod = nn.RNN(1, 18, 1)
h0 = T.randn(2, 3, 13)

paths = glob.glob('Week8/data/names/*.txt')[:18]

X, Y, langs = [], [], []
for i, p in enumerate(paths):
    file = open(p)

    x = [0] * 8 ** 2
    lines = []
    for name in file.read().splitlines():
        x1 = x
        for ch in normalize('NFD', name.lower()):
            if ch == ' ':
                x1[26] = 1
            elif ch == '-':
                x1[27] = 1
            elif ch == '\'':
                x1[28] = 1
            elif ch == ',':
                x1[29] = 1
            elif ch in ascii_lowercase:
                x1[ord(ch) - ord('a')] = 1
        lines.append(x1)

    X.extend(lines)

    y = [0] * 18
    y[i] = 1
    Y.extend([y] * len(lines))
    langs.append(p.split('/')[-1].split('.')[0])
langs = tuple(langs)

dataset = MyDL.MyDataset(T.tensor(X, dtype=T.float), T.tensor(Y, dtype=T.float))

