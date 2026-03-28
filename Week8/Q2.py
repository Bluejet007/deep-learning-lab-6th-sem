import glob
import os
import unicodedata
import string
import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

dev = T.device('cuda' if T.cuda.is_available() else 'cpu')
letters = string.ascii_letters + ' .,;\''
let_size = len(letters)

def unicToAsci(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in letters
    )

def load_data(path='./Week8/data/names'):
    cl_lines = {}
    clss = []
    for file in glob.glob(path + '/*.txt'):
        cl = os.path.splitext(os.path.basename(file))[0]
        clss.append(cl)
        lines = open(file, encoding='utf-8').read().strip().split('\n')
        cl_lines[cl] = [unicToAsci(line) for line in lines]
    return cl_lines, clss

def letterToIndex(letter):
    return letters.find(letter)

def lineToTensor(line):
    # Returns (seq_len, n_letters)
    tensor = T.zeros(len(line), let_size)
    for i, letter in enumerate(line):
        tensor[i][letterToIndex(letter)] = 1
    return tensor

class NamesDataset(Dataset):
    def __init__(self, cl_lines, clss):
        self.data = []
        for cat, names in cl_lines.items():
            for name in names:
                self.data.append((name, clss.index(cat)))
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Sort by length descending (required for pack_padded_sequence)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    names, labels = zip(*batch)
    
    tensors = [lineToTensor(n) for n in names]
    lens = T.tensor([len(t) for t in tensors])
    pad_seq = pad_sequence(tensors, True) # (Batch, Max_Len, Letters)
    labels = T.tensor(labels, dtype=T.long)
    
    return pad_seq, labels, lens

class NameRNN(nn.Module):
    def __init__(self, in_size, hide_size, out_size):
        super(NameRNN, self).__init__()
        self.rnn = nn.RNN(in_size, hide_size, batch_first=True)
        self.fc = nn.Linear(hide_size, out_size)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x, lengths):
        # Pack the sequence to ignore padding
        packed = pack_padded_sequence(x, lengths.to('cpu'), True)
        _, hidden = self.rnn(packed)
        # Use last hidden state for classification
        output = self.fc(hidden[-1])
        return self.softmax(output)

cl_lines, clss = load_data()
num_clss = len(clss)
dataset = NamesDataset(cl_lines, clss)
train_load = DataLoader(dataset, 128, True, collate_fn=collate_fn)

mod = NameRNN(let_size, 32, num_clss).to(dev)
loss_f = nn.NLLLoss()
opter = T.optim.Adam(mod.parameters(), 0.002)

for ep in range(10):
    total_loss = 0
    for names, labels, lengths in train_load:
        names, labels = names.to(dev), labels.to(dev)
        
        opter.zero_grad()
        out = mod(names, lengths)
        loss = loss_f(out, labels)
        loss.backward()
        opter.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {ep + 1} | Loss: {total_loss / len(train_load):.4f}')

def predict(name):
    mod.eval()
    with T.no_grad():
        name = unicToAsci(name)
        line_tensor = lineToTensor(name).unsqueeze(0).to(dev) # Add batch dim
        length = T.tensor([len(name)])
        
        output = mod(line_tensor, length)
        _, topi = output.topk(1)
        return clss[topi.item()]

print('Tests:')
for name in ['Amori', 'Schmidt', 'Dubois']:
    print(f'{name} -> {predict(name)}')