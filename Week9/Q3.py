import torch as T
import torch.nn as nn

dev = T.device('cuda' if T.cuda.is_available() else 'cpu')

train_text = ''
with open('./Week9/Q3_data.txt') as data:
    train_text = data.readlines()
    train_text = [l for l in train_text if l != '\n']
    train_text = ''.join(train_text)

test_text = '''Miro returned to the dock. Rafi returned to the trails.
The baker opened his shop.
The children laughed again.
The wind whispered once more.'''

chars = sorted(tuple(set(train_text) | set(test_text)))
char_int = {ch: i for i, ch in enumerate(chars)}
int_char = {i: ch for i, ch in enumerate(chars)}

seq_len = 20
i_size = len(chars)
h_size = 32
o_size = len(chars)

class SimpleLSTM(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = h_size
        self.lstm = nn.LSTM(i_size, h_size, batch_first=True)
        self.fc = nn.Linear(h_size, o_size)

    def forward(self, x, states):
        out, states = self.lstm(x, states)
        out = self.fc(out[:, -1, :]) 
        return out, states

    def init_hidden(self):
        return T.zeros(1, 1, self.hidden_size, device=dev), T.zeros(1, 1, self.hidden_size, device=dev)
    
mod = None
try:
    mod = T.load('Q3.pth', weights_only=False)
except:
    mod = SimpleLSTM(i_size, h_size, o_size).to(dev)
    loss_f = nn.CrossEntropyLoss().to(dev)
    opter = T.optim.Adam(mod.parameters(), lr=0.001, weight_decay=0.0001)

    for ep in range(20):
        for i in range(len(train_text) - seq_len):
            i_seq = train_text[i:i + seq_len]
            target = train_text[i + seq_len]

            x = T.zeros(1, seq_len, i_size, device=dev)
            for i, ch in enumerate(i_seq):
                x[0, i, char_int[ch]] = 1

            target = T.tensor([char_int[target]], device=dev)

            states = mod.init_hidden()
            out, _ = mod(x, states)
            loss = loss_f(out, target)

            opter.zero_grad()
            loss.backward()
            opter.step()

        if (ep + 1) % 2 == 0:
            print(f'Epoch [{ep + 1} / 20] loss: {loss.item()}')

    T.save(mod, 'Q3.pth')

pred = test_text[:seq_len]
states = mod.init_hidden()
for i in range(len(test_text) - seq_len):
    mod.eval()
    i_seq = pred[-seq_len:]
    target = test_text[i + seq_len]

    x = T.zeros(1, seq_len, i_size, device=dev)
    for i, ch in enumerate(i_seq):
        x[0, i, char_int[ch]] = 1

    target = T.tensor([char_int[target]], device=dev)

    out, states = mod(x, states)
    out = out.squeeze().argmax()
    pred += int_char[out.item()]

print(f'Prediction:\n{pred}')
print(f'Original:\n{test_text}')