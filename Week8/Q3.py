import torch
import torch.nn as nn
import numpy as np

text = "next character prediction with rnn"
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

seq_len = 10
input_size = len(chars)
hidden_size = 32
output_size = len(chars)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :]) 
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    input_seq = text[0:seq_len]
    target_char = text[seq_len]

    x = torch.zeros(1, seq_len, input_size)
    for i, ch in enumerate(input_seq):
        x[0, i, char_to_int[ch]] = 1
    
    target = torch.tensor([char_to_int[target_char]])

    hidden = model.init_hidden()
    output, hidden = model(x, hidden)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')