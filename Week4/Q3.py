import torch as T
T.set_default_device('cuda' if T.cuda.is_available() else 'cpu')

w1 = T.tensor([
    [-3.2116, -3.2110],
    [2.1765, 2.1765]])
b1 = T.tensor([1.4687, -3.1650])
w2 = T.tensor([[-4.1138, -4.1847]])
b2 = T.tensor([-3.5034])

X = T.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]]) 
Y = T.tensor([0.0, 1.0, 1.0, 0.0])

for x, y in zip(X, Y):
    a1 = T.tanh(w1 @ x + b1)
    p = T.sigmoid(w2 @ a1 + b2)
    print(f'p = {p.item()}, y = {y}')