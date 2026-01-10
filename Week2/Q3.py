import torch as T

b = T.tensor(3.0)
x = T.tensor(2.0)
w = T.tensor(1.0, requires_grad=True)

u = w * x
v = u + b
a = T.sigmoid(v)
a.backward()
print(f'da/dw = {w.grad}')

print('\nManual')
print(f'da/dw = {a * (1- a) * x}')