import torch as T
from torchviz import make_dot

a = T.tensor(1.0, requires_grad=True)
b = T.tensor(1.0, requires_grad=True)

x = 2 * a + 3 * b
y = 5 * a ** 2 + 3 * b ** 3
z = 2 * x + 3 * y

z.backward()
print(f'dz/da = {a.grad}')
print(f'dz/db = {b.grad}')

dot = make_dot(z, {'a': a, 'b': b, 'z': z})
dot.format = 'png'
dot.render('Week2/Q1')

print('\nManual:')
print(f'dz/da = {2 * 2 + 3 * 10 * a}')
print(f'dz/db = {2 * 3 + 3 * 9 * b ** 2}')