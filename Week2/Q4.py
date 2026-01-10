import torch as T

x = T.tensor(1.0, requires_grad=True)

p = x ** 2 + 2 * x + T.sin(x)
f = T.exp(-p)

print('f =', f)
f.backward()
print('df/dx =', x.grad)

# Manual
df = -T.exp(-p) * (2 * x + 2 + T.cos(x))
print('Manual:', df)