import torch as T
# T.set_default_device('cuda' if T.cuda.is_available() else 'cpu')
print(T.get_default_device())

# .empty(...)
# .zeros(...)
# .ones(...)
a = T.rand([2, 2]) # T.randn([2, 2])
print(a, a.dtype)

b = T.tensor([1, 2, 3, 4])
print(b, b.dtype)

c = T.tensor([[1, 2], [3, 4]])
print(c, c.dim())

d = T.eye(2)
print(d)

print(a * d)
print(a @ d)
print(a * d.T)

print(c.sum(1))