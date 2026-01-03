import torch as T

a = T.tensor([1, 2, 3, 4])

b = a.reshape((2, 2)) # Reshaping
print(b) # Viewing

c = T.stack((a, a)) # Stacking
print(c)

d = T.permute(c, (1, 0)) # Permuting
print(d)

print(d[2]) # Indexing

import numpy as np
arr = np.array([1, 2, 3])
e = T.tensor(arr) # To Tensor
print(e.numpy()) # To NumPy

r = T.rand((7, 7)) # Random tensor
print(r)

f = T.rand((1, 4))
print(a * f.T) # Matmul

# GPU acceleration
with T.device('cuda'):
    t1 = T.rand((2, 3))
    t2 = T.rand((2, 3))
    print(t1.max(), t1.argmax()) # Max, argmax
    print(t2.min(), t2.argmin()) # Min, argmin

print(r @ T.rand((7, 7))) # Better matml