import torch as T
import torch.nn.functional as F

T.manual_seed(7)
st = 1
pad = 0
K = 3
I = 6

img = T.rand(I, I)
print('Img:')
print(img)
img = img.unsqueeze(0).unsqueeze(0)
print(img.shape)

kern = T.ones(K, K)
print('Kernel:')
print(kern)
kern = kern.unsqueeze(0).unsqueeze(0)

final = F.conv2d(img, kern, stride=st, padding=pad).squeeze()
print('Output:')
print(final)
print(final.shape)
print(f'Calculated size: {(I - K + 1 + 2 * pad) // st}')