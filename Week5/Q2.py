import torch as T
from torch import nn
import torch.nn.functional as F

T.manual_seed(7)
K = 3
I = 6

img = T.rand(I, I)
img = img.unsqueeze(0).unsqueeze(0)

conv = nn.Conv2d(1, 3, K, bias=False)
with T.no_grad():
    print(conv(img))


print('\nManual class:')
class MyConv(nn.Module):
    def __init__(self, outp, K):
        super().__init__()

        kerns = [T.rand(K, K) for _ in range(outp)]
        self.kerns = [k.unsqueeze(0).unsqueeze(0) for k in kerns]
    
    def forward(self, img):
        out = [F.conv2d(img, k).squeeze() for k in self.kerns]
        return T.stack(out).unsqueeze(0)

myConv = MyConv(3, K)
print(myConv(img))