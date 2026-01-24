import torch as T
import matplotlib.pyplot as plt

x = T.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = T.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
lr = T.tensor(0.001)

class RegressionModel():
    def __init__(self):
        self.w = T.rand([1], requires_grad=True)
        self.b = T.rand([1], requires_grad=True)

    def forward(self, x):
        return self.w * x + self.b
    
    def update(self):
        self.w -= lr * self.w.grad
        self.b -= lr * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

    def criterion(self, y, p):
        return (y - p) ** 2
    
mod = RegressionModel()
loss_ep = []
for _ in range(100):
    loss = 0.0

    for inv, outv in zip(x, y):
        p = mod.forward(inv)
        loss += mod.criterion(outv, p)

    loss /= len(x)
    loss_ep.append(loss.item())
    loss.backward()

    with T.no_grad():
        mod.update()
    mod.reset_grad()

print(f'w = {mod.w.item()}')
print(f'b = {mod.b.item()}')
plt.plot(loss_ep)
plt.title('MSE vs Epochs')
plt.show()