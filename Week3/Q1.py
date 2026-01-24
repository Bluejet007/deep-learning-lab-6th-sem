import torch as T
import matplotlib.pyplot as plt

w = T.tensor(1.0)
b = T.tensor(1.0)
lr = T.tensor(0.001)
ep = 3

x = T.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
y = T.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])

loss_ep = []
for _ in range(3):
    loss = T.tensor(0.0)
    
    for inv, outv in zip(x, y):
        p = w * inv + b

        err = p - outv
        w -= lr * err * inv
        b -= lr * err

        loss += err ** 2

    loss = (loss / len(x)) ** 0.5
    loss_ep.append(loss.item())

print(f'w = {w}')
print(f'b = {b}')
plt.plot(loss_ep)
plt.title('MSE vs Epochs')
plt.show()