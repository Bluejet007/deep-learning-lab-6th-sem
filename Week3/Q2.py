import torch as T
import matplotlib.pyplot as plt

w = T.tensor(1.0, requires_grad=True)
b = T.tensor(1.0, requires_grad=True)
lr = T.tensor(0.1)
ep = 3

x = T.tensor([2.0, 4.0])
y = T.tensor([20.0, 40.0]) 

loss_ep = []
for _ in range(3):
    loss = T.tensor(0.0)
    
    for inv, outv in zip(x, y):
        print(inv, outv)
        p = w * inv + b

        p.backward()
        with T.no_grad():
            err = p - outv
            w -= lr * w.grad * err
            b -= lr * b.grad * err

            loss += err.item() ** 2

        w.grad.zero_()
        b.grad.zero_()

    loss = (loss / len(x)) ** 0.5
    loss_ep.append(loss)

print(f'w = {w}')
print(f'b = {b}')
plt.plot(loss_ep)
plt.title('MSE vs Epochs')
plt.show()