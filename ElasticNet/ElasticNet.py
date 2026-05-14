import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
y = np.array([1,3,5,7,9,11,13,15,17,19],dtype=float)
w = 0.0
b = 0.0
lr = 0.01
epoch = 10000
lambda_ = 0.01
for i in range(epoch):
    pred = w * x + b
    loss = np.mean((pred - y)**2) + lambda_ * np.abs(w) + lambda_ * w**2

    dw = (-2 / len(x)) * np.sum(x * (y - pred)) + 2 * lambda_ * w + lambda_ * np.sign(w)
    db = (-2 / len(x)) * np.sum(y - pred)

    w -= lr * dw
    b -= lr * db

    if i % 1000 == 0:
        print(f"epoch: {i} | loss: {loss:.6f} | w: {w:.4f} | b: {b:.4f}")

for i in range(len(x)):
    print(f"x: {x[i]:.0f} | pred: {w*x[i]+b:.2f} | actual: {y[i]:.0f}")