import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
y = np.array([0,0,0,0,0,1,1,1,1,1], dtype=float)
w = 0
b = 0
lr = 0.01
epoch = 10000
for i in range(epoch):
    z = w*x + b
    pred = 1.0/(1.0+np.exp(-z))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    dw = (1 / len(x)) * np.sum((pred - y) * x)
    db = (1 / len(x)) * np.sum(pred - y)
    w = w - lr * dw
    b = b - lr * db
    if i % 100 == 0:
        print(f"epoch: {i}, w: {w:.4f}, b: {b:.4f}, loss: {loss:.6f}")
print("\nPredictions:")
for i in range(len(x)):
    z = w * x[i] + b
    p = 1.0 / (1.0 + np.exp(-z))
    print(f"x: {x[i]:.0f} | prob: {p:.4f} | class: {int(p >= 0.5)} | expected: {y[i]:.0f}")