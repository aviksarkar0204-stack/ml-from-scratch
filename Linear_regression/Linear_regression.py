import numpy as np
x = np.array([1,2,3,4,5],dtype=float)
y = np.array([2,4,6,8,10],dtype=float)
w = 0
b = 0
lr = 0.01
epoch = 10000
for i in range(epoch):
    pred = w * x + b
    loss = np.mean((pred - y)**2)
    dw = (-2/len(x)) * np.sum(x * (y - pred))
    db = (-2/len(x)) * np.sum(y - pred)
    w = w - lr * dw
    b = b - lr * db

    if i % 100 == 0:
        print(f"epoch: {i}, w: {w:.4f}, b: {b:.4f}, loss: {loss:.6f}")

print(f"\nPrediction for x=7: {w*7+b:.4f} (expected 14)")