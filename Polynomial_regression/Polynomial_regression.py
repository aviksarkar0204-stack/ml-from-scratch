import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],dtype=float)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100],dtype=float)
w =np.array([0.0,0.0])
b = 0
lr = 0.1
epochs = 50000
x_norm = x / np.max(x)
for epoch in range(epochs):
    pred = w[0]*x_norm + w[1]*x_norm**2 + b
    loss = np.mean((y - pred)**2)
    dw0 = (-2 / len(x)) * np.sum(x_norm * (y - pred))
    dw1 = (-2 / len(x)) * np.sum(x_norm ** 2 * (y - pred))
    db = (-2 / len(x)) * np.sum(y - pred)

    w[0] -= lr * dw0
    w[1] -= lr * dw1
    b -= lr * db

    if epoch % 1000 == 0:
        print(f"epoch: {epoch} | loss: {loss:.6f} | w0: {w[0]:.4f} | w1: {w[1]:.4f} | b: {b:.4f}")

print("\nPredictions:")
for i in range(len(x)):
    pred_val = w[0]*x_norm[i] + w[1]*x_norm[i]**2 + b
    print(f"x: {x[i]:.0f} | pred: {pred_val:.2f} | actual: {y[i]:.0f}")
