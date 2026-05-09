# Logistic Regression from Scratch

> Building Logistic Regression in raw C and NumPy — no sklearn, no magic. Just math and memory.

---

## What is Logistic Regression?

Logistic Regression is a **binary classification** algorithm. Unlike Linear Regression which outputs any number, Logistic Regression outputs a probability between 0 and 1.

Example: given tumor size, predict malignant (1) or benign (0).

The core equation:

```
z  = w·x + b
ŷ  = σ(z) = 1 / (1 + e^(-z))
```

| Symbol | Meaning |
|--------|---------|
| `w` | weight (slope) |
| `b` | bias (intercept) |
| `z` | linear combination |
| `ŷ` | predicted probability (0 to 1) |

---

## Why Not MSE?

MSE works for regression but breaks for classification — two reasons:

**1. MSE ignores confidence.** It only measures distance, not how sure the model is. A prediction of `0.51` for a true `1` gets penalized heavily even though the class is correct.

**2. MSE + sigmoid = vanishing gradients.** The sigmoid derivative is nearly zero at the extremes. So when the model is confidently wrong (predicting `0.001` for a true `1`), MSE gives a tiny gradient — the model barely updates even though it's completely wrong.

---

## Loss Function — Binary Cross-Entropy (BCE)

```
L = -(1/n) · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
```

BCE thinks in terms of probability:
- Predict `0.999` for true `1` → loss near zero ✓
- Predict `0.001` for true `1` → loss explodes ✗

This heavily penalizes confident wrong predictions — exactly what we want.

---

## Deriving the Gradients

Starting from:

```
z  = w·x + b
ŷ  = σ(z)
L  = -(1/n) · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
```

Apply chain rule: `dL/dw = dL/dŷ · dŷ/dz · dz/dw`

**Step 1 — dL/dŷ:**
```
dL/dŷ = -(1/n) · Σ [ y/ŷ - (1-y)/(1-ŷ) ]
```

**Step 2 — dŷ/dz (sigmoid derivative):**
```
dŷ/dz = σ(z) · (1 - σ(z)) = ŷ · (1 - ŷ)
```

**Step 3 — dz/dw:**
```
dz/dw = x
```

**Multiply all three — the algebra simplifies beautifully:**
```
[ y/ŷ - (1-y)/(1-ŷ) ] · ŷ·(1-ŷ)
= y·(1-ŷ) - (1-y)·ŷ
= y - ŷ
```

**Final gradients:**
```
dL/dw = (1/n) · Σ (ŷ - y) · x
dL/db = (1/n) · Σ (ŷ - y)
```

The sigmoid derivative cancelled out completely — that's why BCE and sigmoid are designed for each other.

---

## Compare to Linear Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| Output | Any number | Probability (0 to 1) |
| Activation | None | Sigmoid |
| Loss | MSE | Binary Cross-Entropy |
| `dL/dw` | `(-2/n) · Σ x · (y - ŷ)` | `(1/n) · Σ (ŷ - y) · x` |
| `dL/db` | `(-2/n) · Σ (y - ŷ)` | `(1/n) · Σ (ŷ - y)` |

Almost identical — just the sign and constant changed.

---

## The Algorithm

```
1. z    = w·x + b
2. ŷ    = σ(z) = 1 / (1 + e^(-z))
3. dw   = (1/n) · Σ (ŷ - y) · x
4. db   = (1/n) · Σ (ŷ - y)
5. w    = w - lr · dw
6. b    = b - lr · db
```

---

## Implementation in C

```c
#include <stdio.h>
#include <math.h>

#define N      10
#define LR     0.01
#define EPOCHS 5000

int main()
{
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double w = 0.0;
    double b = 0.0;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        // forward pass
        double loss = 0;
        for (int i = 0; i < N; i++)
        {
            double z    = w * x[i] + b;
            double pred = 1.0 / (1.0 + exp(-z));
            loss += -(y[i] * log(pred) + (1 - y[i]) * log(1 - pred));
        }
        loss /= N;

        // print every 500 epochs
        if (epoch % 500 == 0)
            printf("epoch: %d, w: %f, b: %f, loss: %f\n", epoch, w, b, loss);

        // early stopping
        if (loss < 0.0001) break;

        // backward pass
        double dw = 0, db = 0;
        for (int i = 0; i < N; i++)
        {
            double z    = w * x[i] + b;
            double pred = 1.0 / (1.0 + exp(-z));
            dw += (pred - y[i]) * x[i];
            db += (pred - y[i]);
        }
        dw /= N;
        db /= N;

        // update weights
        w = w - LR * dw;
        b = b - LR * db;
    }

    printf("\nFinal: w = %f, b = %f\n", w, b);
    printf("\nPredictions:\n");
    for (int i = 0; i < N; i++)
    {
        double z    = w * x[i] + b;
        double pred = 1.0 / (1.0 + exp(-z));
        int cls     = pred >= 0.5 ? 1 : 0;
        printf("x: %.0f | prob: %.4f | class: %d | expected: %.0f\n",
               x[i], pred, cls, y[i]);
    }

    return 0;
}
```

**Compile and run:**

```bash
gcc logistic_regression.c -o logr -lm && ./logr
```

**Output:**

```
epoch: 0,    w: 0.000000, b: 0.000000, loss: 0.693147
epoch: 500,  w: 0.259299, b: -0.772911, loss: 0.482145
epoch: 1000, w: 0.359329, b: -1.402459, loss: 0.400572
...
Final: w = 0.800938, b = -4.052532

Predictions:
x: 1  | prob: 0.0373 | class: 0 | expected: 0
x: 2  | prob: 0.0794 | class: 0 | expected: 0
x: 3  | prob: 0.1611 | class: 0 | expected: 0
x: 4  | prob: 0.2997 | class: 0 | expected: 0
x: 5  | prob: 0.4880 | class: 0 | expected: 0
x: 6  | prob: 0.6799 | class: 1 | expected: 1
x: 7  | prob: 0.8255 | class: 1 | expected: 1
x: 8  | prob: 0.9133 | class: 1 | expected: 1
x: 9  | prob: 0.9591 | class: 1 | expected: 1
x: 10 | prob: 0.9812 | class: 1 | expected: 1
```

Decision boundary learned at `x ≈ 5.5` — exactly correct.

---

## Implementation in NumPy

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  dtype=float)

w = 0.0
b = 0.0
lr = 0.01
epochs = 10000

for i in range(epochs):
    z    = w * x + b
    pred = 1.0 / (1.0 + np.exp(-z))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    if i % 500 == 0:
        print(f"epoch: {i}, w: {w:.4f}, b: {b:.4f}, loss: {loss:.6f}")

    dw = (1 / len(x)) * np.sum((pred - y) * x)
    db = (1 / len(x)) * np.sum(pred - y)

    w = w - lr * dw
    b = b - lr * db

print(f"\nFinal: w = {w:.4f}, b = {b:.4f}")
print("\nPredictions:")
for i in range(len(x)):
    z = w * x[i] + b
    p = 1.0 / (1.0 + np.exp(-z))
    print(f"x: {x[i]:.0f} | prob: {p:.4f} | class: {int(p >= 0.5)} | expected: {y[i]:.0f}")
```

**Output:**

```
epoch: 0,    w: 0.0000, b: 0.0000, loss: 0.693147
epoch: 500,  w: 0.3275, b: -0.9054, loss: 0.460620
...
Final: w = 1.0751, b = -5.6367

Predictions:
x: 1  | prob: 0.0101 | class: 0 | expected: 0
x: 5  | prob: 0.4343 | class: 0 | expected: 0
x: 6  | prob: 0.6932 | class: 1 | expected: 1
x: 10 | prob: 0.9941 | class: 1 | expected: 1
```

NumPy version is sharper — 10000 epochs gave `w` more time to grow, pushing probabilities closer to 0 and 1.

---

## C vs NumPy — What Changes

| | C | NumPy |
|---|---|---|
| Sigmoid | `1.0 / (1.0 + exp(-z))` per element | `1.0 / (1.0 + np.exp(-z))` on whole array |
| Loss | Manual loop + accumulate + `/N` | `np.mean(...)` in one line |
| Gradient | Manual loop + accumulate + `/N` | `np.sum(...) / len(x)` |
| Lines of code | ~50 | ~18 |

---

## Key Takeaways

- Logistic Regression adds sigmoid on top of Linear Regression to squash output to (0, 1)
- MSE fails for classification — use Binary Cross-Entropy instead
- The BCE gradient simplifies to just `(ŷ - y)` — sigmoid derivative cancels out
- Decision boundary is where `ŷ = 0.5`, i.e. where `w·x + b = 0`
- BCE loss never reaches zero — it converges gradually unlike MSE

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*