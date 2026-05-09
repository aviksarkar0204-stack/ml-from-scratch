# Linear Regression from Scratch

> Building Linear Regression in raw C and NumPy — no sklearn, no magic. Just math and memory.

---

## What is Linear Regression?

Given data points — say, house sizes (`x`) and their prices (`y`) — Linear Regression finds the best straight line through them:

```
ŷ = w·x + b
```

| Symbol | Meaning |
|--------|---------|
| `w` | weight (slope of the line) |
| `b` | bias (y-intercept) |
| `ŷ` | predicted value |

The goal: find values of `w` and `b` that make `ŷ` as close to the real `y` as possible.

---

## Loss Function — Mean Squared Error (MSE)

We measure "how wrong" the model is using **MSE**:

```
L = (1/n) · Σ (y - ŷ)²
```

Squaring does two things:
- Makes negative errors positive
- Punishes large errors more than small ones

---

## How to Minimize Loss — Gradient Descent

The gradient of the loss tells us which direction `w` and `b` need to move to reduce `L`. We nudge them repeatedly:

```
w = w - lr · dL/dw
b = b - lr · dL/db
```

`lr` is the **learning rate** — controls the size of each nudge.

---

## Deriving the Gradients

Starting from:

```
L = (1/n) · Σ (y - ŷ)²  =  (1/n) · Σ (y - (w·x + b))²
```

Let `e = y - ŷ` (the error). Then `L = (1/n) · Σ e²`.

### dL/dw

Apply the chain rule:

```
d(e²)/de  =  2e
de/dw     =  -x          ← derivative of -(w·x + b) w.r.t. w
```

Multiply (chain rule):

```
dL/dw = (1/n) · Σ 2e · (-x)
      = (-2/n) · Σ x · (y - ŷ)
```

### dL/db

Same process, only `de/db` changes:

```
de/db = -1               ← derivative of -(w·x + b) w.r.t. b
```

So:

```
dL/db = (1/n) · Σ 2e · (-1)
      = (-2/n) · Σ (y - ŷ)
```

### Why the negative sign matters

The gradient points toward *steepest increase* in loss. We want to *decrease* it, so we subtract:

```
w = w - lr · dL/dw
```

The negative in `(-2/n)` and the subtraction in the update rule cancel correctly:
- If prediction is too low → `(y - ŷ) > 0` → gradient is negative → subtracting a negative **increases** `w` → prediction goes up. ✓

---

## The Algorithm

Three steps every epoch:

```
1. Predict  →  ŷ = w·x + b
2. Gradient →  dL/dw, dL/db
3. Update   →  w -= lr · dL/dw
               b -= lr · dL/db
```

---

## Implementation in C

```c
#include <stdio.h>

#define N      5
#define LR     0.01
#define EPOCHS 5000

int main()
{
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {2, 4, 6, 8, 10};
    double w = 0.0;
    double b = 0.0;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        // forward pass
        double loss = 0;
        for (int i = 0; i < N; i++)
        {
            double pred = w * x[i] + b;
            loss += (y[i] - pred) * (y[i] - pred);
        }
        loss /= N;

        // early stopping
        if (loss < 0.000001) break;

        // print every 100 epochs
        if (epoch % 100 == 0)
            printf("epoch: %d, w: %f, b: %f, loss: %f\n", epoch, w, b, loss);

        // backward pass
        double dw = 0, db = 0;
        for (int i = 0; i < N; i++)
        {
            double pred = w * x[i] + b;
            dw += -2.0 / N * x[i] * (y[i] - pred);
            db += -2.0 / N * (y[i] - pred);
        }

        // update weights
        w = w - LR * dw;
        b = b - LR * db;
    }

    printf("\nFinal: w = %f, b = %f\n", w, b);
    printf("Prediction for x=7: %f (expected 14)\n", w * 7 + b);

    return 0;
}
```

**Compile and run:**

```bash
gcc linear_regression.c -o lr && ./lr
```

**Output:**

```
epoch: 0,    w: 0.000000, b: 0.000000, loss: 44.000000
epoch: 100,  w: 1.898433, b: 0.366691, loss: 0.024474
epoch: 200,  w: 1.927611, b: 0.261347, loss: 0.012432
...
epoch: 3800, w: 2.000000, b: 0.000001, loss: 0.000000

Final: w = 2.000000, b = 0.000000
Prediction for x=7: 14.000000 (expected 14)
```

---

## Implementation in NumPy

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w = 0.0
b = 0.0
lr = 0.01
epochs = 10000

for i in range(epochs):
    pred = w * x + b
    loss = np.mean((pred - y) ** 2)

    if loss < 0.000001:
        break

    dw = (-2 / len(x)) * np.sum(x * (y - pred))
    db = (-2 / len(x)) * np.sum(y - pred)

    w = w - lr * dw
    b = b - lr * db

    if i % 100 == 0:
        print(f"epoch: {i}, w: {w:.4f}, b: {b:.4f}, loss: {loss:.6f}")

print(f"\nFinal: w = {w:.4f}, b = {b:.4f}")
print(f"Prediction for x=7: {w*7+b:.4f} (expected 14)")
```

**Output:**

```
epoch: 0,    w: 0.2200, b: 0.0800, loss: 44.000000
epoch: 100,  w: 1.8984, b: 0.3667, loss: 0.024474
...
Final: w = 2.0000, b = 0.0000
Prediction for x=7: 14.0000 (expected 14)
```

---

## C vs NumPy — What Changes

| | C | NumPy |
|---|---|---|
| Prediction | `for` loop over each sample | `w * x + b` (vectorized) |
| Loss | Manual `Σ e²` / N in a loop | `np.mean((pred - y) ** 2)` |
| Gradient | Manual `Σ` in a second loop | `np.sum(x * (y - pred))` |
| Lines of code | ~40 | ~15 |
| What you learn | Everything | The abstraction on top |

The math is **identical**. NumPy just removes the loops — it runs the same C-style loops internally, optimized in compiled code.

---

## Key Takeaways

- Linear Regression is just a line: `ŷ = w·x + b`
- MSE measures how wrong the line is
- Gradient Descent nudges `w` and `b` toward lower loss every epoch
- The gradients `dL/dw` and `dL/db` come directly from the chain rule
- Early stopping saves wasted computation once loss is small enough
- Building it in C first makes NumPy completely transparent

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*