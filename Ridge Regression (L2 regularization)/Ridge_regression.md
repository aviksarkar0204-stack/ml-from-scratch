# Ridge Regression (L2 Regularization) from Scratch

> Building Ridge Regression in raw C and NumPy — no sklearn, no magic. Just math and memory.

---

## What is Ridge Regression?

Ridge Regression is Linear Regression with a **regularization penalty** added to the loss function. It prevents overfitting by forcing weights to stay small.

Without regularization, a model can memorize training data perfectly — every noise, every outlier. It scores perfectly on training data but fails on new data. That's **overfitting**.

Ridge prevents this by adding a cost for large weights:

```
L = (1/n) · Σ (y - ŷ)² + λ · Σ w²
```

| Term | Meaning |
|------|---------|
| `(1/n) · Σ (y - ŷ)²` | MSE — how wrong the predictions are |
| `λ · Σ w²` | L2 penalty — cost for large weights |
| `λ` (lambda) | regularization strength — you choose this |

Higher λ → stronger penalty → weights pushed closer to zero → simpler model.

---

## Why Regularization?

Think of it like studying for an exam:

- **No regularization** → memorize every past question word for word. Fails on new questions.
- **Ridge** → forced to understand the underlying concept. Generalizes to new questions.

The penalty `λ · Σ w²` is the teacher saying *"I'll deduct marks if your answer is too complicated."*

---

## The Bias-Variance Tradeoff

| λ | Effect |
|---|--------|
| Too high | Weights pushed too far toward zero → underfitting (high bias) |
| Too low | Weights free to grow → overfitting (high variance) |
| Just right | Balance between fitting data and keeping weights small |

In practice, the right λ is found using **cross-validation**.

---

## How the Gradient Changes

Linear Regression gradient:
```
dL/dw = (-2/n) · Σ x · (y - ŷ)
```

Ridge adds the derivative of `λ · w²` → `2λ · w`:
```
dL/dw = (-2/n) · Σ x · (y - ŷ) + 2λ · w
```

Bias is not regularized — it stays the same:
```
dL/db = (-2/n) · Σ (y - ŷ)
```

That's the only change from Linear Regression — one extra term in `dw`.

---

## The Algorithm

```
1. ŷ  = w·x + b
2. dw = (-2/n) · Σ x · (y - ŷ) + 2λ · w   ← Ridge penalty added here
   db = (-2/n) · Σ (y - ŷ)
3. w  = w - lr · dw
   b  = b - lr · db
```

---

## Implementation in C

```c
#include <stdio.h>

#define N      10
#define LR     0.01
#define EPOCHS 5000
#define LAMBDA 0.1

int main()
{
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double y[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
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
        loss += LAMBDA * w * w;   // Ridge penalty

        if (loss < 0.000001) break;

        // print every 1000 epochs
        if (epoch % 1000 == 0)
            printf("epoch: %d | w: %f | b: %f | loss: %f\n", epoch, w, b, loss);

        // backward pass
        double dw = 0, db = 0;
        for (int i = 0; i < N; i++)
        {
            double pred = w * x[i] + b;
            dw += -2.0/N * x[i] * (y[i] - pred);
            db += -2.0/N * (y[i] - pred);
        }
        dw += 2 * LAMBDA * w;   // Ridge gradient term

        // update weights
        w -= LR * dw;
        b -= LR * db;
    }

    printf("\nFinal: w = %f, b = %f\n", w, b);
    printf("\nPredictions:\n");
    for (int i = 0; i < N; i++)
    {
        double pred = w * x[i] + b;
        printf("x: %.0f | pred: %.2f | actual: %.0f\n", x[i], pred, y[i]);
    }

    return 0;
}
```

**Compile and run:**

```bash
gcc ridge_regression.c -o ridge && ./ridge
```

**Output (LAMBDA=0.1):**

```
epoch: 0    | w: 1.430000 | b: 0.200000  | loss: 133.000000
epoch: 1000 | w: 1.973749 | b: -0.852221 | loss: 0.395266
epoch: 4000 | w: 1.976048 | b: -0.868263 | loss: 0.395210

Final: w = 1.976048, b = -0.868263

Predictions:
x: 1  | pred: 1.11  | actual: 1
x: 5  | pred: 9.01  | actual: 9
x: 10 | pred: 18.89 | actual: 19
```

> Note: True values are `w=2.0`, `b=-1.0`. Ridge intentionally biases weights toward zero — `w=1.976` not `2.0`. That's regularization working correctly.

---

## Implementation in NumPy

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], dtype=float)

w       = 0.0
b       = 0.0
lr      = 0.01
epochs  = 10000
lambda_ = 0.1      # 'lambda' is a reserved keyword in Python

for i in range(epochs):
    pred = w * x + b
    loss = np.mean((pred - y)**2) + lambda_ * w**2   # MSE + Ridge penalty

    dw = (-2/len(x)) * np.sum(x * (y - pred)) + 2 * lambda_ * w
    db = (-2/len(x)) * np.sum(y - pred)

    w -= lr * dw
    b -= lr * db

    if i % 1000 == 0:
        print(f"epoch: {i} | loss: {loss:.6f} | w: {w:.4f} | b: {b:.4f}")

print(f"\nFinal: w = {w:.4f}, b = {b:.4f}")
print("\nPredictions:")
for i in range(len(x)):
    pred_val = w * x[i] + b
    print(f"x: {x[i]:.0f} | pred: {pred_val:.2f} | actual: {y[i]:.0f}")
```

**Output:**

```
epoch: 0    | loss: 133.000000 | w: 1.4300 | b: 0.2000
epoch: 1000 | loss: 0.395266   | w: 1.9737 | b: -0.8522
epoch: 4000 | loss: 0.395210   | w: 1.9760 | b: -0.8683

Final: w = 1.9760, b = -0.8683
```

---

## Effect of Lambda

| λ | `w` converged | `b` converged | loss | Predictions |
|---|---|---|---|---|
| 0.1 | 1.9760 | -0.8683 | 0.3952 | slightly off |
| 0.01 | 1.9975 | -0.9867 | 0.0399 | nearly perfect |
| 0.001 | 1.9998 | -0.9987 | 0.0040 | almost exact |
| 0.0 | 2.0000 | -1.0000 | ~0.0 | exact (= Linear Regression) |

Higher λ → stronger regularization → weights further from true values → predictions slightly biased.

---

## Ridge vs Linear Regression

| | Linear Regression | Ridge Regression |
|---|---|---|
| Loss | `MSE` | `MSE + λ · Σ w²` |
| `dL/dw` | `(-2/n) · Σ x · (y - ŷ)` | `(-2/n) · Σ x · (y - ŷ) + 2λ·w` |
| `dL/db` | `(-2/n) · Σ (y - ŷ)` | same — bias not regularized |
| Weights | Can grow large | Shrunk toward zero |
| Overfitting | Prone | Resistant |
| Extra parameter | None | `λ` |

---

## Ridge vs Lasso

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | `λ · Σ w²` | `λ · Σ \|w\|` |
| Effect on weights | Shrinks toward zero | Can push to exactly zero |
| Feature selection | No — keeps all features | Yes — removes irrelevant features |
| Gradient | `2λ · w` | `λ · sign(w)` |

---

## Key Takeaways

- Ridge adds `λ · Σ w²` to MSE — penalizes large weights
- The only gradient change from Linear Regression is `+ 2λ · w` in `dw`
- Bias `b` is never regularized
- Higher λ → more regularization → weights closer to zero → simpler model
- Ridge shrinks weights but never to exactly zero — it keeps all features
- The right λ is found via cross-validation in practice
- Loss converges to a non-zero floor because the penalty term is always positive

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*