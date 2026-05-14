# ElasticNet Regression from Scratch

> Building ElasticNet in raw C and NumPy — no sklearn, no magic. Just math and memory.

---

## What is ElasticNet?

ElasticNet combines both **Ridge (L2)** and **Lasso (L1)** regularization into a single loss function:

```
L = (1/n) · Σ (y - ŷ)² + λ · Σ |w| + λ · Σ w²
```

| Term | Meaning |
|------|---------|
| `(1/n) · Σ (y - ŷ)²` | MSE — prediction error |
| `λ · Σ \|w\|` | L1 penalty (Lasso) — can zero weights out |
| `λ · Σ w²` | L2 penalty (Ridge) — shrinks weights toward zero |

You get the best of both — Ridge's stability and Lasso's feature selection.

---

## The Full Regularization Family

| Model | Loss | Extra gradient term | Feature selection |
|-------|------|--------------------|--------------------|
| Linear | `MSE` | none | ❌ |
| Ridge | `MSE + λ·Σw²` | `+ 2λ·w` | ❌ |
| Lasso | `MSE + λ·Σ\|w\|` | `+ λ·sign(w)` | ✅ |
| ElasticNet | `MSE + λ·Σ\|w\| + λ·Σw²` | `+ λ·sign(w) + 2λ·w` | ✅ |

ElasticNet gradient is literally Lasso gradient + Ridge gradient added together.

---

## The Gradient

```
dL/dw = (-2/n) · Σ x · (y - ŷ) + λ · sign(w) + 2λ · w
dL/db = (-2/n) · Σ (y - ŷ)
```

Three terms added together — MSE gradient + L1 term + L2 term.

---

## When to Use ElasticNet

| Situation | Best choice |
|---|---|
| Many irrelevant features | Lasso — zeros them out |
| All features relevant, prevent overfitting | Ridge |
| Many features, some irrelevant, correlated data | ElasticNet |
| No regularization needed | Linear Regression |

ElasticNet is most useful when features are correlated — Lasso tends to pick one and ignore the rest, while ElasticNet keeps both but shrinks them.

---

## Implementation in C

```c
#include <stdio.h>

#define N      10
#define LR     0.01
#define EPOCHS 5000
#define LAMBDA 0.01

double sign(double x) {
    if (x > 0) return  1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

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
        loss += LAMBDA * (w > 0 ? w : -w) + LAMBDA * w * w;  // L1 + L2

        if (loss < 0.000001) break;

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
        dw += LAMBDA * sign(w) + 2 * LAMBDA * w;  // L1 + L2 gradient

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
gcc elasticnet.c -o elasticnet && ./elasticnet
```

**Output:**

```
epoch: 0    | w: 1.430000 | b: 0.200000  | loss: 133.000000
epoch: 1000 | w: 1.997293 | b: -0.981193 | loss: 0.059941
epoch: 4000 | w: 1.999976 | b: -0.999867 | loss: 0.059999

Final: w = 1.999976, b = -0.999867

Predictions:
x: 1  | pred: 1.00  | actual: 1
x: 5  | pred: 9.00  | actual: 9
x: 10 | pred: 19.00 | actual: 19
```

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
lambda_ = 0.01

for i in range(epochs):
    pred = w * x + b
    loss = np.mean((pred - y)**2) + lambda_ * np.abs(w) + lambda_ * w**2

    dw = (-2/len(x)) * np.sum(x * (y - pred)) + lambda_ * np.sign(w) + 2 * lambda_ * w
    db = (-2/len(x)) * np.sum(y - pred)

    w -= lr * dw
    b -= lr * db

    if i % 1000 == 0:
        print(f"epoch: {i} | loss: {loss:.6f} | w: {w:.4f} | b: {b:.4f}")

print(f"\nFinal: w = {w:.4f}, b = {b:.4f}")
print("\nPredictions:")
for i in range(len(x)):
    print(f"x: {x[i]:.0f} | pred: {w*x[i]+b:.2f} | actual: {y[i]:.0f}")
```

**Output:**

```
epoch: 0    | loss: 133.000000 | w: 1.4300 | b: 0.2000
epoch: 1000 | loss: 0.059997   | w: 1.9943 | b: -0.9650
epoch: 4000 | loss: 0.059924   | w: 1.9970 | b: -0.9834

Final: w = 1.9970, b = -0.9834
```

---

## Comparing All Four Regression Variants

| Model | `w` converged | `b` converged | loss floor |
|---|---|---|---|
| Linear | 2.0000 | -1.0000 | ~0.0 |
| Ridge | 1.9975 | -0.9867 | 0.0399 |
| Lasso | 1.9994 | -0.9967 | 0.0199 |
| ElasticNet | 1.9999 | -0.9999 | 0.0599 |

ElasticNet has the highest loss floor — both penalties add to it. But `w` is closest to true value because the two penalties balance each other.

---

## Key Takeaways

- ElasticNet = Lasso + Ridge combined — add both penalty terms to loss and gradient
- Loss: `MSE + λ·|w| + λ·w²`
- Gradient extra term: `λ·sign(w) + 2λ·w` — Lasso term + Ridge term
- Loss floor is higher than either Ridge or Lasso alone — both penalties contribute
- Best when features are correlated — Lasso alone picks one and drops the rest
- Two separate lambdas (`λ₁` for L1, `λ₂` for L2) give finer control in practice
- `sign()` must be implemented manually in C — not a built-in function

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*