# Gaussian Naive Bayes from Scratch

> Building Gaussian Naive Bayes in raw C and NumPy — no sklearn, no magic. Just probability and statistics.

---

## What is Naive Bayes?

Naive Bayes is a family of classification algorithms based on **Bayes' Theorem**:

```
P(class | features) ∝ P(features | class) · P(class)
```

| Term | Name | Meaning |
|------|------|---------|
| `P(class)` | Prior | How common is this class before seeing any data? |
| `P(features | class)` | Likelihood | How likely are these features given this class? |
| `P(class | features)` | Posterior | What we actually want to predict |

The **"Naive"** assumption — all features are independent of each other. Almost never true in reality, but works surprisingly well in practice.

---

## Why Gaussian?

Different Naive Bayes variants differ only in how they model the likelihood `P(features | class)`:

| Variant | Assumption |
|---------|------------|
| **Gaussian** NB | Features follow a normal distribution |
| Multinomial NB | Features are word counts |
| Bernoulli NB | Features are binary (0 or 1) |

Gaussian NB assumes each feature per class follows a **Gaussian (normal) distribution** — described by mean `μ` and variance `σ²`.

---

## The Algorithm — 3 Steps

### Step 1 — Prior `P(class)`

How common is each class in training data?

```
P(class=0) = count of class 0 / total samples
P(class=1) = count of class 1 / total samples
```

### Step 2 — Likelihood `P(x | class)`

For each feature and each class, compute mean and variance:

```
μ(class, feature)  = mean of feature values for that class
σ²(class, feature) = variance of feature values for that class
```

Then use the Gaussian PDF:

```
P(x | class) = (1 / √(2π σ²)) · exp(-(x - μ)² / (2σ²))
```

### Step 3 — Posterior + Prediction

Multiply prior by likelihood for each class, pick the highest:

```
score(class) = log P(class) + Σ log P(xᵢ | class)
```

We use **log** to avoid underflow — multiplying many small probabilities creates numbers too tiny for `double` to store. Log turns multiplication into addition.

---

## Why Two Passes for Variance?

Variance requires the mean first:

```
σ² = (1/n) · Σ (x - μ)²
```

You can't compute `(x - μ)²` until `μ` is known. So:
- Pass 1 → compute mean
- Pass 2 → compute variance using that mean

---

## Implementation in C

```c
#include <stdio.h>
#include <math.h>

#define N 10   // samples
#define F 2    // features
#define C 2    // classes

int main()
{
    double x[N][F] = {
        {150, 48}, {155, 52}, {160, 55}, {165, 60}, {170, 65},
        {175, 70}, {180, 75}, {185, 82}, {190, 88}, {195, 95}
    };
    int y[N] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double test[F] = {160, 55};

    double mean[C][F], var[C][F], prior[C];

    // initialize
    for (int c = 0; c < C; c++) {
        prior[c] = 0;
        for (int f = 0; f < F; f++) {
            mean[c][f] = 0;
            var[c][f]  = 0;
        }
    }

    // pass 1 — count and accumulate for mean
    int count[C] = {0, 0};
    for (int i = 0; i < N; i++) {
        int c = y[i];
        count[c]++;
        for (int f = 0; f < F; f++)
            mean[c][f] += x[i][f];
    }

    // compute prior and mean
    for (int c = 0; c < C; c++) {
        prior[c] = (double)count[c] / N;
        for (int f = 0; f < F; f++)
            mean[c][f] /= count[c];
    }

    // pass 2 — compute variance
    for (int i = 0; i < N; i++) {
        int c = y[i];
        for (int f = 0; f < F; f++) {
            double diff = x[i][f] - mean[c][f];
            var[c][f] += diff * diff;
        }
    }
    for (int c = 0; c < C; c++)
        for (int f = 0; f < F; f++)
            var[c][f] /= count[c];

    // prediction — log scores
    double scores[C];
    for (int c = 0; c < C; c++) {
        scores[c] = log(prior[c]);
        for (int f = 0; f < F; f++) {
            double diff = test[f] - mean[c][f];
            scores[c] += -0.5 * log(2 * M_PI * var[c][f])
                       - (diff * diff) / (2 * var[c][f]);
        }
    }

    int prediction = scores[0] > scores[1] ? 0 : 1;

    printf("Score class 0: %f\n", scores[0]);
    printf("Score class 1: %f\n", scores[1]);
    printf("Prediction: class %d\n", prediction);

    return 0;
}
```

**Compile and run:**

```bash
gcc gaussian_nb.c -o gnb -lm && ./gnb
```

**Output:**

```
Score class 0: -6.287254
Score class 1: -17.504689
Prediction: class 0
```

---

## Implementation in NumPy

```python
import numpy as np

x = np.array([[150, 48], [155, 52], [160, 55], [165, 60], [170, 65],
              [175, 70], [180, 75], [185, 82], [190, 88], [195, 95]], dtype=float)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

test = np.array([160, 55], dtype=float)

# training — compute statistics per class
classes = np.unique(y)
mean    = np.zeros((len(classes), x.shape[1]))
var     = np.zeros((len(classes), x.shape[1]))
prior   = np.zeros(len(classes))

for c in classes:
    x_c      = x[y == c]              # boolean indexing — filter rows
    mean[c]  = np.mean(x_c, axis=0)
    var[c]   = np.var(x_c, axis=0)
    prior[c] = len(x_c) / len(x)

# prediction — log scores
scores = []
for c in classes:
    log_prior      = np.log(prior[c])
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var[c]) + ((test - mean[c])**2 / var[c]))
    scores.append(log_prior + log_likelihood)

prediction = int(classes[np.argmax(scores)])

print(f"Score class 0: {scores[0]:.6f}")
print(f"Score class 1: {scores[1]:.6f}")
print(f"Prediction: class {prediction}")
```

**Output:**

```
Score class 0: -6.287254
Score class 1: -17.504689
Prediction: class 0
```

---

## Testing Different Points

| Test point | Score 0 | Score 1 | Prediction | Confidence |
|---|---|---|---|---|
| `{160, 55}` | -6.28 | -17.50 | class 0 | high (11.22 gap) |
| `{172, 68}` | -9.73 | -9.59 | class 1 | low (0.14 gap) |
| `{185, 80}` | -20.61 | -6.70 | class 1 | high (13.91 gap) |

Bigger gap = more confident prediction. Points near the boundary get tiny gaps.

---

## C vs NumPy — What Changes

| | C | NumPy |
|---|---|---|
| Filter by class | manual loop + `if y[i]==c` | `x[y == c]` |
| Mean per feature | manual accumulate + divide | `np.mean(x_c, axis=0)` |
| Variance | second manual loop | `np.var(x_c, axis=0)` |
| Best class | ternary `scores[0] > scores[1]` | `np.argmax(scores)` |
| Lines of code | ~60 | ~20 |

---

## Key Takeaways

- Naive Bayes uses Bayes' Theorem — posterior ∝ likelihood × prior
- "Naive" = features assumed independent — wrong in practice but works well
- Gaussian NB models each feature as a normal distribution per class
- Training = computing mean, variance, and prior per class — no gradient descent
- Log scores prevent numerical underflow from multiplying tiny probabilities
- Two passes needed — mean first, then variance (variance needs mean)
- `x[y == c]` is NumPy boolean indexing — replaces an entire manual loop
- `np.argmax()` replaces the ternary comparison from C

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*