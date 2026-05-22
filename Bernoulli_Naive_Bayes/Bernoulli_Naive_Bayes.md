# Bernoulli Naive Bayes from Scratch

> Building Bernoulli Naive Bayes in raw C and NumPy — binary feature classification from scratch.

---

## What is Bernoulli Naive Bayes?

Bernoulli NB is designed for **binary features** — word present (1) or absent (0). Unlike Multinomial NB which cares about *how many times* a word appears, Bernoulli NB only cares about *whether* it appears.

Same Bayes' Theorem:

```
P(class | features) ∝ P(features | class) · P(class)
```

But likelihood uses the **Bernoulli distribution**.

---

## Multinomial vs Bernoulli — Key Difference

| | Multinomial NB | Bernoulli NB |
|---|---|---|
| Feature values | Word counts (0, 1, 2, 3...) | Binary (0 or 1) |
| What it measures | How many times word appears | Whether word appears |
| Likelihood | `x · log(p)` | `x · log(p) + (1-x) · log(1-p)` |
| Smoothing denominator | `total + V` | `class_count + 2` |

The critical difference: Bernoulli explicitly **penalizes absent words**. If a spam word is absent from the test document, Bernoulli adds `log(1-p)` — a negative contribution. Multinomial just ignores absent words entirely.

---

## The Data — Binary Word Presence

```
Feature 0 = "buy" present (1) or absent (0)
Feature 1 = "free" present (1) or absent (0)

X = [[1, 1],   ← spam    (both words present)
     [1, 0],   ← spam    ("buy" present, "free" absent)
     [1, 1],   ← spam
     [1, 1],   ← spam
     [1, 0],   ← spam
     [0, 0],   ← not spam (neither word present)
     [0, 0],   ← not spam
     [0, 1],   ← not spam ("free" present but not "buy")
     [0, 0],   ← not spam
     [0, 1]]   ← not spam
```

---

## The Likelihood Formula

For each feature `f` and class `c`:

```
p = (feat_count[c][f] + 1) / (class_count[c] + 2)
```

`feat_count[c][f]` = number of **documents** in class `c` that contain word `f`
`+2` in denominator = Laplace smoothing for binary feature (2 possible values: 0 or 1)

Then the Bernoulli log likelihood:

```
log P(x | class) = Σ [ x · log(p) + (1-x) · log(1-p) ]
```

- When `x = 1` (word present): adds `log(p)` — how likely is this word in this class?
- When `x = 0` (word absent): adds `log(1-p)` — how likely is this word **absent** from this class?

---

## The Algorithm

```
Training:
1. For each class c:
   - feat_count[c][f] = number of documents in class c containing feature f
   - class_count[c]   = total documents in class c
   - prior[c]         = class_count[c] / total documents

Prediction:
2. For each class c:
   - p[f]     = (feat_count[c][f] + 1) / (class_count[c] + 2)
   - score[c] = log(prior[c]) + Σ [test[f] · log(p[f]) + (1-test[f]) · log(1-p[f])]
3. Predict class with highest score
```

---

## Implementation in C

```c
#include <stdio.h>
#include <math.h>

#define N 10
#define C 2
#define F 2

int main()
{
    double X[N][F] = {
        {1, 1}, {1, 0}, {1, 1}, {1, 1}, {1, 0},
        {0, 0}, {0, 0}, {0, 1}, {0, 0}, {0, 1}
    };
    int y[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    double test[F] = {1, 1};

    double feat_count[C][F];
    int class_count[C];
    double prior[C];

    // initialize
    for (int c = 0; c < C; c++) {
        class_count[c] = 0;
        prior[c] = 0;
        for (int f = 0; f < F; f++)
            feat_count[c][f] = 0;
    }

    // count document presence per class
    for (int i = 0; i < N; i++) {
        int c = y[i];
        class_count[c]++;
        for (int f = 0; f < F; f++)
            feat_count[c][f] += X[i][f];   // X[i][f] is 0 or 1
    }

    // compute prior
    for (int c = 0; c < C; c++)
        prior[c] = (double)class_count[c] / N;

    // prediction with Bernoulli likelihood
    double scores[C];
    for (int c = 0; c < C; c++) {
        scores[c] = log(prior[c]);
        for (int f = 0; f < F; f++) {
            double p = (feat_count[c][f] + 1) / (class_count[c] + 2);
            scores[c] += test[f] * log(p) + (1 - test[f]) * log(1 - p);
        }
    }

    int prediction = scores[0] > scores[1] ? 0 : 1;
    printf("Score class 0 (not spam): %f\n", scores[0]);
    printf("Score class 1 (spam):     %f\n", scores[1]);
    printf("Prediction: class %d %s\n", prediction,
           prediction == 1 ? "(spam)" : "(not spam)");

    return 0;
}
```

**Compile and run:**

```bash
gcc bernoulli_nb.c -o bnb -lm && ./bnb
```

**Output:**

```
Score class 0 (not spam): -3.486355
Score class 1 (spam):     -1.406914
Prediction: class 1 (spam)
```

---

## Implementation in NumPy

```python
import numpy as np

x = np.array([[1, 0], [1, 1], [1, 1], [1, 1], [1, 0],
              [0, 0], [0, 0], [0, 1], [0, 0], [0, 1]], dtype=float)
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

test    = np.array([1, 0], dtype=float)
classes = np.unique(y)

feat_count = np.zeros((len(classes), x.shape[1]))
total      = np.zeros(len(classes))
prior      = np.zeros(len(classes))

for c in classes:
    x_c           = x[y == c]
    feat_count[c] = np.sum(x_c, axis=0)
    total[c]      = len(x_c)              # document count, not word count
    prior[c]      = len(x_c) / len(x)

scores = []
for c in classes:
    c              = int(c)
    log_prior      = np.log(prior[c])
    prob           = (feat_count[c] + 1) / (total[c] + 2)
    log_likelihood = np.sum(test * np.log(prob) + (1 - test) * np.log(1 - prob))
    scores.append(log_prior + log_likelihood)

prediction = int(classes[np.argmax(scores)])
print(f"Score class 0 (not spam): {scores[0]:.6f}")
print(f"Score class 1 (spam):     {scores[1]:.6f}")
print(f"Prediction: class {prediction}")
```

**Output:**

```
Score class 0 (not spam): -3.198673
Score class 1 (spam):     -1.694596
Prediction: class 1
```

---

## Testing Different Points

| Test point | Meaning | Prediction | Confidence |
|---|---|---|---|
| `{1, 1}` | both words present | spam | high |
| `{1, 0}` | only "buy" present | spam | moderate |
| `{0, 0}` | neither word present | not spam | high |
| `{0, 1}` | only "free" present | not spam | low |

---

## All Three Naive Bayes Variants

| Variant | Feature type | Likelihood | Use case |
|---|---|---|---|
| Gaussian NB | Continuous | Normal distribution | Height, weight, sensor data |
| Multinomial NB | Word counts | Count probability | Spam filter, topic classification |
| Bernoulli NB | Binary | Present/absent | Short text, binary features |

---

## Key Takeaways

- Bernoulli NB uses binary features — word present or absent, not count
- Likelihood: `x · log(p) + (1-x) · log(1-p)` — explicitly handles both presence AND absence
- Smoothing denominator is `+2` (not `+V`) — binary has only 2 possible values
- `feat_count[c]` counts **documents** containing each word, not total word occurrences
- `total[c] = len(x_c)` — number of documents, not sum of word counts
- Bernoulli penalizes absent spam words — Multinomial ignores them entirely

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*