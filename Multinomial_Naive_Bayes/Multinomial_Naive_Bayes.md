# Multinomial Naive Bayes from Scratch

> Building Multinomial Naive Bayes in raw C and NumPy — the classic spam filter algorithm.

---

## What is Multinomial Naive Bayes?

Multinomial NB is a member of the Naive Bayes family designed for **count data** — specifically word frequencies in documents.

Same Bayes' Theorem as Gaussian NB:

```
P(class | features) ∝ P(features | class) · P(class)
```

But instead of assuming a normal distribution, it assumes features are **word counts** drawn from a multinomial distribution.

**Classic use case:** spam detection, document classification, sentiment analysis.

---

## The Data — Word Counts

Each sample is an email. Features are how many times each word appears:

```
Feature 0 = count of "buy"
Feature 1 = count of "free"
Feature 2 = count of "meeting"

X = [[3, 2, 0],   ← spam    (lots of "buy" and "free", no "meeting")
     [4, 1, 0],   ← spam
     [5, 3, 0],   ← spam
     [2, 2, 0],   ← spam
     [6, 4, 0],   ← spam
     [0, 0, 3],   ← not spam (no spam words, has "meeting")
     [0, 0, 2],   ← not spam
     [0, 0, 4],   ← not spam
     [0, 0, 2],   ← not spam
     [0, 0, 3]]   ← not spam
```

---

## How Likelihood Works

For each class, compute the probability of each word:

```
P(word | class) = (total count of word in class + 1) / (total words in class + V)
```

Where `V` = vocabulary size (number of features). The `+1` and `+V` is **Laplace smoothing**.

Then for a test document with word counts `[x₀, x₁, x₂]`:

```
log P(doc | class) = Σ xᵢ · log P(wordᵢ | class)
```

Each word's log probability is **weighted by how many times it appears** in the test document.

---

## Laplace Smoothing

Without smoothing, a word never seen in training for a class gets probability 0 — and `log(0) = -∞`, which breaks everything.

Fix: add 1 to every word count, add V to the total:

```
P(word | class) = (count + 1) / (total + V)
```

This gives every word a small non-zero probability even if it never appeared in training.

---

## The Algorithm

```
Training:
1. For each class c:
   - feat_count[c][f] = total count of feature f in class c
   - total[c]         = total word count in class c
   - prior[c]         = count of class c / total samples

Prediction:
2. For each class c:
   - prob[f]  = (feat_count[c][f] + 1) / (total[c] + F)
   - score[c] = log(prior[c]) + Σ test[f] · log(prob[f])
3. Predict class with highest score
```

---

## Implementation in C

```c
#include <stdio.h>
#include <math.h>

#define N 10
#define C 2
#define F 3   // "buy", "free", "meeting"

int main()
{
    double X[N][F] = {
        {3, 2, 0}, {4, 1, 0}, {5, 3, 0}, {2, 2, 0}, {6, 4, 0},
        {0, 0, 3}, {0, 0, 2}, {0, 0, 4}, {0, 0, 2}, {0, 0, 3}
    };
    int y[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    double test[F] = {5, 4, 0};   // heavy spam words

    double feat_count[C][F];
    double total[C];
    double prior[C];

    // initialize
    for (int c = 0; c < C; c++) {
        total[c] = 0;
        prior[c] = 0;
        for (int f = 0; f < F; f++)
            feat_count[c][f] = 0;
    }

    // accumulate word counts per class
    int class_count[C] = {0, 0};
    for (int i = 0; i < N; i++) {
        int c = y[i];
        class_count[c]++;
        for (int f = 0; f < F; f++) {
            feat_count[c][f] += X[i][f];
            total[c]         += X[i][f];
        }
    }

    // compute prior
    for (int c = 0; c < C; c++)
        prior[c] = (double)class_count[c] / N;

    // prediction with Laplace smoothing
    double scores[C];
    for (int c = 0; c < C; c++) {
        scores[c] = log(prior[c]);
        for (int f = 0; f < F; f++) {
            double prob = (feat_count[c][f] + 1) / (total[c] + F);
            scores[c] += test[f] * log(prob);
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
gcc multinomial_nb.c -o mnb -lm && ./mnb
```

**Output:**

```
Score class 0 (not spam): -26.192067
Score class 1 (spam):     -7.208870
Prediction: class 1 (spam)
```

---

## Implementation in NumPy

```python
import numpy as np

x = np.array([[3, 2, 0], [4, 1, 0], [5, 3, 0], [2, 2, 0], [6, 4, 0],
              [0, 0, 3], [0, 0, 2], [0, 0, 4], [0, 0, 2], [0, 0, 3]], dtype=float)
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

test    = np.array([5, 4, 0], dtype=float)
classes = np.unique(y)

feat_count = np.zeros((len(classes), x.shape[1]))
total      = np.zeros(len(classes))
prior      = np.zeros(len(classes))

for c in classes:
    x_c           = x[y == c]
    feat_count[c] = np.sum(x_c, axis=0)
    total[c]      = np.sum(x_c)
    prior[c]      = len(x_c) / len(x)

scores = []
for c in classes:
    c             = int(c)
    log_prior     = np.log(prior[c])
    prob          = (feat_count[c] + 1) / (total[c] + x.shape[1])
    log_likelihood = np.sum(test * np.log(prob))
    scores.append(log_prior + log_likelihood)

prediction = int(classes[np.argmax(scores)])
print(f"Score class 0 (not spam): {scores[0]:.6f}")
print(f"Score class 1 (spam):     {scores[1]:.6f}")
print(f"Prediction: class {prediction}")
```

---

## Multinomial vs Gaussian NB

| | Gaussian NB | Multinomial NB |
|---|---|---|
| Feature type | Continuous (height, weight) | Counts (word frequencies) |
| Likelihood | Gaussian PDF | Count probability |
| Training stats | Mean, variance per class | Word counts, total per class |
| Smoothing | Not needed | Laplace smoothing required |
| Use case | Medical data, sensor readings | Text classification, spam detection |

---

## Key Takeaways

- Multinomial NB is designed for word count data
- Likelihood weights log probability by how many times word appears: `x · log(p)`
- Laplace smoothing (`+1` numerator, `+V` denominator) prevents zero probabilities
- More training data → smoothing has less impact → more accurate predictions
- Sparse class data causes smoothing to dominate — always use balanced datasets
- `np.sum(x_c, axis=0)` replaces the entire manual accumulation loop from C

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*