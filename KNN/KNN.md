# K-Nearest Neighbors from Scratch

> Building KNN in raw C and NumPy — no sklearn, no magic. Just distance and voting.

---

## What is KNN?

K-Nearest Neighbors is a **lazy learner** — it has no training phase at all. Instead of learning parameters like `w` and `b`, it simply memorizes the entire training dataset.

When a new point comes in:
1. Compute distance from the new point to every training point
2. Pick the K closest neighbors
3. Take a majority vote → that's the prediction

No gradient descent. No loss function. No weight update.

---

## Why "Lazy Learner"?

| | Linear/Logistic Regression | KNN |
|---|---|---|
| Training | Learns `w`, `b` from data | Just stores the data |
| Prediction | One equation: `ŷ = w·x + b` | Computes distances to all points |
| Memory | Stores only `w`, `b` | Stores entire dataset |
| Speed | Fast at prediction | Slow at prediction (large datasets) |

The "work" is deferred to prediction time — that's why it's called lazy.

---

## What is K?

K is a hyperparameter — you choose it before running the algorithm.

- K = number of neighbors to look at, **not** number of classes
- Small K → sensitive to noise, can overfit
- Large K → smoother boundary, can underfit
- K should be **odd** for binary classification to avoid tie votes

Typical starting point: K = 3 or K = 5.

---

## Distance Metric

For 1D data, distance is just absolute difference:

```
d = |x_test - x_train|
```

For multi-dimensional data, Euclidean distance:

```
d = √( (x2-x1)² + (y2-y1)² + ... )
```

---

## The Algorithm

```
Given: training data (x, y), test point, K

1. dist[i] = |x[i] - test|  for all i
2. Sort indices by distance
3. Pick K smallest → k_neighbors
4. votes = sum of y[k_neighbors]
5. prediction = 1 if votes > K/2 else 0
```

---

## Implementation in C

```c
#include <stdio.h>
#include <math.h>

#define N 10
#define K 3

int main()
{
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double test = 6.5;

    // step 1 — compute distances
    double dist[N];
    for (int i = 0; i < N; i++)
        dist[i] = fabs(x[i] - test);

    // step 2 — sort indices by distance (bubble sort)
    int idx[N];
    for (int i = 0; i < N; i++) idx[i] = i;

    for (int i = 0; i < N-1; i++)
    {
        for (int j = i+1; j < N; j++)
        {
            if (dist[idx[j]] < dist[idx[i]])
            {
                int tmp = idx[i];
                idx[i]  = idx[j];
                idx[j]  = tmp;
            }
        }
    }

    // step 3 — vote
    int votes = 0;
    for (int i = 0; i < K; i++)
        votes += (int)y[idx[i]];

    int prediction = votes > K/2 ? 1 : 0;

    // print results
    printf("Test point: %.1f\n\n", test);
    printf("K=%d nearest neighbors:\n", K);
    for (int i = 0; i < K; i++)
        printf("  x: %.0f | dist: %.2f | class: %.0f\n",
               x[idx[i]], dist[idx[i]], y[idx[i]]);

    printf("\nVotes for class 1: %d / %d\n", votes, K);
    printf("Prediction: class %d\n", prediction);

    return 0;
}
```

**Compile and run:**

```bash
gcc knn.c -o knn -lm && ./knn
```

**Output:**

```
Test point: 6.5

K=3 nearest neighbors:
  x: 6 | dist: 0.50 | class: 1
  x: 7 | dist: 0.50 | class: 1
  x: 5 | dist: 1.50 | class: 0

Votes for class 1: 2 / 3
Prediction: class 1
```

---

## Implementation in NumPy

```python
import numpy as np

x    = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y    = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  dtype=float)
test = 6.5
K    = 5

# step 1 — compute distances
dist = np.abs(x - test)

# step 2 — sort and pick K nearest
idx   = np.argsort(dist)
k_idx = idx[:K]

# step 3 — vote
votes      = int(np.sum(y[k_idx]))
prediction = 1 if votes > K // 2 else 0

# print results
print(f"Test point: {test}")
print(f"\nK={K} nearest neighbors:")
for i in k_idx:
    print(f"  x: {x[i]:.0f} | dist: {dist[i]:.2f} | class: {y[i]:.0f}")

print(f"\nVotes for class 1: {votes} / {K}")
print(f"Prediction: class {prediction}")
```

**Output:**

```
Test point: 6.5

K=5 nearest neighbors:
  x: 6 | dist: 0.50 | class: 1
  x: 7 | dist: 0.50 | class: 1
  x: 5 | dist: 1.50 | class: 0
  x: 8 | dist: 1.50 | class: 1
  x: 4 | dist: 2.50 | class: 0

Votes for class 1: 3 / 5
Prediction: class 1
```

---

## C vs NumPy — What Changes

| | C | NumPy |
|---|---|---|
| Distance | `for` loop + `fabs()` | `np.abs(x - test)` |
| Sorting | Manual bubble sort on index array | `np.argsort(dist)` |
| Voting | Manual `for` loop + sum | `np.sum(y[k_idx])` |
| Lines of code | ~50 | ~10 |

The bubble sort in C is the most work — `np.argsort()` replaces it entirely in one call.

---

## Tie-Breaking Problem

When two points are equidistant from the test point, different implementations break ties differently:

- C (bubble sort, left to right) → picks lower index first
- NumPy (`np.argsort`) → picks lower index first too, but order may differ

**Fix:** use an odd K and avoid even splits. If ties are frequent in your data, increase K.

---

## Key Takeaways

- KNN has no training phase — it memorizes the data
- K controls how many neighbors vote, not how many classes exist
- Distance is the only computation — simpler math than gradient-based models
- `np.argsort()` replaces an entire bubble sort in one line
- KNN is slow at prediction on large datasets — it computes distances to every training point every time
- Always use odd K for binary classification to avoid tie votes

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*