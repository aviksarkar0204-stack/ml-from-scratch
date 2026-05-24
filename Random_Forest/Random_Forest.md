# Random Forest from Scratch

> Building Random Forest in raw C and NumPy — no sklearn, no magic. Just trees, bootstrapping, and majority voting.

---

## What is Random Forest?

Random Forest is an **ensemble** of Decision Trees. Instead of one tree, you build many trees and let them vote:

```
Tree 1  → predicts class 1
Tree 2  → predicts class 0
Tree 3  → predicts class 1
Tree 4  → predicts class 1
Tree 5  → predicts class 0

Majority vote → class 1 (3 vs 2)
```

---

## Why Not Just One Decision Tree?

A single Decision Tree **overfits** — it memorizes training data perfectly but fails on new data. When you average many different trees that each saw different data and features, errors cancel out.

```
One tree:    high variance, low bias  → overfits
Many trees:  low variance, low bias   → generalizes
```

This is called **variance reduction** through ensemble averaging.

---

## Two Sources of Randomness

**1. Bagging (Bootstrap Aggregating)**

Each tree trains on a random subset of data sampled **with replacement**:

```
Original: [A, B, C, D, E, F]
Tree 1:   [A, A, C, D, D, F]  ← some repeated, some missing
Tree 2:   [B, B, C, C, E, F]
Tree 3:   [A, C, D, D, E, E]
```

Each tree sees a slightly different dataset → trees make different errors → errors cancel when averaged.

**2. Random Feature Selection**

At each split, only consider a random subset of `√F` features instead of all F. Forces trees to be different from each other — even on the same data.

---

## The Algorithm

```
Training:
for each tree t in n_trees:
    sample N rows with replacement → X_boot, y_boot
    build Decision Tree on X_boot, y_boot with random feature subset

Prediction:
for each test point x:
    collect predictions from all trees
    return majority vote
```

---

## What's New Compared to Decision Tree

| Component | Decision Tree | Random Forest |
|---|---|---|
| Data per tree | full dataset | bootstrap sample |
| Features per split | all F | random √F |
| Number of models | 1 | n_trees |
| Prediction | direct | majority vote |
| Overfitting | prone | resistant |

Everything else — `Node`, `gini()`, `build_tree()`, `predict()` — is identical.

---

## Implementation in C

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N           6
#define F           2
#define MAX_DEPTH   3
#define NUM_TREES   10
#define MAX_FEATURES 1

typedef struct Node {
    int feature;
    double threshold;
    int class;
    struct Node *left;
    struct Node *right;
} Node;

double gini(int *labels, int n) {
    if (n == 0) return 0;
    int count[2] = {0, 0};
    for (int i = 0; i < n; i++) count[labels[i]]++;
    double impurity = 1.0;
    for (int c = 0; c < 2; c++) {
        double p = (double)count[c] / n;
        impurity -= p * p;
    }
    return impurity;
}

void best_split(double X[][F], int *labels, int n,
                int *best_feat, double *best_thresh) {
    double best_gini = 1e9;
    *best_feat = -1;
    *best_thresh = 0;

    // random feature subset
    int features[F] = {0, 1};
    for (int i = F-1; i > 0; i--) {
        int j = rand() % (i+1);
        int tmp = features[i]; features[i] = features[j]; features[j] = tmp;
    }

    for (int fi = 0; fi < MAX_FEATURES; fi++) {
        int f = features[fi];
        for (int i = 0; i < n; i++) {
            double thresh = X[i][f];
            int left_labels[N], right_labels[N];
            int nl = 0, nr = 0;
            for (int j = 0; j < n; j++) {
                if (X[j][f] < thresh) left_labels[nl++] = labels[j];
                else                  right_labels[nr++] = labels[j];
            }
            double g = ((double)nl/n) * gini(left_labels, nl)
                     + ((double)nr/n) * gini(right_labels, nr);
            if (g < best_gini) {
                best_gini = g; *best_feat = f; *best_thresh = thresh;
            }
        }
    }
}

Node* build_tree(double X[][F], int *labels, int n, int depth) {
    Node *node = (Node*)malloc(sizeof(Node));
    int count[2] = {0, 0};
    for (int i = 0; i < n; i++) count[labels[i]]++;

    if (depth >= MAX_DEPTH || count[0] == 0 || count[1] == 0) {
        node->feature = -1; node->left = NULL; node->right = NULL;
        node->class = count[1] > count[0] ? 1 : 0;
        return node;
    }

    int best_feat; double best_thresh;
    best_split(X, labels, n, &best_feat, &best_thresh);

    double X_left[N][F], X_right[N][F];
    int l_labels[N], r_labels[N], nl = 0, nr = 0;

    for (int i = 0; i < n; i++) {
        if (X[i][best_feat] < best_thresh) {
            for (int f = 0; f < F; f++) X_left[nl][f] = X[i][f];
            l_labels[nl++] = labels[i];
        } else {
            for (int f = 0; f < F; f++) X_right[nr][f] = X[i][f];
            r_labels[nr++] = labels[i];
        }
    }

    node->feature = best_feat; node->threshold = best_thresh; node->class = -1;
    node->left  = build_tree(X_left,  l_labels, nl, depth+1);
    node->right = build_tree(X_right, r_labels, nr, depth+1);
    return node;
}

int predict(Node *node, double *x) {
    if (node->feature == -1) return node->class;
    if (x[node->feature] < node->threshold) return predict(node->left, x);
    else return predict(node->right, x);
}

int main() {
    srand(time(NULL));

    double X[N][F] = {{2,3},{3,3},{3,4},{6,7},{7,8},{8,7}};
    int y[N] = {0, 0, 0, 1, 1, 1};

    // build forest
    Node *forest[NUM_TREES];
    for (int t = 0; t < NUM_TREES; t++) {
        double X_boot[N][F]; int y_boot[N];
        for (int i = 0; i < N; i++) {
            int idx = rand() % N;
            for (int f = 0; f < F; f++) X_boot[i][f] = X[idx][f];
            y_boot[i] = y[idx];
        }
        forest[t] = build_tree(X_boot, y_boot, N, 0);
    }

    // predict with majority vote
    printf("Predictions:\n");
    for (int i = 0; i < N; i++) {
        int votes[2] = {0, 0};
        for (int t = 0; t < NUM_TREES; t++)
            votes[predict(forest[t], X[i])]++;
        int pred = votes[1] > votes[0] ? 1 : 0;
        printf("  (%.0f, %.0f) -> predicted: %d | actual: %d %s\n",
               X[i][0], X[i][1], pred, y[i],
               pred == y[i] ? "OK" : "WRONG");
    }

    printf("\nNew points:\n");
    double test1[F] = {1, 2}, test2[F] = {7, 7};
    int v1[2] = {0,0}, v2[2] = {0,0};
    for (int t = 0; t < NUM_TREES; t++) {
        v1[predict(forest[t], test1)]++;
        v2[predict(forest[t], test2)]++;
    }
    printf("  (1, 2) -> class %d\n", v1[1] > v1[0] ? 1 : 0);
    printf("  (7, 7) -> class %d\n", v2[1] > v2[0] ? 1 : 0);

    return 0;
}
```

**Compile and run:**

```bash
gcc random_forest.c -o rf -lm && ./rf
```

**Output:**

```
Predictions:
  (2, 3) -> predicted: 0 | actual: 0 OK
  (3, 3) -> predicted: 0 | actual: 0 OK
  (3, 4) -> predicted: 0 | actual: 0 OK
  (6, 7) -> predicted: 1 | actual: 1 OK
  (7, 8) -> predicted: 1 | actual: 1 OK
  (8, 7) -> predicted: 1 | actual: 1 OK

New points:
  (1, 2) -> class 0
  (7, 7) -> class 1
```

---

## Implementation in NumPy / Python

```python
import numpy as np
import random

class Node:
    def __init__(self):
        self.feature = self.threshold = self.class_ = self.left = self.right = None

def gini(labels):
    n = len(labels)
    if n == 0: return 0
    impurity = 1.0
    for c in np.unique(labels):
        p = np.sum(labels == c) / n
        impurity -= p * p
    return impurity

def best_split(X, y):
    best_gini = best_feat = best_thresh = None
    best_gini = 1e9
    features = random.sample(range(X.shape[1]), k=max(1, int(np.sqrt(X.shape[1]))))
    for f in features:
        for thresh in np.unique(X[:, f]):
            left_mask  = X[:, f] < thresh
            right_mask = ~left_mask
            g = (np.sum(left_mask)/len(y))  * gini(y[left_mask]) \
              + (np.sum(right_mask)/len(y)) * gini(y[right_mask])
            if g < best_gini:
                best_gini, best_feat, best_thresh = g, f, thresh
    return best_feat, best_thresh

def build_tree(X, y, depth=0, max_depth=3):
    node = Node()
    if depth >= max_depth or len(np.unique(y)) == 1:
        node.class_ = int(np.bincount(y).argmax())
        return node
    feat, thresh = best_split(X, y)
    left_mask = X[:, feat] < thresh
    node.feature, node.threshold = feat, thresh
    node.left  = build_tree(X[left_mask],  y[left_mask],  depth+1, max_depth)
    node.right = build_tree(X[~left_mask], y[~left_mask], depth+1, max_depth)
    return node

def predict(node, x):
    if node.class_ is not None: return node.class_
    if x[node.feature] < node.threshold: return predict(node.left, x)
    else: return predict(node.right, x)

X = np.array([[2,3],[3,3],[3,4],[6,7],[7,8],[8,7]])
y = np.array([0, 0, 0, 1, 1, 1])

N_TREES = 10
forest  = []
for _ in range(N_TREES):
    idx    = np.random.choice(len(X), len(X), replace=True)
    forest.append(build_tree(X[idx], y[idx]))

print("Predictions:")
for i in range(len(X)):
    votes = [predict(tree, X[i]) for tree in forest]
    pred  = int(np.bincount(votes).argmax())
    print(f"  {X[i]} -> predicted: {pred} | actual: {y[i]} {'OK' if pred==y[i] else 'WRONG'}")

print("\nNew points:")
for point in [np.array([1,2]), np.array([7,7])]:
    votes = [predict(tree, point) for tree in forest]
    print(f"  {point} -> class {int(np.bincount(votes).argmax())}")
```

---

## C vs Python — What Changes

| | C | Python |
|---|---|---|
| Bootstrap | manual `rand() % N` loop | `np.random.choice(len(X), replace=True)` |
| Forest storage | `Node *forest[NUM_TREES]` | `forest = []` |
| Voting | manual `votes[2]` array | `np.bincount(votes).argmax()` |
| Random features | Fisher-Yates shuffle | `random.sample()` |
| Lines of code | ~120 | ~50 |

---

## Key Takeaways

- Random Forest = many Decision Trees + majority vote
- **Bagging** — each tree trains on a bootstrap sample (random with replacement)
- **Random features** — each split considers only `√F` random features
- Both sources of randomness force trees to be different → errors cancel → less overfitting
- All core functions (`gini`, `build_tree`, `predict`) are identical to Decision Tree
- Only additions: bootstrap loop, forest list, voting at prediction time
- More trees → more stable predictions but slower training
- Random Forest is the foundation for understanding **gradient boosting** (XGBoost)

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*