# Decision Tree from Scratch

> Building a Decision Tree in raw C and NumPy — no sklearn, no magic. Just recursion, Gini impurity, and binary trees.

---

## What is a Decision Tree?

A Decision Tree is a classification algorithm that learns a series of **if-else rules** from data:

```
Is feature 0 < 5.5?
    Yes → Is feature 1 < 3.5?
              Yes → class 0
              No  → class 1
    No  → class 1
```

Each split tries to separate classes as cleanly as possible. The tree grows recursively until all leaves are pure or a maximum depth is reached.

---

## Why It's Different from Everything Else

Every model built so far used flat loops and gradient descent. Decision Tree uses:

- **Binary tree data structure** — each node has left and right children
- **Recursion** — `build_tree()` calls itself to grow subtrees
- **Dynamic memory** — `malloc` in C, automatic in Python
- **No gradient descent** — splits are chosen by Gini impurity, not gradients

---

## Gini Impurity

Gini measures how "mixed" a group of samples is:

```
Gini = 1 - Σ p²
```

Where `p` is the proportion of each class. Scale: 0 (pure) to 0.5 (perfectly mixed).

**Examples:**

| Group | Gini | Meaning |
|-------|------|---------|
| `[0,0,0,0]` | 0.0 | Pure — all same class |
| `[0,0,1,1]` | 0.5 | Perfectly mixed |
| `[0,0,0,1]` | 0.375 | Slightly mixed |

**Weighted Gini for a split:**

```
Gini_split = (n_left/n) · Gini_left + (n_right/n) · Gini_right
```

Pick the split with the **lowest** weighted Gini — that's the purest separation.

---

## The Algorithm

```
build_tree(X, y, depth):
    if pure or max_depth reached:
        return leaf node with majority class

    for each feature f:
        for each unique value as threshold:
            split X into left (< thresh) and right (>= thresh)
            compute weighted Gini of split

    pick best (feature, threshold) with lowest Gini
    node.left  = build_tree(X_left,  y_left,  depth+1)
    node.right = build_tree(X_right, y_right, depth+1)
    return node

predict(node, x):
    if leaf: return node.class
    if x[node.feature] < node.threshold: return predict(node.left, x)
    else: return predict(node.right, x)
```

---

## Why Decision Trees Are Slow on Large Datasets

At every node, try every feature × every threshold:

```
Per node: O(N · F)
Full tree: O(N · F · log N)
```

For large N and F that gets expensive. Random Forest builds 100-1000 trees — multiply accordingly. sklearn speeds this up with Cython, sorted features, and parallel tree building.

---

## Implementation in C

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 6
#define F 2
#define MAX_DEPTH 3

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

    for (int f = 0; f < F; f++) {
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
                best_gini    = g;
                *best_feat   = f;
                *best_thresh = thresh;
            }
        }
    }
}

Node* build_tree(double X[][F], int *labels, int n, int depth) {
    Node *node = (Node*)malloc(sizeof(Node));
    int count[2] = {0, 0};
    for (int i = 0; i < n; i++) count[labels[i]]++;

    if (depth >= MAX_DEPTH || count[0] == 0 || count[1] == 0) {
        node->feature  = -1;
        node->left     = NULL;
        node->right    = NULL;
        node->class    = count[1] > count[0] ? 1 : 0;
        return node;
    }

    int best_feat;
    double best_thresh;
    best_split(X, labels, n, &best_feat, &best_thresh);

    double X_left[N][F], X_right[N][F];
    int l_labels[N], r_labels[N];
    int nl = 0, nr = 0;

    for (int i = 0; i < n; i++) {
        if (X[i][best_feat] < best_thresh) {
            for (int f = 0; f < F; f++) X_left[nl][f] = X[i][f];
            l_labels[nl++] = labels[i];
        } else {
            for (int f = 0; f < F; f++) X_right[nr][f] = X[i][f];
            r_labels[nr++] = labels[i];
        }
    }

    node->feature   = best_feat;
    node->threshold = best_thresh;
    node->class     = -1;
    node->left  = build_tree(X_left,  l_labels, nl, depth+1);
    node->right = build_tree(X_right, r_labels, nr, depth+1);
    return node;
}

int predict(Node *node, double *x) {
    if (node->feature == -1) return node->class;
    if (x[node->feature] < node->threshold)
        return predict(node->left, x);
    else
        return predict(node->right, x);
}

int main() {
    double X[N][F] = {{2,3},{3,3},{3,4},{6,7},{7,8},{8,7}};
    int y[N] = {0, 0, 0, 1, 1, 1};

    Node *tree = build_tree(X, y, N, 0);

    printf("Predictions:\n");
    for (int i = 0; i < N; i++) {
        int pred = predict(tree, X[i]);
        printf("  (%.0f, %.0f) -> predicted: %d | actual: %d %s\n",
               X[i][0], X[i][1], pred, y[i],
               pred == y[i] ? "OK" : "WRONG");
    }

    printf("\nNew points:\n");
    double test1[F] = {1, 2};
    double test2[F] = {7, 7};
    printf("  (1, 2) -> class %d\n", predict(tree, test1));
    printf("  (7, 7) -> class %d\n", predict(tree, test2));

    return 0;
}
```

**Compile and run:**

```bash
gcc decision_tree.c -o dt -lm && ./dt
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

class Node:
    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.class_    = None
        self.left      = None
        self.right     = None

def gini(labels):
    n = len(labels)
    if n == 0: return 0
    impurity = 1.0
    for c in np.unique(labels):
        p = np.sum(labels == c) / n
        impurity -= p * p
    return impurity

def best_split(X, y):
    best_gini   = 1e9
    best_feat   = None
    best_thresh = None

    for f in range(X.shape[1]):
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
    left_mask    = X[:, feat] < thresh
    right_mask   = ~left_mask

    node.feature   = feat
    node.threshold = thresh
    node.left  = build_tree(X[left_mask],  y[left_mask],  depth+1, max_depth)
    node.right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    return node

def predict(node, x):
    if node.class_ is not None: return node.class_
    if x[node.feature] < node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)

# data
X = np.array([[2,3],[3,3],[3,4],[6,7],[7,8],[8,7]])
y = np.array([0, 0, 0, 1, 1, 1])

tree = build_tree(X, y)

print("Predictions:")
for i in range(len(X)):
    pred = predict(tree, X[i])
    print(f"  {X[i]} -> predicted: {pred} | actual: {y[i]} {'OK' if pred==y[i] else 'WRONG'}")

print("\nNew points:")
print(f"  (1, 2) -> class {predict(tree, np.array([1, 2]))}")
print(f"  (7, 7) -> class {predict(tree, np.array([7, 7]))}")
```

---

## C vs Python — What Changes

| | C | Python |
|---|---|---|
| Node | `typedef struct` + `malloc` | `class Node` |
| Memory | manual `malloc` / `free` | automatic garbage collection |
| Gini | manual count array + loop | `np.unique` + boolean mask |
| Best split | 3 nested loops + output pointers | 2 loops + `~left_mask` |
| Build tree | recursive + manual array copy | recursive + boolean indexing |
| Predict | recursive pointer traversal | recursive attribute access |
| Lines of code | ~120 | ~55 |

---

## Key Takeaways

- Decision Tree uses a **binary tree data structure** — same as DSA, just with ML logic
- **Gini impurity** measures how mixed a node is — lower is better
- **Best split** tries every feature × threshold — most expensive part
- Tree grows **recursively** — `build_tree()` calls itself for left and right subtrees
- Stops when node is pure (`Gini = 0`) or `max_depth` is reached
- In C: `struct Node *left` is the same concept as linked list `struct Node *next`
- In Python: no pointers, no malloc — class attributes replace everything
- Decision Tree is the foundation for **Random Forest** — many trees voting together

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*