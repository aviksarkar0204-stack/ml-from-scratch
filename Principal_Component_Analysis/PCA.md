# Principal Component Analysis (PCA) from Scratch

> Building PCA in NumPy — no sklearn, no magic. Just linear algebra and eigendecomposition.

---

## What is PCA?

Principal Component Analysis is a **dimensionality reduction** technique. It takes high-dimensional data and finds the directions of maximum variance — called **principal components** — then projects data onto fewer dimensions.

**Example:** 100 features → PCA finds 2 most important directions → project to 2D → now you can visualize it.

---

## Why Dimensionality Reduction?

| Problem | Solution |
|---|---|
| Too many features → slow training | Reduce to top K components |
| Can't visualize high-dimensional data | Project to 2D or 3D |
| Correlated features → redundant information | PCA finds independent directions |
| Noise in data | Lower components often capture noise → drop them |

---

## Dimensions in ML vs Geometry

A common confusion:

| Array shape | ML meaning | Geometry meaning |
|---|---|---|
| `(6, 3)` | 6 samples, 3 features | 6 points in 3D space |
| `(1000, 784)` | 1000 samples, 784 features | 1000 points in 784D space |

In ML, **dimensions = features**, not spatial axes. MNIST images are 784-dimensional because each pixel is a feature — not because they're cubes.

---

## Eigenvalues and Eigenvectors

For a square matrix `A`, an eigenvector `v` only gets **scaled** when multiplied — it never changes direction:

```
A · v = λ · v
```

- `v` → eigenvector (direction)
- `λ` → eigenvalue (how much it scales)

**Why PCA uses them:**

The covariance matrix `C` describes how features vary together. Its eigenvectors point in the directions of maximum variance. Its eigenvalues tell you how much variance exists in each direction.

```
Largest eigenvalue  → direction of most spread  → most important
Second largest      → second most important direction
...
Zero eigenvalue     → no variance in that direction → drop it
```

---

## The Algorithm — 5 Steps

```
1. Center:     X_centered = X - mean(X)
2. Covariance: C = cov(X_centered)
3. Eigen:      eigenvalues, eigenvectors = eig(C)
4. Sort:       sort eigenvectors by eigenvalue descending
5. Project:    X_pca = X_centered @ top_K_eigenvectors
```

---

## Why Not C?

PCA requires eigendecomposition — a numerical algorithm that's complex to implement from scratch in C. NumPy's `np.linalg.eig()` handles it in one line using optimized LAPACK routines under the hood.

PCA's value is in understanding the math, not fighting C's lack of linear algebra libraries.

---

## Implementation in NumPy

```python
import numpy as np

# 3D data — third feature is all zeros (no variance)
X = np.array([[2.5, 2.4, 0.0],
              [0.5, 0.7, 0.0],
              [2.2, 2.9, 0.0],
              [1.9, 2.2, 0.0],
              [3.1, 3.0, 0.0],
              [2.3, 2.7, 0.0]])

K = 2   # reduce to 2 dimensions

# step 1 — center the data
mean       = np.mean(X, axis=0)
X_centered = X - mean
print("Mean:", mean)

# step 2 — covariance matrix
cov = np.cov(X_centered.T)
print("Covariance matrix:\n", cov)

# step 3 — eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)
print("Eigenvalues:", eigenvalues)

# step 4 — sort by eigenvalue descending
idx          = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# keep top K eigenvectors
components = eigenvectors[:, :K]
print("Top K eigenvalues:", eigenvalues[:K])

# step 5 — project data
X_pca = X_centered @ components
print("Projected data:\n", X_pca)

# variance explained
total_var = np.sum(eigenvalues)
explained = eigenvalues[:K] / total_var * 100
print(f"\nVariance explained: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%")
```

**Output:**

```
Mean: [2.083 2.317 0.0]

Covariance matrix:
 [[0.762  0.692  0.0  ]
  [0.692  0.718  0.0  ]
  [0.0    0.0    0.0  ]]

Eigenvalues: [1.432  0.047  0.0]

Top K eigenvalues: [1.432  0.047]

Projected data:
 [[ 0.357  -0.230]
  [-2.262  -0.060]
  [ 0.490   0.338]
  [-0.213   0.044]
  [ 1.206  -0.217]
  [ 0.422   0.125]]

Variance explained: PC1=96.8%, PC2=3.2%
```

---

## Reading the Output

**Covariance matrix:**
```
[[0.762  0.692  0.0]
 [0.692  0.718  0.0]
 [0.0    0.0    0.0]]  ← third row/col all zeros → no variance → will be dropped
```

**Eigenvalues:** `[1.432, 0.047, 0.0]`
- PC1 = 1.432 → most important direction
- PC2 = 0.047 → second direction
- PC3 = 0.0   → zero variance → correctly dropped ✓

**Variance explained:**
- PC1 explains `1.432 / 1.479 = 96.8%` of total variance
- PC2 explains `0.047 / 1.479 = 3.2%`
- PC3 explains `0.0%` — useless dimension detected automatically ✓

**Result:** 3D → 2D with 100% of useful information preserved.

---

## How to Choose K

| Method | Rule |
|---|---|
| Fixed K | Always reduce to 2D (for visualization) |
| Variance threshold | Keep components until 95% variance explained |
| Elbow method | Plot eigenvalues, pick where curve bends |

For this dataset — keeping 2 components captures 100% of useful variance (PC3 = 0).

---

## PCA in 5 Lines

```python
mean       = np.mean(X, axis=0)
X_centered = X - mean
cov        = np.cov(X_centered.T)
vals, vecs = np.linalg.eig(cov)
X_pca      = X_centered @ vecs[:, :K]
```

Everything else is just sorting and printing intermediate results.

---

## Key Takeaways

- PCA finds directions of maximum variance — called principal components
- Each principal component is an eigenvector of the covariance matrix
- Eigenvalue = how much variance exists in that direction — larger = more important
- Sort eigenvectors by eigenvalue descending, keep top K
- Centering data first is mandatory — PCA assumes zero-mean data
- Zero eigenvalue = zero variance = useless dimension — PCA drops it automatically
- `np.linalg.eig()` replaces what would be hundreds of lines of numerical C code
- PCA is unsupervised — no labels needed, no gradient descent, pure linear algebra

---

## What PCA Is Not

- PCA does not select features — it creates new ones (linear combinations)
- PCA does not always work — assumes linear relationships between features
- PCA loses interpretability — projected features have no direct meaning
- For non-linear dimensionality reduction → t-SNE or UMAP

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*