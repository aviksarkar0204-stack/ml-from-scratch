# DBSCAN from Scratch

> Building DBSCAN in raw C and NumPy — no sklearn, no magic. Just density, neighbors, and noise detection.

---

## What is DBSCAN?

DBSCAN (**D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise) finds clusters based on **density** — how many points are close together.

Unlike K-Means:
- You don't specify K — it finds the number of clusters automatically
- It can find **arbitrarily shaped** clusters — not just circular ones
- It identifies **noise points** — outliers that don't belong to any cluster

---

## Two Parameters

| Parameter | Meaning |
|---|---|
| `eps` | Maximum distance between two points to be considered neighbors |
| `min_samples` | Minimum number of neighbors to be a core point |

---

## Three Types of Points

```
Core point   → has ≥ min_samples neighbors within eps distance
Border point → within eps of a core point but < min_samples neighbors
Noise point  → not within eps of any core point → labeled -1
```

---

## DBSCAN vs K-Means

| | K-Means | DBSCAN |
|---|---|---|
| K specified | Yes — must tell it | No — finds automatically |
| Outliers | Assigned to nearest cluster | Labeled as noise (-1) |
| Cluster shape | Only circular | Any shape |
| Speed | Fast O(nK) | Slower O(n²) naive |
| Parameters | K | eps, min_samples |

---

## The Algorithm

```
labels = [-1, -1, ..., -1]   ← all unvisited
cluster_id = 0

for each unvisited point i:
    find all neighbors within eps
    if neighbors < min_samples:
        mark as noise (-1)
    else:
        start new cluster
        add all neighbors to queue
        while queue not empty:
            pt = queue.pop()
            label pt as cluster_id
            find pt's neighbors
            if pt is core point:
                add unvisited neighbors to queue
        cluster_id++
```

---

## Implementation in C

```c
#include <stdio.h>
#include <math.h>

#define N           9
#define EPS         2.0
#define MIN_SAMPLES 2

double distance(double *a, double *b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return sqrt(dx*dx + dy*dy);
}

int get_neighbors(double X[][2], int idx, int *neighbors) {
    int count = 0;
    for (int i = 0; i < N; i++)
        if (distance(X[idx], X[i]) <= EPS)
            neighbors[count++] = i;
    return count;
}

int main() {
    double X[N][2] = {
        {1,1}, {1,2}, {2,1}, {2,2},   // cluster 0
        {8,8}, {8,9}, {9,8}, {9,9},   // cluster 1
        {50,50}                         // noise
    };

    int labels[N];
    for (int i = 0; i < N; i++) labels[i] = -1;

    int cluster_id = 0;

    for (int i = 0; i < N; i++) {
        if (labels[i] != -1) continue;

        int neighbors[N];
        int n_neighbors = get_neighbors(X, i, neighbors);

        if (n_neighbors < MIN_SAMPLES) {
            labels[i] = -1;
            continue;
        }

        labels[i] = cluster_id;

        int queue[N], q_start = 0, q_end = 0;
        for (int k = 0; k < n_neighbors; k++)
            queue[q_end++] = neighbors[k];

        while (q_start < q_end) {
            int pt = queue[q_start++];
            if (labels[pt] == -1) labels[pt] = cluster_id;
            if (labels[pt] != -1 && labels[pt] != cluster_id) continue;

            labels[pt] = cluster_id;

            int pt_neighbors[N];
            int n_pt = get_neighbors(X, pt, pt_neighbors);

            if (n_pt >= MIN_SAMPLES)
                for (int k = 0; k < n_pt; k++)
                    if (labels[pt_neighbors[k]] == -1)
                        queue[q_end++] = pt_neighbors[k];
        }

        cluster_id++;
    }

    printf("DBSCAN Results:\n");
    for (int i = 0; i < N; i++)
        printf("  (%.0f, %.0f) -> cluster %d%s\n",
               X[i][0], X[i][1], labels[i],
               labels[i] == -1 ? " (noise)" : "");
    printf("\nTotal clusters found: %d\n", cluster_id);

    return 0;
}
```

**Compile and run:**

```bash
gcc dbscan.c -o dbscan -lm && ./dbscan
```

**Output:**

```
DBSCAN Results:
  (1, 1)   -> cluster 0
  (1, 2)   -> cluster 0
  (2, 1)   -> cluster 0
  (2, 2)   -> cluster 0
  (8, 8)   -> cluster 1
  (8, 9)   -> cluster 1
  (9, 8)   -> cluster 1
  (9, 9)   -> cluster 1
  (50, 50) -> cluster -1 (noise)

Total clusters found: 2
```

---

## Implementation in NumPy

```python
import numpy as np

def get_neighbors(X, idx, eps):
    distances = np.linalg.norm(X - X[idx], axis=1)
    return np.where(distances <= eps)[0]

x = np.array([[1,1], [1,2], [2,1], [2,2],
              [8,8], [8,9], [9,8], [9,9],
              [50,50]])

EPS         = 2.0
MIN_SAMPLES = 2
labels      = np.full(len(x), -1)
cluster_id  = 0

for i in range(len(x)):
    if labels[i] != -1:
        continue

    neighbors = get_neighbors(x, i, EPS)

    if len(neighbors) < MIN_SAMPLES:
        labels[i] = -1
        continue

    labels[i] = cluster_id
    queue = list(neighbors)

    while queue:
        pt = queue.pop(0)
        if labels[pt] == -1:
            labels[pt] = cluster_id
            pt_neighbors = get_neighbors(x, pt, EPS)
            if len(pt_neighbors) >= MIN_SAMPLES:
                for n in pt_neighbors:
                    if labels[n] == -1:
                        queue.append(n)

    cluster_id += 1

for i in range(len(x)):
    noise = " (noise)" if labels[i] == -1 else ""
    print(f"  {x[i]} -> cluster {labels[i]}{noise}")

print(f"\nTotal clusters found: {cluster_id}")
```

---

## C vs NumPy — What Changes

| | C | NumPy |
|---|---|---|
| Distance | manual `sqrt(dx²+dy²)` | `np.linalg.norm(X - X[idx], axis=1)` |
| Neighbors | manual loop + count | `np.where(distances <= eps)[0]` |
| Queue | manual array + `q_start/q_end` | Python list + `pop(0)` |
| Init labels | `for` loop setting `-1` | `np.full(len(x), -1)` |
| Lines of code | ~70 | ~30 |

---

## Effect of Parameters

| eps | min_samples | Result |
|---|---|---|
| Too small | any | Everything is noise |
| Too large | any | Everything is one cluster |
| Just right | too high | Core points become noise |
| Just right | just right | Correct clusters + noise |

Choosing `eps` and `min_samples` correctly is the main challenge of DBSCAN in practice. Common approach: plot k-distance graph and look for the elbow.

---

## Key Takeaways

- DBSCAN finds clusters automatically — no K needed
- Three point types: core (dense), border (edge), noise (isolated)
- `eps` controls neighborhood size — critical parameter
- `min_samples` controls minimum density for a core point
- Noise points get label `-1` — K-Means would force them into a cluster
- Queue-based expansion replaces recursion — avoids stack overflow on large datasets
- `np.linalg.norm(X - X[idx], axis=1)` computes all distances at once — replaces entire distance loop from C
- DBSCAN is O(n²) in naive implementation — slow on large datasets

---

*Part of the [ml-from-scratch](../) series — classic ML models built in raw C, then rebuilt in NumPy.*