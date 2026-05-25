import numpy as np

def get_neighbors(X, idx, eps):
    distances = np.linalg.norm(X - X[idx], axis=1)
    return np.where(distances <= eps)[0]

x = np.array([[1,1], [1,2], [2,1], [2,2],
     [8,8], [8,9], [9,8], [9,9],
     [50,50]])

EPS = 2.0
MIN_SAMPLES = 2
labels = np.full(len(x), -1)

cluster_id = 0

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

# print results
for i in range(len(x)):
    noise = " (noise)" if labels[i] == -1 else ""
    print(f"  {x[i]} -> cluster {labels[i]}{noise}")

print(f"\nTotal clusters found: {cluster_id}")