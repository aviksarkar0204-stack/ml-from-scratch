import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
y = np.array([0,0,0,0,0,1,1,1,1,1], dtype=float)
test = 4.5
K=5
dist = np.abs(x - test)
idx = np.argsort(dist)
k_idx = idx[:K]
votes = int(np.sum(y[k_idx]))
prediction = 1 if votes > K // 2 else 0
print(f"\nTest point: {test}")
print(f"K={K} nearest neighbors:")
for i in k_idx:
    print(f"  x: {x[i]:.0f} | dist: {dist[i]:.2f} | class: {y[i]:.0f}")
print(f"\nVotes for class 1: {votes} / {K}")
print(f"Prediction: class {prediction}")