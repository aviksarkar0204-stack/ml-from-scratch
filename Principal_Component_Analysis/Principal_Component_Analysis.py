import numpy as np

X = np.array([[2.5, 2.4, 0.0],
              [0.5, 0.7, 0.0],
              [2.2, 2.9, 0.0],
              [1.9, 2.2, 0.0],
              [3.1, 3.0, 0.0],
              [2.3, 2.7, 0.0]])

mean = np.mean(X, axis=0)
X_centered = X - mean
print("Mean:", mean)
print("Centered:\n", X_centered)

cov = np.cov(X_centered.T)
print("Covariance matrix:\n", cov)

eigenvalues, eigenvectors = np.linalg.eig(cov)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

K = 2  # reduce to 2 dimensions

# sort by eigenvalue descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# keep top K eigenvectors
components = eigenvectors[:, :K]

# project data
X_pca = X_centered @ components

print("Top K eigenvalues:", eigenvalues[:K])
print("Components:\n", components)
print("Projected data:\n", X_pca)

total_var = np.sum(eigenvalues)
explained = eigenvalues[:K] / total_var * 100
print(f"\nVariance explained: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%")