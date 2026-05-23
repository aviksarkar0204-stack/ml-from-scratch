# ml-from-scratch

> Classic ML models built from scratch in raw C and NumPy — no sklearn, no magic. Just math and memory.

---

## Why?

Most ML courses teach you to call `model.fit()` without explaining what's happening inside. This repo builds every model twice:

- **In C** — raw loops, manual gradients, no libraries. Forces you to understand every line.
- **In NumPy** — same math, vectorized. Shows you what the abstractions are actually doing.

After doing both, sklearn becomes completely transparent.

---

## Series

| # | Model | C | NumPy | Concepts Covered |
|---|-------|---|-------|-----------------|
| 01 | Linear Regression | ✅ | ✅ | MSE, Gradient Descent, Chain Rule |
| 02 | Polynomial Regression | ✅ | ✅ | Feature Engineering, Overfitting |
| 03 | Ridge Regression (L2) | ✅ | ✅ | L2 Penalty, Regularization |
| 04 | Lasso Regression (L1) | ✅ | ✅ | L1 Penalty, Feature Selection |
| 05 | ElasticNet | ✅ | ✅ | L1 + L2 Combined |
| 06 | Logistic Regression | ✅ | ✅ | Sigmoid, Binary Cross-Entropy |
| 07 | K-Nearest Neighbors | ✅ | ✅ | Distance, No Training, Majority Vote |
| 08 | Gaussian Naive Bayes | ✅ | ✅ | Bayes Theorem, Normal Distribution |
| 09 | Multinomial Naive Bayes | ✅ | ✅ | Word Counts, Laplace Smoothing |
| 10 | Bernoulli Naive Bayes | ✅ | ✅ | Binary Features, Present/Absent |
| 11 | K-Means Clustering | ✅ | 🔜 | Unsupervised, Centroids, Convergence |
| 12 | Decision Tree | 🔜 | 🔜 | Recursion, Gini Impurity, Splitting |
| 13 | Random Forest | 🔜 | 🔜 | Ensemble Methods, Bagging |
| 14 | PCA | 🔜 | 🔜 | Dimensionality Reduction, Eigenvectors |
| 15 | DBSCAN | 🔜 | 🔜 | Density Clustering, Noise Points |
| 16 | SVM | 🔜 | 🔜 | Hyperplane, Margin, Kernel Trick |
| 17 | LDA | 🔜 | 🔜 | Linear Discriminant, Class Separation |
| 18 | Hierarchical Clustering | 🔜 | 🔜 | Dendrogram, Agglomerative |
| 19 | Isolation Forest | 🔜 | 🔜 | Anomaly Detection, Random Splits |
| 20 | Neural Network | 🔜 | 🔜 | Backpropagation, ReLU, Hidden Layers |

---

## Structure

```
ml-from-scratch/
├── 01_linear_regression/
│   ├── linear_regression.c
│   ├── linear_regression.py
│   └── Linear_Regression.md
├── 02_polynomial_regression/
│   ├── polynomial_regression.c
│   ├── polynomial_regression.py
│   └── Polynomial_Regression.md
├── 03_ridge_regression/
│   ├── ridge_regression.c
│   ├── ridge_regression.py
│   └── Ridge_Regression.md
├── 04_lasso_regression/
│   ├── lasso_regression.c
│   ├── lasso_regression.py
│   └── Lasso_Regression.md
├── 05_elasticnet/
│   ├── elasticnet.c
│   ├── elasticnet.py
│   └── ElasticNet.md
├── 06_logistic_regression/
│   ├── logistic_regression.c
│   ├── logistic_regression.py
│   └── Logistic_Regression.md
├── 07_knn/
│   ├── knn.c
│   ├── knn.py
│   └── KNN.md
├── 08_gaussian_naive_bayes/
│   ├── gaussian_nb.c
│   ├── gaussian_nb.py
│   └── Gaussian_Naive_Bayes.md
├── 09_multinomial_naive_bayes/
│   ├── multinomial_nb.c
│   ├── multinomial_nb.py
│   └── Multinomial_Naive_Bayes.md
├── 10_bernoulli_naive_bayes/
│   ├── bernoulli_nb.c
│   ├── bernoulli_nb.py
│   └── Bernoulli_Naive_Bayes.md
├── 11_kmeans/
│   ├── kmeans.c
│   ├── kmeans.py
│   └── KMeans.md
├── 12_decision_tree/
│   ├── decision_tree.c
│   ├── decision_tree.py
│   └── Decision_Tree.md
├── 13_random_forest/
│   ├── random_forest.c
│   ├── random_forest.py
│   └── Random_Forest.md
├── 14_pca/
│   ├── pca.c
│   ├── pca.py
│   └── PCA.md
├── 15_dbscan/
│   ├── dbscan.c
│   ├── dbscan.py
│   └── DBSCAN.md
├── 16_svm/
│   ├── svm.c
│   ├── svm.py
│   └── SVM.md
├── 17_lda/
│   ├── lda.c
│   ├── lda.py
│   └── LDA.md
├── 18_hierarchical_clustering/
│   ├── hierarchical.c
│   ├── hierarchical.py
│   └── Hierarchical_Clustering.md
├── 19_isolation_forest/
│   ├── isolation_forest.c
│   ├── isolation_forest.py
│   └── Isolation_Forest.md
└── 20_neural_network/
    ├── nn_xor.c
    ├── nn_xor.py
    └── Neural_Network.md
```

---

## How to Run

### C

```bash
gcc linear_regression.c -o lr && ./lr
```

> Add `-lm` for models using `exp()`, `log()`, or `sqrt()` — Logistic Regression, Naive Bayes, K-Means, Neural Network.

### Python

```bash
python linear_regression.py
```

> Only dependency is NumPy. Install with `pip install numpy`.

---

## What You Learn

**Building in C forces you to understand:**
- Where loss functions come from
- Why gradients have a `-2/N` factor
- What vectorized operations are actually doing under the hood
- How memory layout affects computation
- Why pointer arithmetic and index arrays matter
- How sorting, distance, and probability work at the lowest level

**Building in NumPy shows you:**
- How loops collapse into array operations
- Why NumPy is faster than pure Python
- The exact abstraction layer between math and sklearn
- Boolean indexing, broadcasting, and vectorized statistics

---

## Progress

```
Regression     ████████████████████ 5/5 complete
Classification ████████████░░░░░░░░ 5/8 complete
Clustering     ██░░░░░░░░░░░░░░░░░░ 1/4 complete
Reduction      ░░░░░░░░░░░░░░░░░░░░ 0/2 complete
Other          ░░░░░░░░░░░░░░░░░░░░ 0/1 complete
```

---

## Stack

![C](https://img.shields.io/badge/C-00599C?style=flat&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## Author

**Avik Sarkar**
B.Tech CSE (AI/ML) — Brainware University
[GitHub](https://github.com/aviksarkar0204-stack) · [Hugging Face](https://huggingface.co/Avik128)
