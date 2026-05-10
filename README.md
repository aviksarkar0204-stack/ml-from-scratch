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
| 02 | Logistic Regression | ✅ | ✅ | Sigmoid, Binary Cross-Entropy |
| 03 | K-Nearest Neighbors | ✅ | ✅ | Distance, No Training, Majority Vote |
| 04 | Polynomial Regression | 🔜 | 🔜 | Feature Engineering, Overfitting |
| 05 | Decision Tree | 🔜 | 🔜 | Recursion, Gini Impurity, Splitting |
| 06 | Random Forest | 🔜 | 🔜 | Ensemble Methods, Bagging |
| 07 | Neural Network | 🔜 | 🔜 | Backpropagation, ReLU, Hidden Layers |

---

## Structure

```
ml-from-scratch/
├── 01_linear_regression/
│   ├── linear_regression.c
│   ├── linear_regression.py
│   └── Linear_Regression.md
├── 02_logistic_regression/
│   ├── logistic_regression.c
│   ├── logistic_regression.py
│   └── Logistic_Regression.md
├── 03_knn/
│   ├── knn.c
│   ├── knn.py
│   └── KNN.md
├── 04_polynomial_regression/
│   ├── polynomial_regression.c
│   ├── polynomial_regression.py
│   └── Polynomial_Regression.md
├── 05_decision_tree/
│   ├── decision_tree.c
│   ├── decision_tree.py
│   └── Decision_Tree.md
├── 06_random_forest/
│   ├── random_forest.c
│   ├── random_forest.py
│   └── Random_Forest.md
└── 07_neural_network/
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

> Add `-lm` for models using `exp()`, `log()`, or `sqrt()` — Logistic Regression, Neural Network.

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

**Building in NumPy shows you:**
- How loops collapse into array operations
- Why NumPy is faster than pure Python
- The exact abstraction layer between math and sklearn

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
