# Machine-Learning-from-Scratch
Implementations of core machine learning algorithms built from scratch using NumPy and Pandas — no sklearn model wrappers. Each notebook covers data prep, the core math, training, and evaluation.

---

## Algorithms Implemented

### Supervised Learning

| Algorithm | File | Task | Key Concepts |
|---|---|---|---|
| Multiple Linear Regression | `multiple_linear_regression.ipynb` | Regression | Gradient descent, MSE loss, R² score |
| Logistic Regression | `logistic_regression.ipynb` | Binary Classification | Sigmoid, cross-entropy loss, decision boundary |
| K-Nearest Neighbors | `KNN.ipynb` | Multi-class Classification | Euclidean distance, majority vote |
| Naive Bayes | `naive_bayes_algorithm.ipynb` | Multi-class Classification | Gaussian PDF, prior/likelihood/posterior |

### Unsupervised Learning

| Algorithm | File | Task | Key Concepts |
|---|---|---|---|
| K-Means Clustering | `k_means.ipynb` | Clustering | Centroid init, inertia, elbow method |
| Principal Component Analysis | `pca.ipynb` | Dimensionality Reduction | Eigendecomposition, explained variance, projection |

### Feature Engineering

| Algorithm | File | Task | Key Concepts |
|---|---|---|---|
| Feature Selection | `feature_selection.ipynb` | Preprocessing | ANOVA F-test (SelectKBest), cross-validation scoring |

---

## Structure

Each notebook follows this pattern:

1. **Data** — generate or load dataset
2. **Normalize** — zero mean, unit variance
3. **Core implementation** — algorithm built with NumPy only
4. **Evaluation** — accuracy, R², cross-val, or inertia
5. **Visualization** — decision boundaries, elbow curves, projections

---

## Stack

- Python 3.13
- NumPy
- Pandas
- Matplotlib
- scikit-learn (datasets and evaluation utilities only — no model imports)

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ml-from-scratch.git
cd ml-from-scratch
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

---

## Notes

- All models use **batch gradient descent** (no mini-batch or stochastic)
- Normalization is applied manually before training in every notebook
- PCA notebook uses eigendecomposition of the covariance matrix directly, then pipes the reduced features into logistic regression
- Naive Bayes uses log-space-free Gaussian PDF with an epsilon guard to prevent division by zero
