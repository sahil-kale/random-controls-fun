import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
n_samples = 100
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
X = np.hstack([np.ones_like(X), X])  # Add bias term

true_weights = np.array([2, 3])  # y = 2 + 3x
y = X @ true_weights + np.random.randn(n_samples) * 2.0  # add noise

# Linear regression using SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)
S_inv = np.diag(1 / S)
X_pseudo_inv = Vt.T @ S_inv @ U.T
w_svd = X_pseudo_inv @ y

# Predictions
y_pred = X @ w_svd

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 1], y, label="Noisy Data", alpha=0.6)
plt.plot(X[:, 1], y_pred, color='red', label="SVD Regression Fit", linewidth=2)
plt.title("Linear Regression using SVD")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()

w_svd, true_weights
plt.show()