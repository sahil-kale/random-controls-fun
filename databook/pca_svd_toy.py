import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Toy dataset: [height in cm, weight in kg]
X = np.array([
    [170, 650],   # <-- 10x weight
    [160, 600],
    [180, 800],
    [175, 750],
    [165, 550]
])

# Transpose to shape (samples, features)
X = X.astype(float)

# ===== PCA on raw data =====
pca_raw = PCA()
X_centered = X - np.mean(X, axis=0)
pca_raw.fit(X_centered)
X_pca_raw = pca_raw.transform(X_centered)

# ===== PCA on standardized data =====
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

pca_std = PCA()
pca_std.fit(X_standardized)
X_pca_std = pca_std.transform(X_standardized)

# ===== Plotting =====
plt.figure(figsize=(12, 5))

# --- Raw PCA ---
plt.subplot(1, 2, 1)
plt.scatter(X_centered[:, 0], X_centered[:, 1], label='Data')
for length, vector in zip(pca_raw.explained_variance_, pca_raw.components_):
    v = vector * 20  # scale for visualization
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')
plt.title("PCA on Raw Data")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.axis('equal')
plt.grid(True)

# --- Standardized PCA ---
plt.subplot(1, 2, 2)
plt.scatter(X_standardized[:, 0], X_standardized[:, 1], label='Standardized Data')
for length, vector in zip(pca_std.explained_variance_, pca_std.components_):
    v = vector * 2  # smaller scale since already normalized
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='g')
plt.title("PCA on Standardized Data")
plt.xlabel("Standardized Height")
plt.ylabel("Standardized Weight")
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()
