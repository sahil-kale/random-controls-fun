# Compressed sensing toy example (NumPy + plain matrix ops, simple OMP solver)
# Signal: two-tone temperature-ish time series (sparse in frequency/DCT basis).
# Steps:
# 1) Build a cosine dictionary Psi (DCT-II) explicitly with NumPy.
# 2) Generate a K-sparse coefficient vector s_true.
# 3) Form x = Psi @ s_true (full-resolution signal).
# 4) Take p << n random samples: y = C x (C selects rows).
# 5) Recover s_hat via Orthogonal Matching Pursuit (OMP) we implement.
# 6) Reconstruct x_hat = Psi @ s_hat and compare.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# --- 1) DCT-II matrix (orthonormal) -------------------------------------------
def dct_matrix(n: int) -> np.ndarray:
    """
    Orthonormal DCT-II matrix (n x n), constructed explicitly with cosines.
    Psi[k, n] = alpha_k * cos( pi * k * (2n+1) / (2N) ), 0 <= k < N
    with alpha_0 = sqrt(1/N), alpha_k = sqrt(2/N) for k>0
    """
    N = n
    k = np.arange(N)[:, None]         # column vector of frequency indices
    n_idx = np.arange(N)[None, :]     # row vector of time indices
    mat = np.cos(np.pi * (2*n_idx + 1) * k / (2*N))
    alpha = np.sqrt(2/N) * np.ones((N, 1))
    alpha[0, 0] = np.sqrt(1/N)
    return alpha @ np.ones((1, N)) * mat  # scale rows by alpha_k

n = 1024
Psi = dct_matrix(n)

# Optional: check near-orthonormality numerically
orth_err = np.linalg.norm(Psi @ Psi.T - np.eye(n)) / np.linalg.norm(np.eye(n))
print(f"Psi orthonormality relative error: {orth_err:.2e}")

# --- 2) Make a K-sparse coefficient vector in this basis ----------------------
K_true = 2
s_true = np.zeros(n)

# Two "temperature-like" low-frequency atoms (daily + HVAC-ish)
k1, k2 = 4, 17
s_true[k1] = 10.0
s_true[k2] = 6.0 * np.sign(rng.standard_normal())

# --- 3) Full signal -----------------------------------------------------------
x = Psi @ s_true  # time-series

# --- 4) Random compressed measurements y = C x -------------------------------
p = 140  # number of measurements << n
idx = np.sort(rng.choice(n, size=p, replace=False))  # random sample times
y = x[idx]                      # C x == x[idx]
Theta = Psi[idx, :]             # (p x n) = C Psi

# --- 5) Orthogonal Matching Pursuit (OMP) ------------------------------------
def omp(Theta, y, K, tol=1e-8):
    """
    Solve y ≈ Theta @ s with K-sparse s via greedy OMP.
    Returns s_hat (length n) and the selected support indices.
    """
    p, n = Theta.shape
    residual = y.copy()
    support = []
    s_hat = np.zeros(n)

    for _ in range(K):
        # pick column most correlated with residual
        corr = Theta.T @ residual 
        # avoid duplicates
        for pick in np.argsort(np.abs(corr))[::-1]:
            if pick not in support:
                j = pick
                break
        support.append(j)

        # least squares on active set
        Th_S = Theta[:, support]                  # (p x |S|)
        s_S, *_ = np.linalg.lstsq(Th_S, y, rcond=None)
        residual = y - Th_S @ s_S

        if np.linalg.norm(residual) <= tol:
            break

    s_hat[support] = s_S
    return s_hat, support

K_max = 6  # overestimate a bit; OMP will settle via residual
s_hat, support = omp(Theta, y, K=K_max, tol=1e-10)
x_hat = Psi @ s_hat

# --- Diagnostics --------------------------------------------------------------
recon_rmse = np.sqrt(np.mean((x - x_hat)**2))
print("---- Diagnostics ----")
print(f"n={n}, p={p}, K_true={K_true}, K_max={K_max}")
print(f"True support:  [{k1}, {k2}]")
print(f"Recovered support: {support}")
print(f"Reconstruction RMSE: {recon_rmse:.3e}")

# --- 6) Plots -----------------------------------------------------------------
# Time-domain (zoom in for visibility)
plt.figure()
T = np.arange(n)
zoom = slice(0, 256)
plt.plot(T[zoom], x[zoom], label="original")
plt.plot(T[zoom], x_hat[zoom], label="reconstruction", linestyle="--")
mask = (idx >= 0) & (idx < 256)
plt.scatter(idx[mask], y[mask], marker="x", label="measured points")
plt.title(f"Compressed sensing reconstruction (time)  RMSE={recon_rmse:.3e}")
plt.xlabel("time index")
plt.ylabel("signal value")
plt.legend()

# True vs recovered coefficient magnitudes
plt.figure()
plt.stem(np.arange(n), np.abs(s_true))
plt.title("True sparse coefficients |s_true| (DCT basis)")
plt.xlabel("DCT index k")
plt.ylabel("|s_k|")

plt.figure()
plt.stem(np.arange(n), np.abs(s_hat))
plt.title(f"Recovered coefficients |s_hat| (support={support})")
plt.xlabel("DCT index k")
plt.ylabel("|s_k|")

plt.show()
