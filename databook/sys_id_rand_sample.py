# Chapter 3.4 + 3.5 toy: identify (m, c, k) from undersampled measurements
# - System: m xddot + c xdot + k x = u(t)
# - "Random" input: sum of a few random sinusoids (sparse in frequency)
# - Step A (simulate): generate full-resolution x(t), v(t) given u(t)
# - Step B (3.4 demo): take p << n random samples of x(t); reconstruct x via CS (DCT + OMP)
# - Step C (3.5 demo): from reconstructed x, estimate xdot, xddot; sparse regression for dynamics
#                      xddot = a*x + b*xdot + d*u  => infer m=1/d, c=-b/d, k=-a/d
#
# Plain NumPy & matrix ops, matplotlib for simple plots.
# No seaborn; one chart per figure; use default colors.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------- Utilities --------------------------------------------
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n x n)."""
    N = n
    k = np.arange(N)[:, None]
    n_idx = np.arange(N)[None, :]
    mat = np.cos(np.pi * (2*n_idx + 1) * k / (2*N))
    alpha = np.sqrt(2/N) * np.ones((N, 1))
    alpha[0, 0] = np.sqrt(1/N)
    return alpha @ np.ones((1, N)) * mat

def omp(Theta, y, K, tol=1e-9):
    """Orthogonal Matching Pursuit: solve y ≈ Theta @ s with K-sparse s."""
    p, n = Theta.shape
    residual = y.copy()
    support = []
    s_hat = np.zeros(n)
    for _ in range(K):
        corr = Theta.T @ residual
        # pick the highest-magnitude correlation not yet chosen
        for pick in np.argsort(np.abs(corr))[::-1]:
            if pick not in support:
                j = pick
                break
        support.append(j)
        Th_S = Theta[:, support]
        s_S, *_ = np.linalg.lstsq(Th_S, y, rcond=None)
        residual = y - Th_S @ s_S
        if np.linalg.norm(residual) <= tol:
            break
    s_hat[support] = s_S
    return s_hat, support

def central_diff(x, dt):
    """First derivative via central differences (second-order, ends with one-sided)."""
    n = x.size
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx

# ----------------------- Step A: Simulate -------------------------------------
# True parameters
m_true = 1.5
c_true = 0.8
k_true = 20.0

# Time grid
n = 4096
T = 8.0
dt = T / n
t = np.linspace(0, T, n, endpoint=False)

# "Random" sparse-frequency input: sum of a few random sinusoids
num_tones = 3
freqs = rng.uniform(0.3, 3.0, size=num_tones)   # Hz
amps  = rng.uniform(0.5, 2.0, size=num_tones)
phs   = rng.uniform(0, 2*np.pi, size=num_tones)
u = np.zeros(n)
for A, f, p in zip(amps, freqs, phs):
    u += A * np.cos(2*np.pi*f*t + p)

# Integrate m x'' + c x' + k x = u with RK4
x = np.zeros(n)
v = np.zeros(n)
def f_state(xv, ui):
    x_, v_ = xv
    a_ = (ui - c_true * v_ - k_true * x_) / m_true
    return np.array([v_, a_])
xv = np.array([0.0, 0.0])
for i in range(n-1):
    ui = u[i]
    k1 = f_state(xv, ui)
    k2 = f_state(xv + 0.5*dt*k1, ui)
    k3 = f_state(xv + 0.5*dt*k2, ui)
    k4 = f_state(xv + dt*k3, ui)
    xv = xv + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    x[i+1], v[i+1] = xv

# ----------------------- Step B: CS reconstruction (3.4) ----------------------
# Take p random samples of x(t)
p = 700  # much less than n (undersampling)
idx = np.sort(rng.choice(n, size=p, replace=False))
y_meas = x[idx]

# Build DCT dictionary and sensing matrix
Psi = dct_matrix(n)            # n x n (orthonormal-ish)
Theta = Psi[idx, :]            # p x n (select rows)

# Recover sparse DCT coefficients via OMP
K_guess = 60  # allow some richness due to dynamics + multi-tone input
s_hat, support = omp(Theta, y_meas, K=K_guess, tol=1e-10)
x_cs = Psi @ s_hat

# ----------------------- Step C: Sparse regression (3.5) ----------------------
# Estimate derivatives from reconstructed x
xdot_cs  = central_diff(x_cs, dt)
xddot_cs = central_diff(xdot_cs, dt)

# Build regression: xddot = a*x + b*xdot + d*u
Phi = np.column_stack([x_cs, xdot_cs, u])  # n x 3
# Solve least squares
coef, *_ = np.linalg.lstsq(Phi, xddot_cs, rcond=None)
a_hat, b_hat, d_hat = coef

# Infer physical parameters
m_hat = 1.0 / d_hat
c_hat = -b_hat / d_hat
k_hat = -a_hat / d_hat

# ----------------------- Diagnostics & Plots ----------------------------------
rmse_recon = np.sqrt(np.mean((x - x_cs)**2))
rel_err = lambda est, tru: abs(est - tru) / (abs(tru) + 1e-12)

print("=== Simulation & Identification Summary ===")
print(f"n={n}, p={p}, dt={dt:.4e}, T={T:.2f}s, tones={num_tones}")
print(f"Input freqs (Hz): {np.round(freqs, 3)}")
print(f"OMP support size: {len(np.flatnonzero(s_hat))} (target K_guess={K_guess})")
print(f"Time-series reconstruction RMSE: {rmse_recon:.3e}")
print("\nTrue params vs. identified (from CS + sparse regression):")
print(f"m_true={m_true:.4f}  |  m_hat={m_hat:.4f}  (rel err={rel_err(m_hat,m_true):.2%})")
print(f"c_true={c_true:.4f}  |  c_hat={c_hat:.4f}  (rel err={rel_err(c_hat,c_true):.2%})")
print(f"k_true={k_true:.4f}  |  k_hat={k_hat:.4f}  (rel err={rel_err(k_hat,k_true):.2%})")

# Plot: x vs reconstructed x_cs (zoom for visibility)
plt.figure()
zoom = slice(0, 1200)
plt.plot(t[zoom], x[zoom], label="x (true)")
plt.plot(t[zoom], x_cs[zoom], linestyle="--", label="x (CS recon)")
plt.scatter(t[idx[(idx>=zoom.start)&(idx<zoom.stop)]], y_meas[(idx>=zoom.start)&(idx<zoom.stop)], marker="x", label="measured points", s=12)
plt.title(f"Compressed sensing reconstruction (time)  RMSE={rmse_recon:.2e}")
plt.xlabel("time (s)")
plt.ylabel("x")
plt.legend()

# Plot: parameter identification bar chart
plt.figure()
names = ["m", "c", "k"]
true_vals = np.array([m_true, c_true, k_true])
hat_vals  = np.array([m_hat, c_hat, k_hat])
xpos = np.arange(3)
width = 0.35
plt.bar(xpos - width/2, true_vals, width, label="true")
plt.bar(xpos + width/2, hat_vals,  width, label="identified")
plt.xticks(xpos, names)
plt.title("Parameter identification (from CS recon + sparse regression)")
plt.ylabel("value")
plt.legend()

plt.show()
