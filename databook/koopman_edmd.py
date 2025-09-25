import numpy as np

# ---------- 1) Define system & simulator ----------
mu  = -0.05
lam = -1.0        # IMPORTANT: negative (fast stable)
dt  = 0.01
T   = 30.0
N   = int(T/dt)

def f(x):
    x1, x2 = x
    return np.array([mu*x1, lam*(x2 - x1**2)])

def step(x):
    return x + dt*f(x)   # forward Euler

# ---------- 2) Generate data pairs (x_k, x_{k+1}) ----------
np.random.seed(0)
# multiple short trajectories to decorrelate data
n_traj   = 20
traj_len = 200
Xs = []
Ys = []
for _ in range(n_traj):
    x = np.array([np.random.uniform(-1.5, 1.5),
                  np.random.uniform(-1.0,  1.0)])
    for k in range(traj_len):
        x_next = step(x)
        Xs.append(x)
        Ys.append(x_next)
        x = x_next.copy()

Xs = np.array(Xs)        # shape (M,2)
Ys = np.array(Ys)        # shape (M,2)
M  = Xs.shape[0]

# ---------- 3) Build dictionary (observables) ----------
def psi(x):
    x1, x2 = x[...,0], x[...,1]
    return np.stack([x1, x2, x1**2], axis=-1)   # shape (...,3)

Psi_X = psi(Xs).T            # shape (p=3, M)
Psi_Y = psi(Ys).T            # shape (p=3, M)

# ---------- 4) EDMD least-squares: K = Psi_Y * Psi_X^+ ----------
# Use SVD pseudo-inverse for numerical stability
U, S, Vt = np.linalg.svd(Psi_X, full_matrices=False)
Psi_X_pinv = Vt.T @ np.diag(1.0/S) @ U.T
K = Psi_Y @ Psi_X_pinv        # shape (3,3)

breakpoint()

# ---------- 5) Recover continuous-time A_hat ≈ (K - I)/dt ----------
A_hat = (K - np.eye(3))/dt

print("K (discrete, lifted):\n", K)
print("\nA_hat (continuous, lifted):\n", A_hat)

# ---------- 6) Compare A_hat to the known 'A' from the analytic lift ----------
A_true = np.array([[mu,   0.0,     0.0],
                   [0.0,  lam,    -lam],
                   [0.0,  0.0,   2*mu]])
print("\nA_true:\n", A_true)
print("\nA_hat - A_true:\n", A_hat - A_true)
print("\n||A_hat - A_true||_F =", np.linalg.norm(A_hat - A_true, 'fro'))

# ---------- 7) Validate prediction on a fresh IC ----------
x0 = np.array([1.2, -0.3])
psi_k = psi(x0[np.newaxis, :]).ravel()  # shape (3,)

# one-step prediction via K in lifted space
psi_k1_pred = K @ psi_k
# ground truth next state via simulator, then lift
x1_true = step(x0)
psi_k1_true = psi(x1_true[np.newaxis, :]).ravel()

print("\nOne-step lifted prediction error (L2):",
      np.linalg.norm(psi_k1_pred - psi_k1_true))

# also check that first two entries track x1,x2 (reconstruction)
print("Pred x1,x2 vs. true:",
      psi_k1_pred[:2], " | ", x1_true)
