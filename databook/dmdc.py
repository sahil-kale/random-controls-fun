import numpy as np

# ----- ground truth -----
A_true = np.array([[1.5, 0.0],
                   [0.0, 0.1]])           # unstable (eigs 1.5, 0.1)
B_true = np.array([[1.0],
                   [0.0]])
K = np.array([[-1.0, 0.0]])               # stabilizing state feedback u_k = K x_k

# ----- collect snapshots under closed-loop operation -----
m = 100                      # number of snapshots
x = np.zeros((2, m))
u = np.zeros((1, m-1))

x[:, 0] = np.array([4.0, 7.0])            # initial condition
for k in range(m-1):
    u[:, k] = (K @ x[:, k])[None, :]      # 1x1
    x[:, k+1] = (A_true @ x[:, k] + B_true @ u[:, k]).ravel()

# build data matrices
X  = x[:, :m-1]             # 2 x (m-1)
Xp = x[:, 1:m]              # 2 x (m-1)
Upsilon = u                 # 1 x (m-1)

# ----- DMDc with known B -----
# SVD of X
U, S, Vt = np.linalg.svd(X, full_matrices=False)       # X = U diag(S) Vt
# choose rank r (here full rank = 2)
r = 2
Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = Vt[:r, :].T

# A_hat = (X' - B Upsilon) Vr Sr^{-1} Ur^T
A_hat = (Xp - B_true @ Upsilon) @ Vr @ np.linalg.inv(Sr) @ Ur.T

print("A_true =\n", A_true)
print("A_hat  =\n", A_hat)
print("||A_hat - A_true||_F =", np.linalg.norm(A_hat - A_true, 'fro'))
