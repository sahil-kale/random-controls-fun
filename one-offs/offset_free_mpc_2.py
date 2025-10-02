# Re-execute the full cell (stateful env was reset).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

np.random.seed(1)

# ------------------------- Problem setup -------------------------
a_p, b_p, d_p = 0.92, 0.12, 0.01
A_hat, B_hat = 0.85, 0.10
C = 1.0

u_min, u_max = -1.0, 2.0
N = 20
Qy, Ru = 10.0, 0.05
T = 90
r = 1.0
x0 = 0.0
process_noise_std = 0.0

# KF for [x; d]
F = np.array([[A_hat, 1.0],
              [0.0  , 1.0]])
G = np.array([[B_hat],
              [0.0  ]])
H = np.array([[1.0, 0.0]])
Q_kf = np.diag([1e-4, 5e-4])
R_kf = np.array([[1e-4]])
xhat0 = np.array([0.0, 0.0])
P0 = np.diag([0.5, 0.5])

def clamp(u, lo, hi):
    return np.minimum(np.maximum(u, lo), hi)

def predict_traj_vanilla(A, B, x0, u_seq):
    x = x0
    ys = []
    for uk in u_seq:
        x = A*x + B*uk
        ys.append(C*x)
    return np.array(ys)

def predict_traj_with_d(A, B, x0, d_hat, u_seq):
    x = x0
    ys = []
    for uk in u_seq:
        x = A*x + B*uk + d_hat
        ys.append(C*x)
    return np.array(ys)

def cost_vanilla(u_seq, x_now, r, A, B):
    u_seq = clamp(u_seq, u_min, u_max)
    y_pred = predict_traj_vanilla(A,B,x_now,u_seq)
    e = y_pred - r
    return Qy*np.sum(e**2) + Ru*np.sum(u_seq**2)

def cost_offset_free(u_seq, x_now, d_hat, r, A, B):
    u_seq = clamp(u_seq, u_min, u_max)
    y_pred = predict_traj_with_d(A,B,x_now,d_hat,u_seq)
    e = y_pred - r
    return Qy*np.sum(e**2) + Ru*np.sum(u_seq**2)

def solve_mpc(cost_fn, x_now, init_u, bounds, args=()):
    res = minimize(
        lambda u: cost_fn(u, x_now, *args),
        x0=init_u,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-6}
    )
    return clamp(res.x, u_min, u_max)

def kf_step(xhat, P, u_prev, y_meas):
    xhat_pred = F @ xhat + G.flatten()*u_prev
    P_pred = F @ P @ F.T + Q_kf
    S = H @ P_pred @ H.T + R_kf
    K = (P_pred @ H.T) @ np.linalg.inv(S)
    z = y_meas - (H @ xhat_pred)[0]
    xhat_new = xhat_pred + (K.flatten()*z)
    P_new = (np.eye(2) - K @ H) @ P_pred
    return xhat_new, P_new

def run_simulations():
    bounds = [(u_min, u_max)]*N

    # Vanilla
    x = x0
    u_plan = np.zeros(N)
    y_hist_v, u_hist_v = [], []
    y_pred_hist_v, u_pred_hist_v = [], []
    for t in range(T):
        u_opt = solve_mpc(cost_vanilla, x, u_plan, bounds, args=(r, A_hat, B_hat))
        y_pred = predict_traj_vanilla(A_hat, B_hat, x, u_opt)

        u_apply = u_opt[0]
        u_plan = np.r_[u_opt[1:], u_opt[-1]]

        w = np.random.randn()*process_noise_std
        x = a_p*x + b_p*u_apply + d_p + w
        y = C*x

        y_hist_v.append(y)
        u_hist_v.append(u_apply)
        y_pred_hist_v.append(y_pred.copy())
        u_pred_hist_v.append(u_opt.copy())

    sim_v = {"y": np.array(y_hist_v), "u": np.array(u_hist_v),
             "y_pred": np.array(y_pred_hist_v), "u_pred": np.array(u_pred_hist_v)}

    # Offset-free with KF
    x = x0
    xhat = xhat0.copy()
    P = P0.copy()
    u_prev = 0.0
    u_plan = np.zeros(N)
    y_hist_o, u_hist_o = [], []
    y_pred_hist_o, u_pred_hist_o = [], []
    d_hat_hist = []

    for t in range(T):
        y_meas = C*x
        xhat, P = kf_step(xhat, P, u_prev, y_meas)
        xhat_x, xhat_d = xhat[0], xhat[1]
        d_hat_hist.append(xhat_d)

        u_opt = solve_mpc(cost_offset_free, xhat_x, u_plan, bounds, args=(xhat_d, r, A_hat, B_hat))
        y_pred = predict_traj_with_d(A_hat, B_hat, xhat_x, xhat_d, u_opt)

        u_apply = u_opt[0]
        u_plan = np.r_[u_opt[1:], u_opt[-1]]

        w = np.random.randn()*process_noise_std
        x = a_p*x + b_p*u_apply + d_p + w
        y = C*x

        y_hist_o.append(y)
        u_hist_o.append(u_apply)
        y_pred_hist_o.append(y_pred.copy())
        u_pred_hist_o.append(u_opt.copy())

        u_prev = u_apply

    sim_o = {"y": np.array(y_hist_o), "u": np.array(u_hist_o),
             "y_pred": np.array(y_pred_hist_o), "u_pred": np.array(u_pred_hist_o),
             "d_hat": np.array(d_hat_hist)}
    return sim_v, sim_o

sim_v, sim_o = run_simulations()

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
ax_y_v, ax_u_v, ax_y_o, ax_u_o = axes.ravel()

for ax in [ax_y_v, ax_u_v, ax_y_o, ax_u_o]:
    ax.grid(True, alpha=0.3)

ax_y_v.set_title("Vanilla MPC (model mismatch)")
ax_y_o.set_title("Offset-free MPC (KF-estimated disturbance)")

ax_u_v.set_xlabel("time step k")
ax_u_o.set_xlabel("time step k")
ax_y_v.set_ylabel("y")
ax_u_v.set_ylabel("u")
ax_y_o.set_ylabel("y")
ax_u_o.set_ylabel("u")

(y_line_v,) = ax_y_v.plot([], [], lw=2)
(u_line_v,) = ax_u_v.plot([], [], lw=2)
(y_line_o,) = ax_y_o.plot([], [], lw=2)
(u_line_o,) = ax_u_o.plot([], [], lw=2)

ax_y_v.axhline(r, linestyle="--")
ax_y_o.axhline(r, linestyle="--")

(y_pred_line_v,) = ax_y_v.plot([], [], lw=1)
(u_pred_line_v,) = ax_u_v.plot([], [], lw=1)
(y_pred_line_o,) = ax_y_o.plot([], [], lw=1)
(u_pred_line_o,) = ax_u_o.plot([], [], lw=1)

ax_y_v.set_xlim(0, T)
ax_y_o.set_xlim(0, T)
ax_u_v.set_xlim(0, T)
ax_u_o.set_xlim(0, T)

ax_y_v.set_ylim(-0.2, 1.6)
ax_y_o.set_ylim(-0.2, 1.6)
ax_u_v.set_ylim(u_min - 0.1, u_max + 0.1)
ax_u_o.set_ylim(u_min - 0.1, u_max + 0.1)

y_hist_v, u_hist_v = [], []
y_hist_o, u_hist_o = [], []

def animate(k):
    y_hist_v.append(sim_v["y"][k])
    u_hist_v.append(sim_v["u"][k])
    y_hist_o.append(sim_o["y"][k])
    u_hist_o.append(sim_o["u"][k])

    y_line_v.set_data(np.arange(len(y_hist_v)), y_hist_v)
    u_line_v.set_data(np.arange(len(u_hist_v)), u_hist_v)
    y_line_o.set_data(np.arange(len(y_hist_o)), y_hist_o)
    u_line_o.set_data(np.arange(len(u_hist_o)), u_hist_o)

    k_grid = np.arange(k, k + N)
    if k_grid[-1] >= T + N:
        k_grid = np.arange(k, min(k + N, T + N - 1))

    y_pred_line_v.set_data(k_grid, sim_v["y_pred"][k])
    u_pred_line_v.set_data(k_grid, sim_v["u_pred"][k])
    y_pred_line_o.set_data(k_grid, sim_o["y_pred"][k])
    u_pred_line_o.set_data(k_grid, sim_o["u_pred"][k])

    return (y_line_v, u_line_v, y_line_o, u_line_o,
            y_pred_line_v, u_pred_line_v, y_pred_line_o, u_pred_line_o)

ani = FuncAnimation(fig, animate, frames=T, interval=60, blit=True)
plt.tight_layout()
plt.show()
