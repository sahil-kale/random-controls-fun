# Cleaned-up, discrete-time, offset-free MPC demo with disturbance estimator (KF)
# - Pretty plot: three stacked panels (y, u, disturbance) with "Past: Estimation" vs "Future: MPC" shading
# - Discrete-time 1st-order plant; controller runs naive MPC with horizon N over inputs
# - Offset-free via augmented-state KF that estimates a constant disturbance d
#
# Libraries: numpy, scipy, matplotlib only

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import blended_transform_factory
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

np.random.seed(1)

# ------------------------- Problem setup -------------------------
# True (mismatched) plant: x[k+1] = a_p*x[k] + b_p*u[k] + d_p ; y = x
a_p, b_p, d_p = 0.92, 0.12, 0.01

# Controller's nominal model (used in MPC & KF), missing disturbance explicitly
A_hat, B_hat = 0.85, 0.10
C = 1.0

# MPC weights/limits
u_min, u_max = -1.0, 2.0
N = 20
Qy, Ru = 10.0, 0.05

# Simulation
T = 100
r = 1.0
x0 = 0.0
process_noise_std = 0.0

# KF (augmented [x, d] with random-walk disturbance)
F = np.array([[A_hat, 1.0],
              [0.0  , 1.0]])
G = np.array([[B_hat],
              [0.0  ]])
H = np.array([[1.0, 0.0]])
Q_kf = np.diag([1e-4, 5e-4])   # allow slow drift on d
R_kf = np.array([[1e-4]])
xhat0 = np.array([0.0, 0.0])   # [x_hat, d_hat]
P0 = np.diag([0.5, 0.5])

# ------------------------- Helpers -------------------------
def clamp(u, lo, hi):
    return np.minimum(np.maximum(u, lo), hi)

def predict_traj_with_d(A, B, x0, d_hat, u_seq):
    x = x0
    ys = []
    for uk in u_seq:
        x = A*x + B*uk + d_hat
        ys.append(C*x)
    return np.array(ys)

def predict_traj_nominal(A, B, x0, u_seq):
    x = x0
    ys = []
    for uk in u_seq:
        x = A*x + B*uk
        ys.append(C*x)
    return np.array(ys)

def cost_vanilla(u_seq, x_now, r, A, B):
    u_seq = clamp(u_seq, u_min, u_max)
    y_pred = predict_traj_nominal(A, B, x_now, u_seq)
    e = y_pred - r
    return Qy*np.sum(e**2) + Ru*np.sum(u_seq**2)

def cost_offsetfree(u_seq, x_now, d_hat, r, A, B):
    u_seq = clamp(u_seq, u_min, u_max)
    y_pred = predict_traj_with_d(A, B, x_now, d_hat, u_seq)
    e = y_pred - r
    return Qy*np.sum(e**2) + Ru*np.sum(u_seq**2)

def solve_mpc(cost_fn, x_now, init_u, bounds, args=()):
    res = minimize(lambda u: cost_fn(u, x_now, *args),
                   x0=init_u, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 120, "ftol": 1e-6})
    return clamp(res.x, u_min, u_max)

def kf_step(xhat, P, u_prev, y_meas):
    # Predict
    xhat_pred = F @ xhat + G.flatten()*u_prev
    P_pred = F @ P @ F.T + Q_kf
    # Update
    S = H @ P_pred @ H.T + R_kf
    K = (P_pred @ H.T) @ np.linalg.inv(S)
    innov = y_meas - (H @ xhat_pred)[0]
    xhat_new = xhat_pred + (K.flatten()*innov)
    P_new = (np.eye(2) - K @ H) @ P_pred
    return xhat_new, P_new

# ------------------------- Run simulations -------------------------
def run(vanilla=False):
    x = x0
    u_prev = 0.0
    xhat = xhat0.copy()
    P = P0.copy()

    bounds = [(u_min, u_max)]*N
    u_plan = np.zeros(N)

    y_hist, u_hist = [], []
    y_pred_hist, u_pred_hist = [], []
    d_hat_hist = []

    for k in range(T):
        y_meas = C*x

        if vanilla:
            # No estimator; assume d = 0
            d_hat = 0.0
            x_for_mpc = x  # measured x
            u_opt = solve_mpc(cost_vanilla, x_for_mpc, u_plan, bounds, args=(r, A_hat, B_hat))
            y_pred = predict_traj_nominal(A_hat, B_hat, x_for_mpc, u_opt)
        else:
            # KF update first (uses last u_prev)
            xhat, P = kf_step(xhat, P, u_prev, y_meas)
            x_for_mpc, d_hat = xhat[0], xhat[1]
            d_hat_hist.append(d_hat)
            u_opt = solve_mpc(cost_offsetfree, x_for_mpc, u_plan, bounds, args=(d_hat, r, A_hat, B_hat))
            y_pred = predict_traj_with_d(A_hat, B_hat, x_for_mpc, d_hat, u_opt)

        # Apply control (first element), shift warm start
        u_apply = u_opt[0]
        u_plan = np.r_[u_opt[1:], u_opt[-1]]
        u_prev = u_apply

        # True plant step
        w = np.random.randn()*process_noise_std
        x = a_p*x + b_p*u_apply + d_p + w
        y = C*x

        # Log
        y_hist.append(y)
        u_hist.append(u_apply)
        y_pred_hist.append(y_pred.copy())
        u_pred_hist.append(u_opt.copy())

        if vanilla is False and len(d_hat_hist) < len(y_hist):
            # Keep d_hat history aligned in length (pad first step)
            d_hat_hist = [d_hat_hist[0]] + d_hat_hist

    return {
        "y": np.array(y_hist),
        "u": np.array(u_hist),
        "y_pred": np.array(y_pred_hist),
        "u_pred": np.array(u_pred_hist),
        "d_hat": np.array(d_hat_hist) if not vanilla else None
    }

sim_v = run(vanilla=True)
sim_o = run(vanilla=False)

# ------------------------- Nice, “report-style” animation -------------------------
fig, (ax_y, ax_u, ax_d) = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for ax in (ax_y, ax_u, ax_d):
    ax.grid(True, alpha=0.3)

# Titles
ax_y.set_title("Past: Estimation  |  Future: MPC (Offset-free with KF disturbance)")

# Reference lines
ax_y.axhline(r, linestyle="--", label="Reference r")
ax_d.axhline(d_p, linestyle="--", label="True disturbance $d_p$")

# Labels
ax_y.set_ylabel("Output y")
ax_u.set_ylabel("Input u")
ax_d.set_ylabel("Disturbance d")
ax_d.set_xlabel("Time step k")

# Plot objects
(y_hist_line,) = ax_y.plot([], [], "C3", lw=2, label="y measured")
(y_pred_line,) = ax_y.plot([], [], "C1", lw=2, label="y horizon (MPC)")

(u_hist_step,) = ax_u.step([], [], "C3", where="post", lw=2, label="u history")
(u_pred_step,) = ax_u.step([], [], "C1", where="post", lw=2, label="u plan")

(d_hat_line,) = ax_d.plot([], [], "C0", lw=2, label="$\\hat d$ (KF)")

# Limits
ax_y.set_ylim(-0.2, 1.6)
ax_u.set_ylim(u_min - 0.1, u_max + 0.1)
ax_d.set_ylim(d_p - 0.3, d_p + 0.6)

# Shaded regions + current-time line
# Build a transform that uses x in DATA coords and y in AXES coords
trans_y = blended_transform_factory(ax_y.transData, ax_y.transAxes)
trans_u = blended_transform_factory(ax_u.transData, ax_u.transAxes)
trans_d = blended_transform_factory(ax_d.transData, ax_d.transAxes)

# Past (purple) and future (tan) rectangles for each subplot
past_rect_y   = patches.Rectangle((0, 0), 0, 1, transform=trans_y, color="#d5c6ea", alpha=0.5, zorder=-1)
future_rect_y = patches.Rectangle((0, 0), T, 1, transform=trans_y, color="#f3e3c7", alpha=0.5, zorder=-2)
ax_y.add_patch(future_rect_y); ax_y.add_patch(past_rect_y)

past_rect_u   = patches.Rectangle((0, 0), 0, 1, transform=trans_u, color="#d5c6ea", alpha=0.5, zorder=-1)
future_rect_u = patches.Rectangle((0, 0), T, 1, transform=trans_u, color="#f3e3c7", alpha=0.5, zorder=-2)
ax_u.add_patch(future_rect_u); ax_u.add_patch(past_rect_u)

past_rect_d   = patches.Rectangle((0, 0), 0, 1, transform=trans_d, color="#d5c6ea", alpha=0.5, zorder=-1)
future_rect_d = patches.Rectangle((0, 0), T, 1, transform=trans_d, color="#f3e3c7", alpha=0.5, zorder=-2)
ax_d.add_patch(future_rect_d); ax_d.add_patch(past_rect_d)

# Current-time vertical lines (keep your existing ones or recreate)
current_time_line   = ax_y.axvline(0, color="k", lw=1.5)
current_time_line_u = ax_u.axvline(0, color="k", lw=1.5)
current_time_line_d = ax_d.axvline(0, color="k", lw=1.5)

# Legends
ax_y.legend(loc="upper left")
ax_u.legend(loc="upper left")
ax_d.legend(loc="upper left")

ax_y.set_xlim(0, T)
ax_u.set_xlim(0, T)
ax_d.set_xlim(0, T)

def update_spans(k):
    # Past rectangles extend from x=0 to x=k
    past_rect_y.set_x(0); past_rect_y.set_width(k)
    past_rect_u.set_x(0); past_rect_u.set_width(k)
    past_rect_d.set_x(0); past_rect_d.set_width(k)

    # Future rectangles extend from x=k to x=T
    future_rect_y.set_x(k); future_rect_y.set_width(T - k)
    future_rect_u.set_x(k); future_rect_u.set_width(T - k)
    future_rect_d.set_x(k); future_rect_d.set_width(T - k)

    # Move the current-time lines
    current_time_line.set_xdata([k, k])
    current_time_line_u.set_xdata([k, k])
    current_time_line_d.set_xdata([k, k])

    # Return artists if you're using blit=True
    return (past_rect_y, future_rect_y, past_rect_u, future_rect_u,
            past_rect_d, future_rect_d,
            current_time_line, current_time_line_u, current_time_line_d)


def animate(k):
    # Histories
    y_hist = sim_o["y"][:k+1]
    u_hist = sim_o["u"][:k+1]
    d_hist = sim_o["d_hat"][:k+1]

    y_hist_line.set_data(np.arange(len(y_hist)), y_hist)
    u_hist_step.set_data(np.arange(len(u_hist)), u_hist)
    d_hat_line.set_data(np.arange(len(d_hist)), d_hist)

    # Horizon overlays (starting at k)
    k_grid = np.arange(k, min(k+N, T))
    y_pred_line.set_data(k_grid, sim_o["y_pred"][k][:len(k_grid)])
    # For discrete u plan, plot as step starting at k, "post"
    u_plan = sim_o["u_pred"][k][:len(k_grid)]
    u_pred_step.set_data(k_grid, u_plan)

    update_spans(k)
    return (y_hist_line, y_pred_line, u_hist_step, u_pred_step,
            d_hat_line, current_time_line, current_time_line_u, current_time_line_d)

ani = FuncAnimation(fig, animate, frames=T, interval=60, blit=True)
plt.tight_layout()
plt.show()
