import numpy as np
import control 
import click
import matplotlib.pyplot as plt

plant_A = np.array([[-1.0, -0.2],
                    [ -0.2, -2.0]])
plant_B = np.eye(2)
plant_C = np.eye(2)
plant_D = np.zeros((2,2))

# define e = r - y
"""
e = r - y
[x_dot, z_dot].T = A_aug @ [x, z].T + B_aug @ u + E @ r

Where z = integral(e) = integral(r - y)
therefore z_dot = e = r - y

So E = [0, I].T, and the subtraction of "y" comes from A_aug matrix by doing -C on the x state part and -D on the u part
"""

# augment with integral action
A_augmented = np.block([[plant_A, np.zeros((2,2))],
                        [-plant_C, np.zeros((2,2))]])
B_augmented = np.block([[plant_B],
                        [-plant_D]])
E = np.block([[np.zeros((2,2))],
              [np.eye(2)]])

Q = np.diag([1.0, 1.0, 10.0, 1.0]) # state cost [2], integral cost [2]
R = np.diag([0.01, 0.01])          # input cost

K, _, _ = control.lqr(A_augmented, B_augmented, Q, R)
click.secho(f"K = {K}", fg="yellow")

# --- FEEDFORWARD: compute Kr (general MIMO) -------------------------------
n = plant_A.shape[0]
p = plant_C.shape[0]

Kx = K[:, :n]                                 # <<< NEW (split augmented gain)

ABCD = np.block([[plant_A, plant_B],          # <<< NEW
                 [plant_C, plant_D]])         # <<< NEW
rhs  = np.block([[np.zeros((n, p))],          # <<< NEW
                 [np.eye(p)]])                # <<< NEW

sol  = np.linalg.lstsq(ABCD, rhs, rcond=None)[0]   # <<< NEW
Nx   = sol[:n, :]                                   # <<< NEW
Nu   = sol[n:, :]                                   # <<< NEW
Kr   = Nu + Kx @ Nx                                 # <<< NEW
# -------------------------------------------------------------------------

def control_law(x, r):
    u = -K @ x + Kr @ r 
    return u

def propogate_dynamics(x, u, r, dt):
    x_full_dot = A_augmented @ x + B_augmented @ u + E @ r
    x_next = x + x_full_dot * dt
    return x_next

dt = 0.001
T = 10.0
t = np.arange(0, T, dt)

x = np.array([10.0, -5.0, 0.0, 0.0]) # initial state
r = np.array([-50.0, 50.0])              # reference input

x_log = np.zeros((len(t), len(x)))
u_log = np.zeros((len(t), 2))

for i in range(len(t)):
    u = control_law(x, r)
    x_log[i, :] = x
    u_log[i, :] = u
    x = propogate_dynamics(x, u, r, dt)

def plot_state(t, x_log, r):
    plt.figure()
    plt.title("State vs Time")
    plt.plot(t, x_log[:, 0], label="x1")
    plt.plot(t, x_log[:, 1], label="x2")
    plt.plot(t, r[0] * np.ones_like(t), label="r1", linestyle="--")
    plt.plot(t, r[1] * np.ones_like(t), label="r2", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("State value")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_state(t, x_log, r)