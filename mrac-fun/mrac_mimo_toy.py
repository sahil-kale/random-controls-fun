import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T = 10                # Total simulation time (seconds)
dt = 0.001            # Time step
N = int(T / dt)       # Number of time steps
t = np.linspace(0, T, N)

# Reference input (step)
r = np.ones((2, N))  # 2D reference input

# Initialize state vectors (2D systems)
x_p = np.zeros((2, N))
x_r = np.zeros((2, N))
e = np.zeros((2, N))
u = np.zeros((2, N))

# Plant matrices
A_p = np.array([[-1, -6],
                [-10, -2]])
B_p = np.eye(2)

# Reference model matrices
A_r = np.array([[-3, 0],
                [0, -5]])
B_r = np.array([[3, 0],
                [0, 5]])

# Adaptive gains (diagonal)
gamma_r = 5.0
gamma_xp = 5.0

# Adaptive parameters: theta_r and theta_xp are 2x2 matrices
theta_r = np.zeros((2, 2))
theta_xp = np.zeros((2, 2))

# History for plotting
theta_r_hist = np.zeros((2, 2, N))
theta_xp_hist = np.zeros((2, 2, N))

# Simulation loop
for i in range(N - 1):
    # Control input
    u[:, i] = theta_r @ r[:, i] + theta_xp @ x_p[:, i]

    # Plant dynamics
    x_p[:, i+1] = x_p[:, i] + dt * (A_p @ x_p[:, i] + B_p @ u[:, i])

    # Reference model dynamics
    x_r[:, i+1] = x_r[:, i] + dt * (A_r @ x_r[:, i] + B_r @ r[:, i])

    # Tracking error
    e[:, i] = x_p[:, i] - x_r[:, i]

    # Adaptive update (Lyapunov rule)
    theta_r += -gamma_r * dt * np.outer(e[:, i], r[:, i])
    theta_xp += -gamma_xp * dt * np.outer(e[:, i], x_p[:, i])

    # Log adaptive parameters
    theta_r_hist[:, :, i+1] = theta_r
    theta_xp_hist[:, :, i+1] = theta_xp

# Final input and error update
u[:, -1] = theta_r @ r[:, -1] + theta_xp @ x_p[:, -1]
e[:, -1] = x_p[:, -1] - x_r[:, -1]

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(t, x_r[0], label='$x_{r1}$')
axs[0].plot(t, x_p[0], '--', label='$x_{p1}$')
axs[0].set_ylabel('State 1')
axs[0].legend()
axs[0].set_title('MRAC with Lyapunov Rule (2-State MIMO System)')

axs[1].plot(t, x_r[1], label='$x_{r2}$')
axs[1].plot(t, x_p[1], '--', label='$x_{p2}$')
axs[1].set_ylabel('State 2')
axs[1].legend()

axs[2].plot(t, theta_r_hist[0, 0], label='$\\theta_{r11}$')
axs[2].plot(t, theta_r_hist[0, 1], label='$\\theta_{r12}$')
axs[2].plot(t, theta_r_hist[1, 0], label='$\\theta_{r21}$')
axs[2].plot(t, theta_r_hist[1, 1], label='$\\theta_{r22}$')
axs[2].set_ylabel('$\\theta_r$ entries')
axs[2].set_xlabel('Time [s]')
axs[2].legend()

plt.tight_layout()
plt.show()
