import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T = 20                # Total simulation time (seconds)
dt = 0.01             # Time step
N = int(T / dt)       # Number of time steps
t = np.linspace(0, T, N)

# Reference model: ẋm = -a_m * x_m + b_m * r
a_m = 2.0
b_m = 2.0

# Plant: ẋp = -a_p * x_p + b_p * u, where a_p = 1, b_p = 1 (1st order: 1/(s+1))
a_p = 1.0
b_p = 1.0

# MIT rule learning rate
gamma = 5.0

# Initial conditions
x_m = np.zeros(N)      # Reference model state
x_p = np.zeros(N)      # Plant state
theta = 0.5            # Initial adaptive parameter
r = np.ones(N)         # Step input

# Adaptive control history
u = np.zeros(N)
theta_hist = np.zeros(N)
error = np.zeros(N)

# Simulation loop
for i in range(N - 1):
    # Reference model
    x_m[i+1] = x_m[i] + dt * (-a_m * x_m[i] + b_m * r[i])

    # Control input using current theta estimate
    u[i] = theta * r[i]

    # Plant dynamics
    x_p[i+1] = x_p[i] + dt * (-a_p * x_p[i] + b_p * u[i])

    # Error
    error[i] = x_p[i] - x_m[i]

    # MIT rule: update theta
    dtheta = -gamma * error[i] * r[i]
    theta = theta + dt * dtheta
    theta_hist[i+1] = theta

# Final input and error update
u[-1] = theta * r[-1]
error[-1] = x_p[-1] - x_m[-1]

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x_m, label='Reference Model $x_m$')
plt.plot(t, x_p, label='Plant Output $x_p$', linestyle='--')
plt.title('MRAC with MIT Rule (1st-Order Plant)')
plt.ylabel('State')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, theta_hist, label='Adaptive Gain $\\theta$')
plt.xlabel('Time [s]')
plt.ylabel('$\\theta$')
plt.legend()
plt.tight_layout()
plt.show()
