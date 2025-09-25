import numpy as np

# parameters: fast x2, slow x1
mu = -0.05
lam = -1.0

# time grid
dt = 0.01
T  = 15.0
N  = int(T/dt)

# storage
x  = np.zeros((N+1, 2))      # [x1, x2]
y  = np.zeros((N+1, 3))      # [y1, y2, y3] = [x1, x2, x1^2]

# initial conditions
x[0] = np.array([1.0, -0.2])
y[0] = np.array([x[0,0], x[0,1], x[0,0]**2])

# lifted linear system matrix A (Koopman-restricted)
A = np.array([[mu,   0.0,     0.0],
              [0.0,  lam,    -lam],
              [0.0,  0.0,   2*mu]])

# simulate
for k in range(N):
    # --- original nonlinear system ---
    x1, x2 = x[k]
    dx1 = mu*x1
    dx2 = lam*(x2 - x1**2)
    x[k+1] = x[k] + dt*np.array([dx1, dx2])

    # --- lifted linear system (Euler on dy/dt = A y) ---
    y[k+1] = y[k] + dt*(A @ y[k])

# check agreement (reconstruction)
x1_from_y = y[:,0]
x2_from_y = y[:,1]
err = np.max(np.abs(x[:,0]-x1_from_y)) + np.max(np.abs(x[:,1]-x2_from_y))
print(f"max abs reconstruction error in x1,x2 (Euler/Euler): {err:.3e}")

# Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x[:,0], x[:,1], 'k', label='nonlinear')
plt.plot(x1_from_y, x2_from_y, 'r--', label='lifted linear')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Phase plane')
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.arange(N+1)*dt, x[:,0], 'k', label='x1 nonlinear')
plt.plot(np.arange(N+1)*dt, x[:,1], 'k--', label='x2 nonlinear')
plt.plot(np.arange(N+1)*dt, x1_from_y, 'r', label='x1 lifted linear')
plt.plot(np.arange(N+1)*dt, x2_from_y, 'r--', label='x2 lifted linear')
plt.xlabel('time')
plt.ylabel('states')
plt.title('Time series')
plt.legend()
plt.tight_layout()
plt.show()