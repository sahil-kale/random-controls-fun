import numpy as np
import matplotlib.pyplot as plt

def simulate_explicit_euler(x0, v0, w, dt, T):
    N = int(T / dt)
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    x[0], v[0] = x0, v0

    for k in range(N):
        # explicit Euler: use old state for derivatives
        x[k+1] = x[k] + dt * v[k]
        v[k+1] = v[k] + dt * (-w*w * x[k])

    return x, v

def simulate_symplectic_euler(x0, v0, w, dt, T):
    N = int(T / dt)
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    x[0], v[0] = x0, v0

    for k in range(N):
        # symplectic Euler (velocity-first):
        v[k+1] = v[k] + dt * (-w*w * x[k])
        x[k+1] = x[k] + dt * v[k+1]

    return x, v

def energy(x, v, w):
    return 0.5 * v*v + 0.5 * (w*w) * x*x

def main():
    # system params
    w = 1.0            # natural frequency rad/s
    x0, v0 = 1.0, 0.0  # start at max displacement

    # integration params
    dt = 0.05           # try 0.05, 0.1, 0.2, 0.3 to see effect
    T  = 80.0

    t = np.arange(0.0, T + dt, dt)

    x_e, v_e = simulate_explicit_euler(x0, v0, w, dt, T)
    x_s, v_s = simulate_symplectic_euler(x0, v0, w, dt, T)

    E_e = energy(x_e, v_e, w)
    E_s = energy(x_s, v_s, w)

    # exact solution for reference (optional, but nice)
    x_exact = x0*np.cos(w*t) + (v0/w)*np.sin(w*t)
    v_exact = -x0*w*np.sin(w*t) + v0*np.cos(w*t)
    E_exact = energy(x_exact, v_exact, w)

    # ---- Plots ----
    plt.figure()
    plt.plot(t, x_exact, label="Exact x(t)")
    plt.plot(t, x_e, label="Explicit Euler x(t)")
    plt.plot(t, x_s, label="Symplectic Euler x(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("x")
    plt.title(f"Simple Harmonic Oscillator (dt={dt})")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t, E_exact, label="Exact energy")
    plt.plot(t, E_e, label="Explicit Euler energy")
    plt.plot(t, E_s, label="Symplectic Euler energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy")
    plt.title("Energy Drift Comparison")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
