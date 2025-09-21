import numpy as np
import matplotlib.pyplot as plt

# --- setup ---
fs, fin, N = 20_000, 437.5, 8000_000
t = np.arange(N)/fs
x_in  = np.sign(np.sin(2*np.pi*fin*t))  # square input

# --- PLL state ---
f_ctl = 410.0     # start freq guess
phi   = 0.0
f_log = np.zeros(N)
y_out = np.zeros(N)

kp = 5e4          # proportional gain (tune this!)

last_in_edge  = 0.0
last_nco_edge = 0.0
err = 0.0

x_prev = x_in[0]
y_prev = -1.0

for n in range(N):
    # NCO generate square
    y = 1.0 if (phi % (2*np.pi)) < np.pi else -1.0
    y_out[n] = y

    # detect rising edges, update error (in seconds)
    if x_prev < 0 and x_in[n] > 0:
        last_in_edge = n / fs
        err = last_in_edge - last_nco_edge
    if y_prev < 0 and y > 0:
        last_nco_edge = n / fs
        err = last_in_edge - last_nco_edge

    # frequency control update
    f_ctl += (kp * err) / fs
    f_ctl = max(1.0, min(f_ctl, fs/2 - 1.0))  # clamp to valid range
    f_log[n] = f_ctl

    # advance NCO phase
    phi += 2*np.pi * f_ctl / fs

    x_prev = x_in[n]
    y_prev = y

print(f"final estimated frequency: {f_ctl:.2f} Hz (true {fin} Hz)")

# --- plots inline ---
# input vs PLL output at the end (lock check)
win_end = 1500
plt.figure()
plt.title("Input vs PLL Output (end, time domain)")
plt.plot(t[-win_end:], x_in[-win_end:], label="Input square")
plt.plot(t[-win_end:], y_out[-win_end:], label="PLL NCO output")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# frequency convergence
plt.figure()
plt.title("PLL Frequency Estimate")
plt.plot(t, f_log, label="Estimate")
plt.axhline(fin, linestyle="--", color="r", label="True frequency")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.legend()
plt.grid(True)

plt.show()
