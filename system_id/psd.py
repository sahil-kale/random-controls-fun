"""
System ID demo: 2nd-order underdamped plant, chirp excitation, Welch's method.

Plant:  G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Define the "true" plant (unknown to the identification step)
# ---------------------------------------------------------------
wn = 2 * np.pi * 15    # natural frequency [rad/s]  -> ~15 Hz
zeta = 0.05             # underdamped -> resonance peak

plant = signal.TransferFunction([wn**2], [1, 2*zeta*wn, wn**2])

# ---------------------------------------------------------------
# 2. Generate a chirp input and "record" the output
# ---------------------------------------------------------------
fs = 500.0              # sample rate [Hz]
T = 60.0                # record length [s]
t = np.arange(0, T, 1/fs)

# sweep from 0.5 Hz to 50 Hz, covering the resonance at 15 Hz
u = signal.chirp(t, f0=0.5, f1=50, t1=T, method='linear')

# simulate the true plant's response to the chirp
_, y_clean, _ = signal.lsim(plant, U=u, T=t)

# add measurement noise to mimic a real experimental recording
noise_std = 0.05
y = y_clean + noise_std * np.random.randn(len(t))

# ---------------------------------------------------------------
# 3. Identify the plant: Welch's method (averaged cross/auto-spectra)
# ---------------------------------------------------------------
nperseg = 2048  # segment length -> resolution/variance tradeoff

f, Suu = signal.welch(u, fs=fs, nperseg=nperseg)
_, Syy = signal.welch(y, fs=fs, nperseg=nperseg)
_, Suy = signal.csd(u, y, fs=fs, nperseg=nperseg)

G_est = Suy / Suu                          # estimated FRF
coherence = np.abs(Suy)**2 / (Suu * Syy)   # quality diagnostic

# ---------------------------------------------------------------
# 4. Ground truth FRF for comparison (only possible since this is sim data)
# ---------------------------------------------------------------
w_true, G_true = signal.freqresp(plant, w=2*np.pi*f[1:])
f_true = w_true / (2*np.pi)

# ---------------------------------------------------------------
# 5. Plot: Bode comparison + coherence
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axes[0].semilogx(f[1:], 20*np.log10(np.abs(G_est[1:])), label='Estimated (Welch)')
axes[0].semilogx(f_true, 20*np.log10(np.abs(G_true)), '--', label='True plant')
axes[0].set_ylabel('Magnitude [dB]')
axes[0].legend()
axes[0].set_title('Frequency Response: Identified vs True')

axes[1].semilogx(f[1:], np.angle(G_est[1:], deg=True), label='Estimated (Welch)')
axes[1].semilogx(f_true, np.angle(G_true, deg=True), '--', label='True plant')
axes[1].set_ylabel('Phase [deg]')
axes[1].legend()

axes[2].semilogx(f, coherence)
axes[2].set_ylabel('Coherence')
axes[2].set_xlabel('Frequency [Hz]')
axes[2].set_ylim(0, 1.05)
axes[2].axhline(1.0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('plant_id_result.png', dpi=120)
print("Saved plot to plant_id_result.png")