import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

# --- 1. Create the clean signal ---
Fs = 1000  # Sampling frequency [Hz]
T = 1 / Fs  # Sampling interval [s]
L = 250     # Number of samples
t = np.linspace(0, L*T, L, endpoint=False)  # Time vector (0 to 0.25s)

# Clean signal: sum of two sine waves
f1, f2 = 50, 120
clean_signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# --- 2. Add Gaussian noise ---
noise = 2.5 * np.random.randn(L)
noisy_signal = clean_signal + noise

# --- 3. Perform FFT ---
yf = fft(noisy_signal)
xf = fftfreq(L, T)  # Frequency vector

# --- 4. Filter: Keep only frequencies near 50 and 120 Hz ---
filtered_fft = np.zeros_like(yf)

step_size = xf[1] - xf[0]  # Frequency resolution

# Find indices where frequencies are close to f1 and f2
indices_to_keep = (np.abs(xf - f1) < step_size) | (np.abs(xf - f2) < step_size)
filtered_fft[indices_to_keep] = yf[indices_to_keep]

# --- 5. Inverse FFT to get time domain signal ---
filtered_signal = 2.0 * ifft(filtered_fft).real

# --- 6. Plot ---
plt.figure(figsize=(10, 8))

# Top plot: clean vs noisy
plt.subplot(3, 1, 1)
plt.plot(t, noisy_signal, 'r', label='Noisy')
plt.plot(t, clean_signal, 'k', label='Clean')
plt.title('Signal in Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('f')
plt.legend()

# Middle plot: Power Spectrum
plt.subplot(3, 1, 2)
plt.plot(xf[:L//2], 2.0/L * np.abs(yf[:L//2]), 'r', label='Noisy')
plt.plot(xf[:L//2], 2.0/L * np.abs(filtered_fft[:L//2]), 'b', label='Filtered')
plt.title('FFT Magnitude Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD')
plt.legend()

# Bottom plot: clean vs filtered
plt.subplot(3, 1, 3)
plt.plot(t, clean_signal, 'k', label='Clean')
plt.plot(t, filtered_signal, 'b', label='Filtered')
plt.title('Filtered Signal in Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('f')
plt.legend()

plt.tight_layout()
plt.show()
