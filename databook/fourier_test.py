import numpy as np
import matplotlib.pyplot as plt

def f_hat(x):
    """
    Computes f(x) for the hat function defined in Eq. (2.13).
    Assumes x in the interval [-π, π).
    """
    if -np.pi <= x < -np.pi/2:
        return 0
    elif -np.pi/2 <= x < 0:
        return 1 + (2 * x / np.pi)
    elif 0 <= x < np.pi/2:
        return 1 - (2 * x / np.pi)
    elif np.pi/2 <= x < np.pi:
        return 0
    else:
        raise ValueError(f"x={x} is out of bounds. Must be in [-π, π).")
    
# plot the f_hat function
x = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
y = []

for xi in x:
    y.append(f_hat(xi))

# calculate a0: DC component of function over period L
L = np.pi
a0 = (1/L) * np.trapezoid(y, x)
print(f"a0 = {a0}") 

ak = []
bk = []

# Compute Fourier coefficients
num_terms = 200
ak = []
bk = []

for k in range(1, num_terms + 1):  # include term k=num_terms
    ak_val = (1 / L) * np.trapz([f_hat(xi) * np.cos(k * xi) for xi in x], x)
    bk_val = (1 / L) * np.trapz([f_hat(xi) * np.sin(k * xi) for xi in x], x)
    ak.append(ak_val)
    bk.append(bk_val)
    print(f"a{k} = {ak_val:.5f}, b{k} = {bk_val:.5f}")

# Reconstruct function
def fourier_series(x, a0, ak, bk, num_terms):
    result = a0 / 2
    for k in range(1, num_terms + 1):
        result += ak[k-1] * np.cos(k * x) + bk[k-1] * np.sin(k * x)
    return result

y_reconstructed = [fourier_series(xi, a0, ak, bk, num_terms) for xi in x]

# Plot
plt.plot(x, y, label="Original f(x)")
plt.plot(x, y_reconstructed, '--', label=f"Fourier Series (n={num_terms})")
plt.title("Fourier Series Approximation of Hat Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
