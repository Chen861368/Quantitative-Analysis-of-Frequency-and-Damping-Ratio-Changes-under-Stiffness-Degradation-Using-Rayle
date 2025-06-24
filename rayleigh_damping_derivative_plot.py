import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font to Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16  # Increase font size

# Rayleigh damping coefficients
alpha = 0.05
beta = 0.06

# Frequency range (in rad/s)
omega = np.linspace(0.1, 2, 1000)  # Avoid division by zero at ω = 0
freq_hz = omega / (2 * np.pi)

# Compute damping ratio and its derivative
zeta = alpha / (2 * omega) + beta * omega / 2
dzeta_domega = (-alpha + beta * omega**2) / (2 * omega**2)

# Compute the critical frequency where the derivative is zero
omega_crit = np.sqrt(alpha / beta)
freq_crit_hz = omega_crit / (2 * np.pi)

# Plot the derivative of the damping ratio
plt.figure(figsize=(8, 6))
plt.plot(freq_hz, dzeta_domega, label=r'$\frac{d\zeta}{d\omega}$', linewidth=1.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(freq_crit_hz, color='red', linestyle='--', linewidth=1.2, label=f'Critical frequency = {freq_crit_hz:.3f} Hz')
plt.scatter(freq_crit_hz, 0, color='red', zorder=5)

# Axis labels and legend
plt.xlabel('Frequency (Hz)', fontsize=18)
plt.ylabel(r'Derivative of Damping Ratio $\frac{d\zeta}{d\omega}$', fontsize=18)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()












