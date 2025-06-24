import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# ---- Functions ---- #
def modal_params(M, C, K):
    """
    Compute modal frequencies (Hz) and damping ratios.
    
    Args:
        M: Mass matrix (n x n)
        C: Damping matrix (n x n)
        K: Stiffness matrix (n x n)
    
    Returns:
        freq: Array of natural frequencies in Hz
        zeta: Array of damping ratios
    """
    n = M.shape[0]
    zero = np.zeros((n, n))
    I = np.eye(n)
    Minv = np.linalg.inv(M)

    A_top = np.hstack((zero, I))
    A_bottom = np.hstack((-Minv @ K, -Minv @ C))
    A = np.vstack((A_top, A_bottom))

    eigvals, _ = eig(A)
    eigvals = eigvals[np.argsort(np.real(eigvals))]
    eigvals_unique = eigvals[::2]  # Keep one from each conjugate pair

    omega_d = np.abs(np.imag(eigvals_unique))
    sigma = -np.real(eigvals_unique)
    omega_n = np.sqrt(omega_d**2 + sigma**2)
    zeta = sigma / omega_n
    freq = omega_n / (2 * np.pi)
    return freq, zeta

def compute_critical_omega(alpha, beta):
    """
    Compute the critical frequency where mass and stiffness damping contributions are equal.
    
    Args:
        alpha: Mass-proportional damping coefficient
        beta: Stiffness-proportional damping coefficient
    
    Returns:
        omega_crit: Critical circular frequency (rad/s)
        freq_crit: Critical frequency (Hz)
    """
    omega_crit = np.sqrt(alpha / beta)
    freq_crit = omega_crit / (2 * np.pi)
    return omega_crit, freq_crit

# ---- System Parameters ---- #
m, n = 1.0, 5                      # Unit mass, number of DOFs
alpha, beta = 0.05, 0.06           # Rayleigh damping coefficients
k_values = np.linspace(1.0, 0.2, 20)  # Varying stiffness values

# ---- Data Storage ---- #
frequencies_by_mode = [[] for _ in range(n)]
damping_by_mode = [[] for _ in range(n)]

# ---- Main Computation ---- #
for k in k_values:
    M = m * np.eye(n)
    K = np.diag([2 * k] * (n - 1) + [k]) - np.diag([k] * (n - 1), 1) - np.diag([k] * (n - 1), -1)
    C = alpha * M + beta * K
    freq, zeta = modal_params(M, C, K)
    for i in range(n):
        frequencies_by_mode[i].append(freq[i])
        damping_by_mode[i].append(zeta[i])

# ---- Reference Critical Values ---- #
f_crit = 0.1453     # Critical frequency (Hz)
zeta_crit = 0.0548  # Damping ratio at critical frequency

# ---- Plot: Natural Frequencies vs Stiffness ---- #
plt.figure(figsize=(8, 6))
for i in range(n):
    plt.plot(k_values, frequencies_by_mode[i], label=f'Mode {n - i}')
plt.axhline(y=f_crit, color='black', linestyle='--', linewidth=2, label='Critical Frequency')
plt.xlabel('Stiffness k')
plt.ylabel('Natural Frequency (Hz)')
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

# ---- Plot: Damping Ratios vs Stiffness ---- #
plt.figure(figsize=(8, 6))
for i in range(n):
    plt.plot(k_values, damping_by_mode[i], label=f'Mode {n - i}')
plt.xlabel('Stiffness k')
plt.ylabel('Damping Ratio ζ')
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
