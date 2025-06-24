# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(0)
from numpy.linalg import eig, inv

def compute_critical_omega(alpha, beta):
    """
    Compute the critical circular frequency ω at which the mass- and stiffness-proportional
    damping contributions are equal (i.e., when β·ω² = α).

    Returns:
        omega (float): Critical circular frequency in rad/s
        freq_hz (float): Corresponding frequency in Hz
        zeta_crit (float): Critical damping ratio at this frequency
    """
    if beta <= 0:
        raise ValueError("β must be greater than zero.")

    omega = np.sqrt(alpha / beta)
    freq_hz = omega / (2 * np.pi)

    # Compute the damping ratio at the critical frequency using the Rayleigh damping formula
    zeta_crit = alpha / (2 * omega) + beta * omega / 2

    return omega, freq_hz, zeta_crit


def compute_natural_frequencies(M, K):
    """
    Compute the natural frequencies (in Hz) of an undamped system.
    
    Args:
        M (ndarray): Mass matrix (n × n)
        K (ndarray): Stiffness matrix (n × n)
    
    Returns:
        freq_hz (ndarray): Array of natural frequencies in Hz, sorted in ascending order
    """
    # Solve the generalized eigenvalue problem: K φ = λ M φ
    eigvals, _ = eig(K, M)

    # Retain only the real, positive eigenvalues (filtering out invalid or negative ones)
    eigvals = np.real(eigvals)
    eigvals = eigvals[eigvals > 0]

    # Convert eigenvalues to circular natural frequencies ω = sqrt(λ)
    omega_n = np.sqrt(eigvals)
    freq_hz = omega_n / (2 * np.pi)

    # Sort frequencies in ascending order
    idx = np.argsort(freq_hz)
    freq_hz = freq_hz[idx]

    # Print the results
    print("Mode\tNatural Frequency (Hz)")
    for i, f in enumerate(freq_hz):
        print(f"{i+1}\t{f:.4f} Hz")

    return freq_hz


def modal_params(M, C, K):
    """
    Compute natural frequencies (Hz) and damping ratios of an MDOF system.
    
    Parameters
    ----------
    M, C, K : (n, n) ndarray
        Mass, damping, and stiffness matrices.
    
    Returns
    -------
    freq_hz : ndarray, shape (n,)
        Sorted natural frequencies in Hz.
    zeta    : ndarray, shape (n,)
        Corresponding damping ratios.
    """
    n = M.shape[0]
    zero = np.zeros((n, n))
    I    = np.eye(n)
    Minv = inv(M)

    # State-space matrix
    A = np.block([[zero,            I],
                  [-Minv @ K, -Minv @ C]])

    # Eigen-analysis
    eigvals, _ = eig(A)
    eigvals = eigvals[np.argsort(np.real(eigvals))]   # sort for consistency
    eigvals = eigvals[::2]                            # pick one from each conjugate pair

    # Damped circular freq. & decay
    omega_d = np.abs(np.imag(eigvals))
    sigma   = -np.real(eigvals)

    # Natural (undamped) frequencies & damping ratios
    omega_n = np.sqrt(omega_d**2 + sigma**2)
    zeta    = sigma / omega_n

    # Sort ascending by natural frequency
    idx = np.argsort(omega_n)
    omega_n = omega_n[idx]
    zeta    = zeta[idx]

    # Convert to Hz
    freq_hz = omega_n / (2 * np.pi)
    return freq_hz, zeta


# ----------------------------------------------------------------------


# System definition
# Example: m = 1.0, k = 1.0, n = 5
m, k, n = 1.0, 0.5, 5
t_end, dt = 1000, 0.005
time = np.arange(0, t_end + dt, dt)

# Input Rayleigh damping coefficients
alpha = 0.05
beta = 0.06

# Construct mass, stiffness, and damping matrices
M = m * np.eye(n)
K = np.diag([2 * k] * (n - 1) + [k]) - np.diag([k] * (n - 1), 1) - np.diag([k] * (n - 1), -1)
C = alpha * M + beta * K  # Proportional damping

# Compute modal parameters (natural frequencies and damping ratios)
f, z = modal_params(M, C, K)
for i, (fi, zi) in enumerate(zip(f, z), 1):
    print(f"Mode {i}:  {fi:.4f} Hz   ζ = {zi:.4f}")

# Compute critical frequency and corresponding damping ratio
omega_crit, freq_crit, zeta_crit = compute_critical_omega(alpha, beta)
print(f"\nCritical circular frequency ω = {omega_crit:.4f} rad/s")
print(f"Corresponding frequency f = {freq_crit:.4f} Hz")
print(f"Critical damping ratio ζ = {zeta_crit:.4f}")






















