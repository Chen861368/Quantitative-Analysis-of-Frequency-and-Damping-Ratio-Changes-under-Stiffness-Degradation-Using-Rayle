import numpy as np

def compute_rayleigh_coefficients(freq1, xi1, freq2, xi2):
    """
    Compute Rayleigh damping coefficients α and β based on two modal frequencies and damping ratios.
    Input frequencies should be in Hz, and damping ratios should be dimensionless (e.g., 0.02).
    """
    # Convert frequency to angular frequency ω = 2πf
    omega1 = 2 * np.pi * freq1
    omega2 = 2 * np.pi * freq2

    # Construct the linear system:
    # xi1 = 0.5 * (alpha/omega1 + beta * omega1)
    # xi2 = 0.5 * (alpha/omega2 + beta * omega2)
    A = np.array([
        [1/(2 * omega1), omega1 / 2],
        [1/(2 * omega2), omega2 / 2]
    ])
    b = np.array([xi1, xi2])

    # Solve the system for alpha and beta
    alpha, beta = np.linalg.solve(A, b)

    return alpha, beta


def compute_critical_omega(alpha, beta):
    """
    Compute the critical circular frequency ω at which the mass and stiffness damping contributions are equal,
    i.e., when β·ω² = α.
    Returns both ω (in rad/s) and the corresponding frequency in Hz.
    """
    if beta <= 0:
        raise ValueError("β must be greater than zero.")

    omega = np.sqrt(alpha / beta)
    freq_hz = omega / (2 * np.pi)
    return omega, freq_hz


# # SSI-COV
# f1, xi1 = 2.872776694, 0.037313625
# f2, xi2 = 5.831685229, 0.025622501

# # SSI-DATA
# f1, xi1 = 2.874263865, 0.023991137
# f2, xi2 = 5.732930819, 0.019830695

#  PCSSI
f1, xi1 = 2.857891913, 0.014261722
f2, xi2 = 5.805083654, 0.008906429

alpha, beta = compute_rayleigh_coefficients(f1, xi1, f2, xi2)
print("Calculated Rayleigh damping coefficients:")
print(f"α (mass-proportional)        = {alpha:.6f}")
print(f"β (stiffness-proportional)   = {beta:.6f}")

omega, freq_hz = compute_critical_omega(alpha, beta)
print(f"Critical circular frequency ω = {omega:.6f} rad/s")
print(f"Corresponding frequency f     = {freq_hz:.6f} Hz")






