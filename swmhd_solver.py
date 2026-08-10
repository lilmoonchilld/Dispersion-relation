import numpy as np

def get_coefficients_dimensional(m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n):
    # Calculations exactly at r = r0
    # Heq(r0) = H0 + (Omega^2 / (2g))*(r0^2 - r0^2) + (C / g)*(1/r0 - 1/r0) = H0
    Heq = H0
    # Q(r0) = Omega^2 * r0 - C / r0^2
    Q = Omega**2 * r0 - C / r0**2
    # f = 2 * Omega
    f = 2.0 * Omega
    # omega_A^2 = B0^2 / (mu0 * rho0 * Heq^2)
    omega_A_sq = B0**2 / (mu0 * rho0 * Heq**2)
    # kr^2 = (n * pi / (r2 - r1))^2
    kr = n * np.pi / (r2 - r1)
    kr_sq = kr**2
    # kappa^2 = kr^2 + m^2 / r0^2
    kappa_sq = kr_sq + m**2 / r0**2

    # Coefficients:
    A8 = 1.0
    A7 = 0.0
    A6 = 4.0 * omega_A_sq - 2.0 * f**2 - kappa_sq * g * Heq
    A5 = m * f * Q / r0
    A4 = (6.0 * omega_A_sq**2 - 4.0 * f**2 * omega_A_sq + f**4
          - kappa_sq * g * Heq * (3.0 * omega_A_sq - f**2))
    A3 = (m * f * Q / r0) * (6.0 * omega_A_sq - f**2)
    A2 = (4.0 * omega_A_sq**3 - 2.0 * f**2 * omega_A_sq**2
          - kappa_sq * g * Heq * (3.0 * omega_A_sq**2 - f**2 * omega_A_sq))
    A1 = 5.0 * m * f * Q * omega_A_sq**2 / r0
    A0 = omega_A_sq**4 - kappa_sq * g * Heq * omega_A_sq**3

    return [A8, A7, A6, A5, A4, A3, A2, A1, A0], Heq, Q, f, omega_A_sq, kr, kr_sq, kappa_sq

def get_coefficients_dimensionless(m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n):
    # We define dimensionless frequency as \hat{\omega} = \omega / \Omega.
    # Therefore, we substitute x = \Omega * \hat{\omega} into the dimensional polynomial:
    # \sum_{p=0}^8 A_p (\Omega \hat{\omega})^p = 0
    # Dividing by \Omega^8 gives the dimensionless polynomial with coefficients \tilde{A}_p = A_p \Omega^{p-8}.

    coeffs_dim, Heq, Q, f, omega_A_sq, kr, kr_sq, kappa_sq = get_coefficients_dimensional(
        m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n
    )

    coeffs_scaled = []
    for p in range(9):
        # coeffs_dim is in order A8, A7, ..., A0, which correspond to p = 8, 7, ..., 0
        A_p = coeffs_dim[8 - p]
        A_p_tilde = A_p * (Omega ** (p - 8))
        coeffs_scaled.append(A_p_tilde)

    # Reverse to keep the order from high power to low power: \tilde{A}_8, \tilde{A}_7, ..., \tilde{A}_0
    coeffs_scaled.reverse()
    return coeffs_scaled

def solve_m_dimensional(m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n):
    coeffs, Heq, Q, f, omega_A_sq, kr, kr_sq, kappa_sq = get_coefficients_dimensional(
        m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n
    )
    roots = np.roots(coeffs)
    return roots, coeffs

def solve_m_dimensionless(m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n):
    coeffs_scaled = get_coefficients_dimensionless(
        m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n
    )
    roots_scaled = np.roots(coeffs_scaled)
    return roots_scaled, coeffs_scaled

def main():
    # Verify the code compiles and can run for m = 1
    Omega = 0.5e-4
    H0 = 500.0
    g = 9.81
    B0 = 5e-4
    rho0 = 1000.0
    C = 0.0
    r1 = 0.5e6
    r2 = 1.0e6
    r0 = 0.5 * (r1 + r2)
    mu0 = 4.0 * np.pi * 1e-7
    n = 2

    roots_dim, coeffs_dim = solve_m_dimensional(1, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n)
    roots_scaled, coeffs_scaled = solve_m_dimensionless(1, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n)

    print("Dimensional roots:")
    print(roots_dim)
    print("\nDimensionless roots * Omega:")
    print(roots_scaled * Omega)

    # Compare sorted roots to see if they are identical
    sorted_dim = np.sort_complex(roots_dim)
    sorted_scaled = np.sort_complex(roots_scaled * Omega)
    max_diff = np.max(np.abs(sorted_dim - sorted_scaled))
    print(f"\nMax difference between dimensional and scaled solvers: {max_diff}")

if __name__ == "__main__":
    main()
