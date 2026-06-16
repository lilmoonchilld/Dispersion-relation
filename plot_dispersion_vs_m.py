import numpy as np
import matplotlib.pyplot as plt

def main():
    # Base parameters
    G = 6.67430e-11
    M = 5.683e26
    R_saturn = 60268000.0
    r0 = 100000000.0
    H0 = 10.0
    rho = 100.0
    mu0 = 4 * np.pi * 1e-7

    # Fixed parameters
    r = r0
    kr = 0.05

    # m values (continuous for smooth curve)
    m_vals = np.linspace(-20, 20, 400)

    Omega = np.sqrt(G * M / r**3)
    H = H0 * (r / r0)**(1.5)
    B0 = 2e-5 * (R_saturn / r)**3

    omega_a_sq = B0**2 / (mu0 * rho * H**2)

    root1_real = []
    root2_real = []
    root3_real = []
    root4_real = []

    for m in m_vals:
        k_sq = kr**2 + (m/r)**2
        k = np.sqrt(k_sq)

        Phi_eff = H * (G * M * H / r**3 - 2 * np.pi * G * rho / k)

        C4 = 1
        C3 = 0
        C2 = -(2 * omega_a_sq + 4 * Omega**2 + Phi_eff * k_sq)
        C1 = - (5 * m * Omega * Phi_eff) / r**2
        C0 = omega_a_sq * (omega_a_sq + Phi_eff * k_sq)

        coeffs = [C4, C3, C2, C1, C0]
        roots = np.roots(coeffs)

        # Sort roots by real part to trace the branches consistently
        roots_sorted = sorted(roots, key=lambda x: x.real)

        root1_real.append(roots_sorted[0].real)
        root2_real.append(roots_sorted[1].real)
        root3_real.append(roots_sorted[2].real)
        root4_real.append(roots_sorted[3].real)

    plt.figure(figsize=(10, 6))

    plt.plot(m_vals, np.array(root1_real) / (2 * Omega), label='Root 1 (Most Retrograde)')
    plt.plot(m_vals, np.array(root2_real) / (2 * Omega), label='Root 2')
    plt.plot(m_vals, np.array(root3_real) / (2 * Omega), label='Root 3')
    plt.plot(m_vals, np.array(root4_real) / (2 * Omega), label='Root 4 (Most Prograde)')

    plt.xlabel('$m$ (azimuthal wavenumber)')
    plt.ylabel('Re[$\\omega$] / $2\\Omega$')
    plt.title(f'Dispersion Relation vs $m$ (at $r = r_0$, $k_r = {kr}$)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dispersion_vs_m.png')
    print("Plot saved to dispersion_vs_m.png")

if __name__ == "__main__":
    main()
