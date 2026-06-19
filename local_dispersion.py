import numpy as np
import matplotlib.pyplot as plt

def main():
    G = 6.67430e-11
    M = 5.683e26
    R_saturn = 60268000.0
    r0 = 100000000.0
    H0 = 10.0
    rho = 100.0
    mu0 = 4 * np.pi * 1e-7
    m = 1  # Non-zero azimuthal wavenumber

    # Range of K values
    K_vals = np.linspace(-0.01, 0.2, 1000)

    # Specific radius to plot
    r = 1e8

    # 2 Subplots: Real and Imaginary
    fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    Omega = np.sqrt(G * M / r**3)
    H = H0 * (r / r0)**(1.5)
    B0 = 2e-5 * (R_saturn / r)**3

    omega_a_sq = B0**2 / (mu0 * rho * H**2)

    real_parts = []
    imag_parts = []

    A = omega_a_sq
    D = 5 * m * Omega / r**2

    for K in K_vals:
        kr = K
        k_sq = K**2 + m**2 / r**2
        k_abs = np.sqrt(k_sq)

        B_term = H * (G * M * H / r**3 - 2 * np.pi * G * rho / k_abs)
        C_term = k_sq - 1j * 5 * kr / (2 * r)

        C4 = 1
        C3 = 0
        C2 = -(2*A + 4*Omega**2 + B_term * C_term)
        C1 = -(D * B_term)
        C0 = A * (A + B_term * C_term)

        coeffs = [C4, C3, C2, C1, C0]
        rts = np.roots(coeffs)

        # Find the root with the maximum imaginary part (most unstable)
        max_rt = max(rts, key=lambda rt: rt.imag)
        real_parts.append(max_rt.real)
        imag_parts.append(max_rt.imag)

    ax_real.plot(K_vals, real_parts, 'b')
    ax_real.set_ylabel(r'Re($\omega$)')
    ax_real.set_title('Local Dispersion Relation (Real part)')
    ax_real.grid(True)

    ax_imag.plot(K_vals, imag_parts, 'r')
    ax_imag.set_xlabel('Radial wavenumber K')
    ax_imag.set_ylabel(r'Max Im($\omega$) [Growth Rate]')
    ax_imag.set_title('Local Dispersion Relation (Imaginary part)')
    ax_imag.grid(True)

    plt.tight_layout()
    plt.savefig('local_dispersion.png')
    plt.close()

if __name__ == "__main__":
    main()
