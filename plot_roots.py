import numpy as np
import matplotlib.pyplot as plt

def main():
    # Given parameters
    Omega = 2.0         # rad/s
    r0 = 0.5            # m
    H0 = 0.05           # m
    rho0 = 1000.0       # kg/m^3
    B0 = 0.003          # T
    C = 0.05            # m^3/s^2
    m = 2.0             # azimuthal wavenumber
    mu0 = 4 * np.pi * 1e-7  # T m / A

    # Derived parameters
    f = 2 * Omega
    c0_sq = (Omega**2 * r0 - C / r0**2) * H0
    Omega_A_sq = B0**2 / (mu0 * rho0 * H0**2)
    B_param = (Omega**2 + 2 * C / r0**3) * H0
    k_theta = m / r0

    # Wavenumber range for k_r
    kr_vals = np.linspace(0.1, 50, 1000)

    roots_1 = []
    roots_2 = []
    roots_3 = []
    roots_4 = []
    k_vals = []

    for kr in kr_vals:
        k_sq = kr**2 + k_theta**2
        k = np.sqrt(k_sq)
        k_vals.append(k)

        # Coefficients of the quartic equation:
        # C4 * w^4 + C3 * w^3 + C2 * w^2 + C1 * w + C0 = 0

        C4 = 1.0
        C3 = 0.0
        C2 = -(k_sq * c0_sq - 2 * Omega_A_sq + f**2 - 1j * B_param * kr)
        C1 = f * B_param * k_theta
        C0 = -(Omega_A_sq * (k_sq * c0_sq - Omega_A_sq)) + 1j * B_param * kr * Omega_A_sq

        coeffs = [C4, C3, C2, C1, C0]

        # Calculate roots using standard numpy
        rts = np.roots(coeffs)

        # Sort roots to maintain consistency across iterations
        # We can sort by real part first
        rts = sorted(rts, key=lambda x: (x.real, x.imag))

        # Normalize roots by 2*Omega
        rts_norm = [r / (2 * Omega) for r in rts]

        roots_1.append(rts_norm[0])
        roots_2.append(rts_norm[1])
        roots_3.append(rts_norm[2])
        roots_4.append(rts_norm[3])

    roots_1 = np.array(roots_1)
    roots_2 = np.array(roots_2)
    roots_3 = np.array(roots_3)
    roots_4 = np.array(roots_4)
    k_vals = np.array(k_vals)

    # Function to create individual plots
    def plot_root(k, root_array, title, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(k, root_array.real, label='Real')
        plt.plot(k, root_array.imag, label='Imaginary', linestyle='dashed')
        plt.xlabel(r'Total Wavenumber $k = \sqrt{k_r^2 + (m/r_0)^2}$')
        plt.ylabel(r'Normalized Frequency ($\omega / 2\Omega$)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    # Create individual plots
    plot_root(k_vals, roots_1, 'Root 1 vs k', 'root_1.png')
    plot_root(k_vals, roots_2, 'Root 2 vs k', 'root_2.png')
    plot_root(k_vals, roots_3, 'Root 3 vs k', 'root_3.png')
    plot_root(k_vals, roots_4, 'Root 4 vs k', 'root_4.png')

    # Create combined plot
    plt.figure(figsize=(10, 8))
    plt.plot(k_vals, roots_1.real, label='Root 1 (Real)', color='blue')
    plt.plot(k_vals, roots_2.real, label='Root 2 (Real)', color='red')
    plt.plot(k_vals, roots_3.real, label='Root 3 (Real)', color='green')
    plt.plot(k_vals, roots_4.real, label='Root 4 (Real)', color='purple')

    # Also plot imaginary parts if you want them on the combined plot
    plt.plot(k_vals, roots_1.imag, label='Root 1 (Imag)', color='blue', linestyle='dashed')
    plt.plot(k_vals, roots_2.imag, label='Root 2 (Imag)', color='red', linestyle='dashed')
    plt.plot(k_vals, roots_3.imag, label='Root 3 (Imag)', color='green', linestyle='dashed')
    plt.plot(k_vals, roots_4.imag, label='Root 4 (Imag)', color='purple', linestyle='dashed')

    plt.xlabel(r'Total Wavenumber $k = \sqrt{k_r^2 + (m/r_0)^2}$')
    plt.ylabel(r'Normalized Frequency ($\omega / 2\Omega$)')
    plt.title(r'All Dispersion Branches (Normalized $\omega$) vs Total Wavenumber $k$')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('combined_roots.png')
    plt.close()

if __name__ == "__main__":
    main()
