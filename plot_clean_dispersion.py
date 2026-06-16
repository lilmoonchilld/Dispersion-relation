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
    m = 1  # Non-zero azimuthal wavenumber

    # Range of K values
    K_vals = np.linspace(0.001, 0.15, 1000)

    # Radii to plot
    r_vals = [1e8, 1.5e8, 2e8, 2.5e8]

    # 4 Subplots for 4 roots
    fig, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

    for r in r_vals:
        Omega = np.sqrt(G * M / r**3)
        H = H0 * (r / r0)**(1.5)
        B0 = 2e-5 * (R_saturn / r)**3

        omega_a_sq = B0**2 / (mu0 * rho * H**2)

        root1_real = []
        root2_real = []
        root3_real = []
        root4_real = []

        root1_imag = []
        root2_imag = []
        root3_imag = []
        root4_imag = []

        for k in K_vals:
            # k is the absolute wavenumber. In the formula k^2 = kr^2 + m^2/r^2.
            # But the x-axis is just k as an independent variable representing |k|.
            k_sq = k**2

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
            root1_imag.append(roots_sorted[0].imag)

            root2_real.append(roots_sorted[1].real)
            root2_imag.append(roots_sorted[1].imag)

            root3_real.append(roots_sorted[2].real)
            root3_imag.append(roots_sorted[2].imag)

            root4_real.append(roots_sorted[3].real)
            root4_imag.append(roots_sorted[3].imag)

        label_r = f'$r = {r/1e8:.1f} \\times 10^8$ m'

        # Plot real parts normalized by 2*Omega
        line1, = axs[0].plot(K_vals, np.array(root1_real) / (2 * Omega), label=label_r)
        line2, = axs[1].plot(K_vals, np.array(root2_real) / (2 * Omega), label=label_r)
        line3, = axs[2].plot(K_vals, np.array(root3_real) / (2 * Omega), label=label_r)
        line4, = axs[3].plot(K_vals, np.array(root4_real) / (2 * Omega), label=label_r)

        # Plot imaginary parts as dotted lines if they exist (instability)
        r1i = np.array(root1_imag)
        r2i = np.array(root2_imag)
        r3i = np.array(root3_imag)
        r4i = np.array(root4_imag)

        if np.any(np.abs(r1i) > 1e-10):
            axs[0].plot(K_vals, r1i / (2 * Omega), linestyle=':', color=line1.get_color())
        if np.any(np.abs(r2i) > 1e-10):
            axs[1].plot(K_vals, r2i / (2 * Omega), linestyle=':', color=line2.get_color())
        if np.any(np.abs(r3i) > 1e-10):
            axs[2].plot(K_vals, r3i / (2 * Omega), linestyle=':', color=line3.get_color())
        if np.any(np.abs(r4i) > 1e-10):
            axs[3].plot(K_vals, r4i / (2 * Omega), linestyle=':', color=line4.get_color())

    root_names = ['Root 1 (Most Retrograde)', 'Root 2', 'Root 3', 'Root 4 (Most Prograde)']

    for i, ax in enumerate(axs):
        ax.set_ylabel('Re[$\\omega$] / $2\\Omega$')
        ax.set_title(root_names[i] + f' ($m={m}$)')
        ax.legend()
        ax.grid(True)

    axs[-1].set_xlabel('$k$ ($|\\vec{k}|$ in 1/m)')

    plt.tight_layout()
    plt.savefig('clean_dispersion_relation_m.png')
    print("Plot saved to clean_dispersion_relation_m.png")

if __name__ == "__main__":
    main()
