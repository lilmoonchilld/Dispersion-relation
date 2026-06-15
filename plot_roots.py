import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parameters
    G = 6.67430e-11
    M = 5.683e26
    R_saturn = 60268000.0
    r = 100000000.0
    Omega = 1.637e-4
    H = 10.0
    rho = 100.0
    mu0 = 4 * np.pi * 1e-7
    B0 = 2e-5 * (R_saturn / r)**3
    m = 0

    # We want K values.
    # Instability usually occurs for a range of k around a few times 10^-3 to 10^-1.
    K_vals = np.linspace(0.001, 0.15, 1000)

    # Let's extract the components for each root
    roots_1 = []
    roots_2 = []
    roots_3 = []
    roots_4 = []

    for K in K_vals:
        kr = K
        k_sq = K**2
        k_abs = K

        A = B0**2 / (mu0 * rho * H**2)
        B_term = H * (G * M * H / r**3 - 2 * np.pi * G * rho / k_abs)
        C_term = k_sq - 1j * 5 * kr / (2 * r)

        C4 = 1
        C3 = 0
        C2 = -(2*A + 4*Omega**2 + B_term * C_term)
        C1 = 0
        C0 = A * (A + B_term * C_term)

        coeffs = [C4, C3, C2, C1, C0]
        rts = np.roots(coeffs)

        # We need a stable way to assign roots 1-4.
        # Since it's a biquadratic equation (roughly), we expect roots to be around +/- omega_a and +/- omega_b.
        # Let's just sort them by real part, then imaginary part.
        rts = sorted(rts, key=lambda x: (x.real, x.imag))

        roots_1.append(rts[0] / (2 * Omega))
        roots_2.append(rts[1] / (2 * Omega))
        roots_3.append(rts[2] / (2 * Omega))
        roots_4.append(rts[3] / (2 * Omega))

    roots_1 = np.array(roots_1)
    roots_2 = np.array(roots_2)
    roots_3 = np.array(roots_3)
    roots_4 = np.array(roots_4)

    # Plotting Root 1
    plt.figure(figsize=(8, 6))
    plt.plot(K_vals, roots_1.real, label='Real')
    plt.plot(K_vals, roots_1.imag, label='Imaginary', linestyle='dashed')
    plt.xlabel('K (|k|)')
    plt.ylabel('Normalized Frequency ($\\omega / 2\\Omega$)')
    plt.title('Root 1 vs K')
    plt.legend()
    plt.grid(True)
    plt.savefig('root_1.png')
    plt.close()

    # Plotting Root 2
    plt.figure(figsize=(8, 6))
    plt.plot(K_vals, roots_2.real, label='Real')
    plt.plot(K_vals, roots_2.imag, label='Imaginary', linestyle='dashed')
    plt.xlabel('K (|k|)')
    plt.ylabel('Normalized Frequency ($\\omega / 2\\Omega$)')
    plt.title('Root 2 vs K')
    plt.legend()
    plt.grid(True)
    plt.savefig('root_2.png')
    plt.close()

    # Plotting Root 3
    plt.figure(figsize=(8, 6))
    plt.plot(K_vals, roots_3.real, label='Real')
    plt.plot(K_vals, roots_3.imag, label='Imaginary', linestyle='dashed')
    plt.xlabel('K (|k|)')
    plt.ylabel('Normalized Frequency ($\\omega / 2\\Omega$)')
    plt.title('Root 3 vs K')
    plt.legend()
    plt.grid(True)
    plt.savefig('root_3.png')
    plt.close()

    # Plotting Root 4
    plt.figure(figsize=(8, 6))
    plt.plot(K_vals, roots_4.real, label='Real')
    plt.plot(K_vals, roots_4.imag, label='Imaginary', linestyle='dashed')
    plt.xlabel('K (|k|)')
    plt.ylabel('Normalized Frequency ($\\omega / 2\\Omega$)')
    plt.title('Root 4 vs K')
    plt.legend()
    plt.grid(True)
    plt.savefig('root_4.png')
    plt.close()

if __name__ == "__main__":
    main()
