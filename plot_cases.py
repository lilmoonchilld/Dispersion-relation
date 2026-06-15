import numpy as np
import matplotlib.pyplot as plt

# Constants
G_grav = 6.67430e-11
M_saturn = 5.6834e26
R_saturn = 6.0268e7
r0 = 1.0e8 # 100,000 km
Omega = np.sqrt(G_grav * M_saturn / r0**3) # Keplerian Omega
H0 = 10.0 # 10 meters
rho = 100.0 # kg/m^3
mu0 = 4 * np.pi * 1e-7
B0 = 2e-5 * (R_saturn / r0)**3
m = 1

# Local variables at r0
r = r0
H = H0

# Frequencies
omega_A2 = B0**2 / (mu0 * rho * H**2)
omega_A = np.sqrt(omega_A2)

# Range of k
k = np.logspace(-8, 0, 1000)

roots_exact = []
roots_ig = []
roots_rossby_nm = []
roots_mag_slow_1 = []
roots_mag_slow_2 = []

for ki in k:
    Phi = G_grav * M_saturn * H / r**3 - 2 * np.pi * G_grav * rho / ki

    # Exact roots
    C4 = 1.0
    C3 = 0.0
    C2 = -(2 * omega_A2 + 4 * Omega**2 + H * Phi * ki**2)
    C1 = -(H * Phi * 5 * m * Omega / r**2)
    C0 = omega_A2 * (omega_A2 + H * Phi * ki**2)
    coeffs = [C4, C3, C2, C1, C0]
    w = np.roots(coeffs)
    w_sorted = w[np.argsort(w.real)]
    roots_exact.append(w_sorted / (2 * Omega))

    # Non-magnetic IG waves
    val_ig = 4 * Omega**2 + H * Phi * ki**2
    if val_ig >= 0:
        w_ig_1 = np.sqrt(val_ig)
        w_ig_2 = -np.sqrt(val_ig)
    else:
        w_ig_1 = 1j * np.sqrt(-val_ig)
        w_ig_2 = -1j * np.sqrt(-val_ig)
    roots_ig.append([w_ig_1 / (2 * Omega), w_ig_2 / (2 * Omega)])

    # Non-magnetic Rossby mode
    w_rossby = - (H * Phi * 5 * m * Omega / r**2) / val_ig if val_ig != 0 else 0
    roots_rossby_nm.append(w_rossby / (2 * Omega))

    # Magnetic low-frequency modes
    a = 4 * Omega**2
    b = H * Phi * 5 * m * Omega / r**2
    c = - omega_A2 * (omega_A2 + H * Phi * ki**2)
    discriminant = b**2 - 4 * a * c
    if discriminant >= 0:
        w_mag_1 = (-b + np.sqrt(discriminant)) / (2 * a)
        w_mag_2 = (-b - np.sqrt(discriminant)) / (2 * a)
    else:
        w_mag_1 = (-b + 1j * np.sqrt(-discriminant)) / (2 * a)
        w_mag_2 = (-b - 1j * np.sqrt(-discriminant)) / (2 * a)
    roots_mag_slow_1.append(w_mag_1 / (2 * Omega))
    roots_mag_slow_2.append(w_mag_2 / (2 * Omega))

roots_exact = np.array(roots_exact)
roots_ig = np.array(roots_ig)
roots_rossby_nm = np.array(roots_rossby_nm)
roots_mag_slow_1 = np.array(roots_mag_slow_1)
roots_mag_slow_2 = np.array(roots_mag_slow_2)

# Plotting Analytical Cases Separate
fig, axs = plt.subplots(3, 2, figsize=(14, 15))

# IG waves
axs[0, 0].plot(k, np.real(roots_ig[:, 0]), label='IG+')
axs[0, 0].plot(k, np.real(roots_ig[:, 1]), label='IG-')
axs[0, 0].set_xscale('log')
axs[0, 0].set_title('Non-Magnetic IG Waves (Real)')
axs[0, 0].set_xlabel('Wavenumber k (m^-1)')
axs[0, 0].set_ylabel('Re(omega) / 2Omega')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(k, np.imag(roots_ig[:, 0]), label='IG+')
axs[0, 1].plot(k, np.imag(roots_ig[:, 1]), label='IG-')
axs[0, 1].set_xscale('log')
axs[0, 1].set_title('Non-Magnetic IG Waves (Imaginary)')
axs[0, 1].set_xlabel('Wavenumber k (m^-1)')
axs[0, 1].set_ylabel('Im(omega) / 2Omega')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Rossby NM
axs[1, 0].plot(k, np.real(roots_rossby_nm), label='Rossby NM', color='green')
axs[1, 0].set_xscale('log')
axs[1, 0].set_title('Non-Magnetic Rossby Mode (Real)')
axs[1, 0].set_xlabel('Wavenumber k (m^-1)')
axs[1, 0].set_ylabel('Re(omega) / 2Omega')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(k, np.imag(roots_rossby_nm), label='Rossby NM', color='green')
axs[1, 1].set_xscale('log')
axs[1, 1].set_title('Non-Magnetic Rossby Mode (Imaginary)')
axs[1, 1].set_xlabel('Wavenumber k (m^-1)')
axs[1, 1].set_ylabel('Im(omega) / 2Omega')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Magnetic Slow
axs[2, 0].plot(k, np.real(roots_mag_slow_1), label='Mag Slow 1', color='purple')
axs[2, 0].plot(k, np.real(roots_mag_slow_2), label='Mag Slow 2', color='brown')
axs[2, 0].set_xscale('log')
axs[2, 0].set_title('Magnetic Slow Modes (Real)')
axs[2, 0].set_xlabel('Wavenumber k (m^-1)')
axs[2, 0].set_ylabel('Re(omega) / 2Omega')
axs[2, 0].legend()
axs[2, 0].grid(True)

axs[2, 1].plot(k, np.imag(roots_mag_slow_1), label='Mag Slow 1', color='purple')
axs[2, 1].plot(k, np.imag(roots_mag_slow_2), label='Mag Slow 2', color='brown')
axs[2, 1].set_xscale('log')
axs[2, 1].set_title('Magnetic Slow Modes (Imaginary)')
axs[2, 1].set_xlabel('Wavenumber k (m^-1)')
axs[2, 1].set_ylabel('Im(omega) / 2Omega')
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
fig.savefig('analytical_cases.png')

# Plotting Comparison Together
fig2, ax2 = plt.subplots(1, 2, figsize=(16, 8))

# Exact roots
for i in range(4):
    ax2[0].plot(k, np.real(roots_exact[:, i]), 'k-', alpha=0.3, lw=5, label='Exact Roots' if i==0 else "")
    ax2[1].plot(k, np.imag(roots_exact[:, i]), 'k-', alpha=0.3, lw=5, label='Exact Roots' if i==0 else "")

# IG
ax2[0].plot(k, np.real(roots_ig[:, 0]), '--', label='Analytical IG+')
ax2[0].plot(k, np.real(roots_ig[:, 1]), '--', label='Analytical IG-')
ax2[1].plot(k, np.imag(roots_ig[:, 0]), '--', label='Analytical IG+')
ax2[1].plot(k, np.imag(roots_ig[:, 1]), '--', label='Analytical IG-')

# Rossby NM
ax2[0].plot(k, np.real(roots_rossby_nm), ':', label='Analytical Rossby NM')
ax2[1].plot(k, np.imag(roots_rossby_nm), ':', label='Analytical Rossby NM')

# Mag Slow
ax2[0].plot(k, np.real(roots_mag_slow_1), '-.', label='Analytical Mag Slow 1')
ax2[0].plot(k, np.real(roots_mag_slow_2), '-.', label='Analytical Mag Slow 2')
ax2[1].plot(k, np.imag(roots_mag_slow_1), '-.', label='Analytical Mag Slow 1')
ax2[1].plot(k, np.imag(roots_mag_slow_2), '-.', label='Analytical Mag Slow 2')

ax2[0].set_xscale('log')
ax2[0].set_title('All Modes Combined (Real)')
ax2[0].set_xlabel('Wavenumber k (m^-1)')
ax2[0].set_ylabel('Re(omega) / 2Omega')
ax2[0].legend(loc='best', fontsize='small')
ax2[0].grid(True)

ax2[1].set_xscale('log')
ax2[1].set_title('All Modes Combined (Imaginary)')
ax2[1].set_xlabel('Wavenumber k (m^-1)')
ax2[1].set_ylabel('Im(omega) / 2Omega')
ax2[1].legend(loc='best', fontsize='small')
ax2[1].grid(True)

plt.tight_layout()
fig2.savefig('dispersion_comparison.png')

print("Plotting complete. Saved 'analytical_cases.png' and 'dispersion_comparison.png'.")
