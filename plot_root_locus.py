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

# --- Plot Cartesian Root Locus ---
plt.figure(figsize=(10, 8))
# Exact roots
for i in range(4):
    plt.plot(np.real(roots_exact[:, i]), np.imag(roots_exact[:, i]), 'k-', alpha=0.3, lw=4, label='Exact Roots' if i==0 else "")

# Analytical
plt.plot(np.real(roots_ig[:, 0]), np.imag(roots_ig[:, 0]), '--', label='Analytical IG+')
plt.plot(np.real(roots_ig[:, 1]), np.imag(roots_ig[:, 1]), '--', label='Analytical IG-')
plt.plot(np.real(roots_rossby_nm), np.imag(roots_rossby_nm), ':', label='Analytical Rossby NM', color='green')
plt.plot(np.real(roots_mag_slow_1), np.imag(roots_mag_slow_1), '-.', label='Analytical Mag Slow 1', color='purple')
plt.plot(np.real(roots_mag_slow_2), np.imag(roots_mag_slow_2), '-.', label='Analytical Mag Slow 2', color='brown')

plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.title('Root Locus Plot (Complex Plane)')
plt.xlabel('Re(omega) / 2Omega')
plt.ylabel('Im(omega) / 2Omega')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('root_locus.png')
plt.close()

# --- Plot Polar Root Locus ---
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

# Helper function to get polar coords
def to_polar(z):
    r = np.abs(z)
    theta = np.angle(z)
    return theta, r

# Exact roots
for i in range(4):
    theta, r_val = to_polar(roots_exact[:, i])
    ax.plot(theta, r_val, 'k-', alpha=0.3, lw=4, label='Exact Roots' if i==0 else "")

# Analytical
theta_ig0, r_ig0 = to_polar(roots_ig[:, 0])
theta_ig1, r_ig1 = to_polar(roots_ig[:, 1])
ax.plot(theta_ig0, r_ig0, '--', label='Analytical IG+')
ax.plot(theta_ig1, r_ig1, '--', label='Analytical IG-')

theta_r, r_r = to_polar(roots_rossby_nm)
ax.plot(theta_r, r_r, ':', label='Analytical Rossby NM', color='green')

theta_m1, r_m1 = to_polar(roots_mag_slow_1)
theta_m2, r_m2 = to_polar(roots_mag_slow_2)
ax.plot(theta_m1, r_m1, '-.', label='Analytical Mag Slow 1', color='purple')
ax.plot(theta_m2, r_m2, '-.', label='Analytical Mag Slow 2', color='brown')

ax.set_title('Polar Root Locus Plot (Radius=|omega|, Angle=arg(omega))', va='bottom')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('root_locus_polar.png')
plt.close()

print("Plotting complete. Saved 'root_locus.png' and 'root_locus_polar.png'.")
