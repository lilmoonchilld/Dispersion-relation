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
H = H0 # Since we evaluate locally at r0, H(r0) = H0

# Frequencies
omega_A2 = B0**2 / (mu0 * rho * H**2)
omega_A = np.sqrt(omega_A2)

# Range of k
k = np.logspace(-8, 0, 1000)

roots_real = []
roots_imag = []

for ki in k:
    Phi = G_grav * M_saturn * H / r**3 - 2 * np.pi * G_grav * rho / ki
    C4 = 1.0
    C3 = 0.0
    C2 = -(2 * omega_A2 + 4 * Omega**2 + H * Phi * ki**2)
    C1 = -(H * Phi * 5 * m * Omega / r**2)
    C0 = omega_A2 * (omega_A2 + H * Phi * ki**2)

    # Solve quartic: C4*w^4 + C3*w^3 + C2*w^2 + C1*w + C0 = 0
    coeffs = [C4, C3, C2, C1, C0]
    w = np.roots(coeffs)

    # Sort roots by real part to track them somewhat consistently
    w_sorted = w[np.argsort(w.real)]
    roots_real.append(w_sorted.real / (2 * Omega))
    roots_imag.append(w_sorted.imag / (2 * Omega))

roots_real = np.array(roots_real)
roots_imag = np.array(roots_imag)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i in range(4):
    plt.plot(k, roots_real[:, i], label=f'Root {i+1}')
plt.xscale('log')
plt.xlabel('Wavenumber k (m^-1)')
plt.ylabel('Re(omega) / 2Omega')
plt.title('Real part of Dispersion Relation Roots')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for i in range(4):
    plt.plot(k, roots_imag[:, i], label=f'Root {i+1}')
plt.xscale('log')
plt.xlabel('Wavenumber k (m^-1)')
plt.ylabel('Im(omega) / 2Omega')
plt.title('Imaginary part of Dispersion Relation Roots')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('dispersion.png')

print("omega_A2:", omega_A2)
print("Omega:", Omega)
