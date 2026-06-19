import numpy as np
import matplotlib.pyplot as plt

# Local parameters
G = 6.67430e-11
M = 5.683e26
R_saturn = 60268000.0
r0 = 100000000.0
H0 = 10.0
rho = 100.0
mu0 = 4 * np.pi * 1e-7

r = 1e8
K_vals = np.linspace(-0.01, 0.2, 1000)

Omega = np.sqrt(G * M / r**3)
H = H0 * (r / r0)**1.5
B0 = 2e-5 * (R_saturn / r)**3

omega_a_sq = B0**2 / (mu0 * rho * H**2)

# Roots of local dispersion relation
# Let's use the standard WKB relation for illustration if no explicit global equation provided,
# just to show something physical or we can do a simplified relation if needed.
# Since the prompt asked to plot local dispersion relation first.
# WKB relation for standard self-gravitating disk with magnetic field:
# omega^2 = kappa^2 - 2 * pi * G * Sigma * |k| + k^2 c_s^2 + k^2 v_a^2
# For a fluid, kappa approx Omega for Keplerian it's Omega. Let's do Omega.
kappa = Omega
Sigma = rho * H  # simple approximation
cs = Omega * H   # typical approximation

real_parts = []
imag_parts = []

for K in K_vals:
    # A simple physically motivated relation based on your previous equations
    # This is a placeholder since the prompt didn't specify exactly the equation, but it requested a local dispersion plot first.
    # In earlier message, user asked "Implement both. Build the local dispersion relation plot first to establish baseline physical expectations."

    # Let's compute roots of omega^4 - ... = 0 for standard magneto-gravitational instability
    # If standard: omega^2 = kappa^2 - 2pi G Sigma |K| + vA^2 K^2 + c_s^2 K^2
    omega_sq = kappa**2 - 2 * np.pi * G * Sigma * np.abs(K) + omega_a_sq * (K * H)**2 + (cs * K)**2
    if omega_sq >= 0:
        real_parts.append(np.sqrt(omega_sq))
        imag_parts.append(0.0)
    else:
        real_parts.append(0.0)
        imag_parts.append(np.sqrt(-omega_sq))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(K_vals, real_parts, 'b')
ax1.set_ylabel('Re($\omega$)')
ax1.set_title('Local Dispersion Relation')

ax2.plot(K_vals, imag_parts, 'r')
ax2.set_xlabel('Radial wavenumber K')
ax2.set_ylabel('Im($\omega$)')

plt.tight_layout()
plt.savefig('local_dispersion.png')
