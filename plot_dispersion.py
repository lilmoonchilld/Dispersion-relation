import numpy as np
import mpmath
import matplotlib.pyplot as plt

# Set mpmath precision
mpmath.mp.dps = 30  # Double precision equivalent plus some buffer, can be adjusted

# Physical parameters
Omega = 2.0         # rad s^-1
r0 = 0.5            # m
H0 = 0.05           # m
rho0 = 1000.0       # kg m^-3
B0 = 0.003          # T
C = 0.05            # m^3 s^-2
mu0 = 4 * np.pi * 1e-7 # T m A^-1

# Derived parameters
c0 = 0.3            # m s^-1 (sqrt(g*H0), though we don't use g explicitly here)
Omega_A_sq = (B0**2) / (mu0 * rho0 * H0**2) # ~ 2.86 s^-2
f = 4.0             # s^-1 (2*Omega)

# Structural parameter
alpha = (Omega**2 + 2*C/(r0**3)) / (Omega**2 * r0)

# Boundaries
epsilon = 0.1
r1 = 0.45
r2 = 0.55

def get_hat_omega(omega):
    """Calculate effective frequency hat_omega."""
    return omega + Omega_A_sq / omega

def get_beta(omega, hat_omega):
    """Calculate beta parameter."""
    return omega * (hat_omega**2 - f**2) / (hat_omega * Omega**2 * r0 * H0)

def get_k(beta):
    """Calculate k scaling parameter."""
    # Using mpmath for complex square root
    return mpmath.sqrt(alpha**2 / 4.0 - beta)

def get_z(r, k):
    """Calculate dimensionless coordinate z."""
    return 2.0 * k * r

def get_kappa(m, hat_omega, k):
    """Calculate Whittaker index kappa."""
    return (alpha / (2.0 * k)) * (0.5 + m * f / hat_omega)

def get_Gamma(m, hat_omega):
    """Calculate boundary operator factor Gamma."""
    return 0.5 + m * f / hat_omega

def boundary_M(z, k, kappa, mu, Gamma):
    """Evaluate boundary operator on Whittaker M."""
    M_val = mpmath.whitm(kappa, mu, z)
    M_prev = mpmath.whitm(kappa - 1, mu, z)

    term1 = (z/2.0 * (1.0 + alpha/(2.0*k)) - kappa - Gamma) * M_val
    term2 = (kappa + mu - 0.5) * M_prev
    return (term1 + term2) / z

def boundary_W(z, k, kappa, mu, Gamma):
    """Evaluate boundary operator on Whittaker W."""
    W_val = mpmath.whitw(kappa, mu, z)
    W_next = mpmath.whitw(kappa + 1, mu, z)

    term1 = (z/2.0 * (1.0 + alpha/(2.0*k)) - kappa - Gamma) * W_val
    term2 = W_next
    return (term1 - term2) / z

def D_determinant(omega_complex, m):
    """Calculate the global dispersion relation determinant D(omega, m)."""
    # Exclude points too close to origin
    if abs(omega_complex) < 1e-4:
        return mpmath.mpc(float('nan'), float('nan'))

    omega = mpmath.mpc(omega_complex)
    hat_omega = get_hat_omega(omega)
    beta = get_beta(omega, hat_omega)
    k = get_k(beta)

    z1 = get_z(r1, k)
    z2 = get_z(r2, k)

    kappa = get_kappa(m, hat_omega, k)
    mu = m  # As per derivation: mu = m
    Gamma = get_Gamma(m, hat_omega)

    # Check if k is too small (causing z1, z2 to approach 0, which might be singular)
    if abs(k) < 1e-10:
        return mpmath.mpc(float('nan'), float('nan'))

    B_z1_M = boundary_M(z1, k, kappa, mu, Gamma)
    B_z1_W = boundary_W(z1, k, kappa, mu, Gamma)

    B_z2_M = boundary_M(z2, k, kappa, mu, Gamma)
    B_z2_W = boundary_W(z2, k, kappa, mu, Gamma)

    D = B_z1_M * B_z2_W - B_z2_M * B_z1_W
    return D

def generate_heatmap(m, re_bounds, im_bounds, grid_size, title, filename):
    print(f"Generating heatmap for m={m}, Re: {re_bounds}, Im: {im_bounds}")

    re_vals = np.linspace(re_bounds[0], re_bounds[1], grid_size)
    im_vals = np.linspace(im_bounds[0], im_bounds[1], grid_size)

    Re, Im = np.meshgrid(re_vals, im_vals)
    Z_abs = np.zeros_like(Re, dtype=float)

    # Avoid exact origin
    for i in range(grid_size):
        for j in range(grid_size):
            omega_val = complex(Re[i, j], Im[i, j])
            if abs(omega_val) < 1e-4:
                Z_abs[i, j] = np.nan
            else:
                try:
                    d_val = D_determinant(omega_val, m)
                    if d_val != d_val: # Check for NaN
                         Z_abs[i, j] = np.nan
                    else:
                         Z_abs[i, j] = float(mpmath.fabs(d_val))
                except Exception as e:
                    Z_abs[i, j] = np.nan

    plt.figure(figsize=(10, 8))
    # We plot log of absolute value to better visualize minima across orders of magnitude
    # Using np.log10 directly is fine since we filtered nans
    log_Z = np.log10(np.where(Z_abs == 0, 1e-15, Z_abs))

    plt.contourf(Re, Im, log_Z, levels=50, cmap='viridis')
    plt.colorbar(label='log10 |D(omega, m)|')
    plt.xlabel('Re(omega)')
    plt.ylabel('Im(omega)')
    plt.title(title)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    # Kelvin/Poincare modes grid
    generate_heatmap(m=1, re_bounds=[-5, 5], im_bounds=[-1, 1], grid_size=100,
                    title='Determinant Magnitude |D| for m=1 (Kelvin/Poincare)',
                    filename='heatmap_m1_kelvin_poincare.png')

    generate_heatmap(m=2, re_bounds=[-5, 5], im_bounds=[-1, 1], grid_size=100,
                    title='Determinant Magnitude |D| for m=2 (Kelvin/Poincare)',
                    filename='heatmap_m2_kelvin_poincare.png')

    # Rossby modes grid
    generate_heatmap(m=1, re_bounds=[-0.5, 0.5], im_bounds=[-1, 1], grid_size=100,
                    title='Determinant Magnitude |D| for m=1 (Rossby)',
                    filename='heatmap_m1_rossby.png')

    generate_heatmap(m=2, re_bounds=[-0.5, 0.5], im_bounds=[-1, 1], grid_size=100,
                    title='Determinant Magnitude |D| for m=2 (Rossby)',
                    filename='heatmap_m2_rossby.png')
