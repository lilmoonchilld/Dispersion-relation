import numpy as np
import matplotlib.pyplot as plt
import os

# Set publication-ready style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

def calculate_dispersion():
    # 1. Physical Parameters
    MU0   = 4 * np.pi * 1e-7
    Omega = 0.5e-4        # rad/s
    H0    = 500.0         # m
    g     = 9.81          # m/s^2
    B0    = 1e-5          # T
    rho0  = 1000.0        # kg/m^3
    C     = 1.5e9         # m^3/s^2

    r1    = 0.5e6         # inner radius [m]
    r2    = 1e6           # outer radius [m]
    r0    = 0.5 * (r1 + r2)

    VA2      = B0**2 / (MU0 * rho0)
    c0sq     = g * H0
    f_scale  = 2.0 * Omega

    # 2. Dimensionless Parameters
    hat_r1 = r1 / r0
    hat_r2 = r2 / r0
    hat_c0sq = c0sq / (Omega**2 * r0**2)
    gamma = 2.0 * C / (Omega**2 * r0**3)
    hat_omA = np.sqrt(VA2) / (f_scale * H0)
    hat_omA2 = hat_omA**2

    # Grid for spatial averaging
    num_r = 100
    r_grid = np.linspace(hat_r1, hat_r2, num_r)
    dr = r_grid[1] - r_grid[0]

    m_vals = np.arange(0, 31)
    n_vals = [1, 2, 3, 4]

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for idx, n in enumerate(n_vals):
        ax = axes[idx]
        hat_kr = n * np.pi / (hat_r2 - hat_r1)

        # Store averaged roots for each m
        # Expecting 4 roots per m since it's a quartic in omega
        avg_roots_m = []

        for m in m_vals:
            local_roots = []

            for r in r_grid:
                # Calculate functions at r
                R_r = (1 + gamma) * (2*r - 1) / r
                S_r = m * (1 + gamma) * (r - 1) / r
                kappa2_r = hat_kr**2 + (m**2) / (r**2)

                K_r = hat_c0sq * kappa2_r + R_r

                # Equation: 4*omega*(omega_*^2 - 1) = omega_* * K_r + S_r
                # where omega_* = omega + hat_omA2 / omega
                # Expanding this gives a quartic in omega:
                # C4 * omega^4 + C3 * omega^3 + C2 * omega^2 + C1 * omega + C0 = 0

                # Derivation of coefficients:
                # LHS = 4 * omega * [ (omega + hat_omA2/omega)^2 - 1 ]
                # LHS = 4 * omega * [ omega^2 + 2*hat_omA2 + hat_omA2^2/omega^2 - 1 ]
                # LHS = 4*omega^3 + 8*hat_omA2*omega + 4*hat_omA2^2/omega - 4*omega

                # RHS = (omega + hat_omA2/omega) * K_r + S_r
                # RHS = omega * K_r + hat_omA2 * K_r / omega + S_r

                # Equating and multiplying by omega:
                # 4*omega^4 + 8*hat_omA2*omega^2 + 4*hat_omA2^2 - 4*omega^2 = omega^2 * K_r + hat_omA2 * K_r + S_r * omega
                # 4*omega^4 + (8*hat_omA2 - 4 - K_r)*omega^2 - S_r * omega + (4*hat_omA2^2 - hat_omA2 * K_r) = 0

                C4 = 4.0
                C3 = 0.0
                C2 = 8.0 * hat_omA2 - 4.0 - K_r
                C1 = -S_r
                C0 = 4.0 * hat_omA2**2 - hat_omA2 * K_r

                roots = np.roots([C4, C3, C2, C1, C0])

                # Sort roots by real part for consistency across r to track branches
                roots_sorted = np.sort(roots) # Sort complex numbers? usually sorts by real part
                local_roots.append(roots_sorted)

            # Convert to array of shape (num_r, 4)
            local_roots = np.array(local_roots)

            # Better branch tracking: sort by distance from previous r
            tracked_roots = np.zeros_like(local_roots)
            tracked_roots[0] = local_roots[0]
            for i in range(1, num_r):
                # Simple distance matching
                prev = tracked_roots[i-1]
                curr = local_roots[i]

                # Distance matrix
                dist = np.abs(curr[:, None] - prev[None, :])

                # Assign to closest
                assigned_curr = np.zeros(4, dtype=complex)
                used_j = set()
                for j in range(4):
                    # Find min in dist
                    min_idx = np.unravel_index(np.argmin(dist), dist.shape)
                    assigned_curr[min_idx[1]] = curr[min_idx[0]]
                    # Invalidate row and col
                    dist[min_idx[0], :] = np.inf
                    dist[:, min_idx[1]] = np.inf

                tracked_roots[i] = assigned_curr

            # Average over r
            avg_roots = np.mean(tracked_roots, axis=0)
            avg_roots_m.append(avg_roots)

        avg_roots_m = np.array(avg_roots_m)

        # Plot each branch
        colors = ['blue', 'orange', 'green', 'red']
        for b in range(4):
            # Extract real parts
            real_parts = np.real(avg_roots_m[:, b])
            # Only plot if the imaginary part is small (i.e. physical mode)
            # Some roots might be purely complex or damped, let's plot real parts regardless
            label_str = f'Branch {b+1}'
            ax.plot(m_vals, real_parts, marker='o', markersize=4, color=colors[b], label=f'Branch {b+1}')

        ax.set_title(f'$n = {n}$')
        ax.set_ylabel(r'$\mathrm{Re}(\hat{\omega})$')
        ax.grid(True, linestyle=':', alpha=0.6)

    axes[2].set_xlabel('$m$')
    axes[3].set_xlabel('$m$')

    # Legend for the first subplot
    axes[0].legend(fontsize=12, loc='best')

    plt.tight_layout()
    plt.savefig('outputs/dispersion.png', dpi=300)
    print("Plot saved to outputs/dispersion.png")

if __name__ == "__main__":
    calculate_dispersion()
