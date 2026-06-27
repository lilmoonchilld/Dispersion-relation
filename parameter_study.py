import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det

# =============================================================================
# Physical parameters
# =============================================================================
MU0   = 4 * np.pi * 1e-7
Omega = 1.0
H0    = 0.05
g     = 9.81
B0    = 0.01
rho0  = 1000.0

r1    = 0.3
r2    = 0.8
r0    = 0.5 * (r1 + r2)
m     = 1
N     = 64

# =============================================================================
# Core functions
# =============================================================================
def chebyshev_lobatto(N, r1, r2):
    j  = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N)
    c[0] = 2.0; c[-1] = 2.0
    X = np.tile(xi, (N, 1))
    dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    scale = 2.0 / (r2 - r1)
    D1 = scale * D
    D2 = D1 @ D1
    r_phys = 0.5 * (r1 + r2) + 0.5 * (r2 - r1) * xi
    return r_phys[::-1], D1[::-1, ::-1], D2[::-1, ::-1]

r_grid, D1, D2 = chebyshev_lobatto(N, r1, r2)

def compute_parameters(C, B_0=B0, Omg=Omega):
    VA2 = B_0**2 / (MU0 * rho0)
    omA2 = VA2 / H0**2
    beta_g = 2.0 * C / r0**3
    beta_eff = Omg**2 + beta_g
    c0sq = g * H0
    return omA2, beta_eff, c0sq

def build_matrix(omega, C, B_0=B0, Omg=Omega):
    omA2, beta_eff, c0sq = compute_parameters(C, B_0, Omg)
    ws = omega + omA2 / omega

    P_vec = 1.0 / r_grid - beta_eff * (r_grid - r0) / c0sq

    Ds = 4.0 * Omg**2 - ws**2
    Q_vec = -omega * Ds / (ws * c0sq) - m**2 / r_grid**2 - beta_eff * (2.0 * r_grid - r0) / (c0sq * r_grid) - 2.0 * Omg * m * beta_eff * (r_grid - r0) / (ws * c0sq * r_grid)

    L = D2 + np.diag(P_vec) @ D1 + np.diag(Q_vec)

    for idx in [0, N - 1]:
        r_bc = r_grid[idx]
        bc_eta_coeff = -(ws * beta_eff * (r_bc - r0) + 2.0 * Omg * m * c0sq / r_bc)
        L[idx, :] = ws * c0sq * D1[idx, :] + bc_eta_coeff * np.eye(N)[idx]
    return L

def det_L(omega, C, B_0=B0, Omg=Omega):
    return det(build_matrix(omega, C, B_0, Omg))

def scan_and_polish(C, B_0=B0, Omg=Omega, omega_range=(-20, 20), n_scan=1200, tol=1e-10, max_iter=50):
    omega_min, omega_max = omega_range
    omega_scan = np.linspace(omega_min, omega_max, n_scan)

    # Exclude tiny values that cause zero division
    omega_scan = omega_scan[np.abs(omega_scan) > 1e-3]

    log_det = []
    for w in omega_scan:
        try:
            d = np.log(np.abs(det_L(w, C, B_0, Omg)) + 1e-300)
        except Exception:
            d = np.nan
        log_det.append(d)
    log_det = np.array(log_det)

    roots = []
    for k in range(1, len(log_det) - 1):
        if (log_det[k] < log_det[k - 1]) and (log_det[k] < log_det[k + 1]):
            w0 = omega_scan[k]
            w = complex(w0)
            for _ in range(max_iter):
                if abs(w) < 1e-4: break
                dw = 1e-6 * abs(w) + 1e-10
                f0 = det_L(w, C, B_0, Omg)
                try:
                    fp = (det_L(w + dw, C, B_0, Omg) - det_L(w - dw, C, B_0, Omg)) / (2 * dw)
                except Exception:
                    break
                if abs(fp) < 1e-300 or np.isnan(abs(fp)): break
                step = -f0 / fp
                w += step
                if abs(step) < tol * (abs(w) + 1e-10): break

            if abs(w) > 1e-4 and abs(det_L(w, C, B_0, Omg)) < 1e-4 * np.exp(np.nanmax(log_det[np.isfinite(log_det)])):
                is_dup = any(abs(w - wr) < 1e-4 * abs(w) + 1e-8 for wr in roots)
                if not is_dup:
                    roots.append(w)
    return np.array([wr.real for wr in roots])

# =============================================================================
# Parameter Study
# =============================================================================
if __name__ == '__main__':
    C_max = 2.0 * Omega**2 * r0**3
    C_vals = np.linspace(0, C_max, 40)
    C_norm = C_vals / (Omega**2 * r0**3)

    rossby_branch = []
    magneto_branch = []
    beta_eff_list = []

    for C_val in C_vals:
        _, beff, _ = compute_parameters(C_val)
        beta_eff_list.append(beff)

        roots = scan_and_polish(C_val)
        slow_roots = np.sort(roots[np.abs(roots) < 3.0])

        if len(slow_roots) >= 2:
            # Typically Rossby is negative and MS is positive or less negative
            # At C=0.5 they were ~ -1.19 and ~ +1.17
            m1 = slow_roots[0]
            m2 = slow_roots[1] if len(slow_roots) > 1 else np.nan
            rossby_branch.append(m1)
            magneto_branch.append(m2)
        else:
            rossby_branch.append(np.nan)
            magneto_branch.append(np.nan)

    rossby_branch = np.array(rossby_branch)
    magneto_branch = np.array(magneto_branch)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axs[0]
    ax1.plot(C_norm, rossby_branch, 'b-o', label='Rossby Branch (Retrograde)', markersize=4)
    ax1.plot(C_norm, magneto_branch, 'r-s', label='Magnetostrophic Branch', markersize=4)
    ax1.set_xlabel(r'$C / (\Omega^2 r_0^3)$')
    ax1.set_ylabel(r'$\omega$ [rad/s]')
    ax1.set_title(r'Slow Wave Branches vs Radial Gravity Strength $C$')
    ax1.legend()
    ax1.grid(True)

    ax2 = axs[1]
    ax2.plot(C_norm, beta_eff_list, 'k-', lw=2)
    ax2.set_xlabel(r'$C / (\Omega^2 r_0^3)$')
    ax2.set_ylabel(r'$\beta_{eff}$ [s$^{-2}$]')
    ax2.set_title(r'Effective Beta Parameter vs $C$')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/parameter_study.pdf')
    print("Parameter study saved to outputs/parameter_study.pdf")
