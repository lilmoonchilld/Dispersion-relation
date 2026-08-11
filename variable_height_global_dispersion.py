"""
Variable-Height SWMHD Global Dispersion Solver
===============================================
Solves the global eigenvalue problem for linearized variable-depth SWMHD in physical coordinates (SI units).
"""

import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# =============================================================================
# Style & Conventions for Publication Quality Plots
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

# =============================================================================
# 0. Physical Parameters (SI Units)
# =============================================================================
MU0   = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]

# Definitive parameters as requested
Omega = 0.5e-4        # Rotation rate [rad/s]
H0    = 500.0         # Reference depth [m]
g     = 9.81          # Gravitational acceleration [m/s^2]
B0    = 0.0           # External vertical magnetic field [T]
rho0  = 1000.0        # Density [kg/m^3]
C     = 0.0           # Radial gravity constant [m^3/s^2]

r1    = 0.5e6         # Inner radius [m]
r2    = 1.0e6         # Outer radius [m]
r0    = 0.5 * (r1 + r2) # Reference radius [m]

f     = 2.0 * Omega   # Coriolis parameter [rad/s]

# Validation switches
USE_VARIABLE_DEPTH = True

# =============================================================================
# 1. Equilibrium Depth
# =============================================================================
def H_eq(r):
    if not USE_VARIABLE_DEPTH:
        return np.ones_like(r) * H0 if isinstance(r, np.ndarray) else H0
    return H0 + (Omega**2 / (2.0 * g)) * (r**2 - r0**2) + (C / g) * (1.0 / r - 1.0 / r0)

def dH_eq_dr(r):
    if not USE_VARIABLE_DEPTH:
        return np.zeros_like(r) if isinstance(r, np.ndarray) else 0.0
    return (Omega**2 / g) * r - C / (g * r**2)

# =============================================================================
# 2. Magnetic Frequency & Modified Frequency
# =============================================================================
def omega_A2(r):
    if B0 == 0.0:
        return np.zeros_like(r) if isinstance(r, np.ndarray) else 0.0
    return B0**2 / (MU0 * rho0 * H_eq(r)**2)

def omega_star(omega, r):
    om_A2 = omega_A2(r)
    return omega + om_A2 / omega

def domega_star_dr(omega, r):
    if B0 == 0.0:
        return np.zeros_like(r) if isinstance(r, np.ndarray) else 0.0
    # d(omega^*)/dr = -2 * omega_A^2(r) / (omega * H_eq(r)) * H_eq'(r)
    om_A2 = omega_A2(r)
    Heq = H_eq(r)
    Heqp = dH_eq_dr(r)
    return -2.0 * om_A2 * Heqp / (omega * Heq)

# =============================================================================
# 3. Dimensional Chebyshev Collocation
# =============================================================================
def chebyshev_lobatto(N, r_min, r_max):
    """
    Constructs the dimensional Chebyshev grid on [r_min, r_max] (in meters),
    returning nodes, first derivative matrix D1 (1/m), and second derivative matrix D2 (1/m^2).
    """
    j = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    X = np.tile(xi, (N, 1))
    dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))

    # Scale matrix to [r_min, r_max]
    scale = 2.0 / (r_max - r_min)
    D1 = scale * D
    D2 = D1 @ D1

    # Map physical nodes (ascending: r[0] = r_min, r[-1] = r_max)
    r_grid = 0.5 * (r_min + r_max) + 0.5 * (r_max - r_min) * xi
    r_grid = r_grid[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]

    return r_grid, D1, D2

# =============================================================================
# 4. Radial ODE Coefficients
# =============================================================================
def radial_coefficients(omega, m, r_grid):
    Heq = H_eq(r_grid)
    Heqp = dH_eq_dr(r_grid)
    ws = omega_star(omega, r_grid)
    wsp = domega_star_dr(omega, r_grid)

    ws2_minus_f2 = ws**2 - f**2

    C2 = g * Heq * ws / ws2_minus_f2

    num_C1 = Heq * ws + r_grid * Heqp * ws + r_grid * Heq * wsp
    term1_C1 = num_C1 / ws2_minus_f2
    term2_C1 = 2.0 * r_grid * Heq * (ws**2) * wsp / (ws2_minus_f2**2)
    C1 = (g / r_grid) * (term1_C1 - term2_C1)

    term1_C0 = Heqp / ws2_minus_f2
    term2_C0 = 2.0 * Heq * ws * wsp / (ws2_minus_f2**2)
    C0 = omega - (m * g * f / r_grid) * (term1_C0 - term2_C0) - (m**2 * g * Heq * ws) / (r_grid**2 * ws2_minus_f2)

    return C2, C1, C0

# =============================================================================
# 5. Global Matrix Assembly
# =============================================================================
def build_matrix(omega, m, N, r_grid, D1_mat, D2_mat):
    # Protection against zero and resonance singularities
    omega_tol = 1e-12
    resonance_tol = 1e-12

    if np.abs(omega) < omega_tol:
        return None

    ws = omega_star(omega, r_grid)
    if np.any(np.abs(ws**2 - f**2) < resonance_tol):
        return None

    C2, C1, C0 = radial_coefficients(omega, m, r_grid)

    # Matrix differential operator
    L = np.diag(C2) @ D2_mat + np.diag(C1) @ D1_mat + np.diag(C0)

    # Boundaries: v_r = 0. Replace row 0 (r1) and row N-1 (r2)
    for idx in [0, N - 1]:
        rb = r_grid[idx]
        ws_b = omega_star(omega, rb)
        L[idx, :] = ws_b * D1_mat[idx, :] - (m * f / rb) * np.eye(N)[idx]

    return L

# =============================================================================
# 6. Global Eigenvalue Search
# =============================================================================
def find_eigenvalues(m_val, N, r_grid, D1_mat, D2_mat, omega_range=(-5e-3, 5e-3), n_scan=2500):
    """
    Scans the dimensional frequency range for sign changes of det(L(omega)),
    and isolates precise roots using scipy.optimize.brentq.
    """
    scan = np.linspace(*omega_range, n_scan)
    # Exclude omega = 0
    scan = scan[np.abs(scan) > 1e-12]

    signs = []
    log_dets = []

    for w in scan:
        L = build_matrix(w, m_val, N, r_grid, D1_mat, D2_mat)
        if L is None:
            signs.append(np.nan)
            log_dets.append(np.nan)
        else:
            try:
                sign, log_det = np.linalg.slogdet(L)
                signs.append(sign)
                log_dets.append(log_det)
            except Exception:
                signs.append(np.nan)
                log_dets.append(np.nan)

    signs = np.array(signs)
    log_dets = np.array(log_dets)

    roots = []

    # Detect sign changes and solve precisely with brentq
    for i in range(len(scan) - 1):
        if np.isnan(signs[i]) or np.isnan(signs[i+1]):
            continue
        if signs[i] * signs[i+1] < 0:
            # Scale function internally to prevent brentq underflow/overflow
            ref_log_det = log_dets[i]

            def f_to_solve(w):
                L = build_matrix(w, m_val, N, r_grid, D1_mat, D2_mat)
                if L is None:
                    return np.nan
                sign, log_det = np.linalg.slogdet(L)
                return sign * np.exp(log_det - ref_log_det)

            try:
                root = brentq(f_to_solve, scan[i], scan[i+1], xtol=1e-15)
                # Filter duplicates
                if not any(np.abs(root - r) < 1e-7 * np.abs(root) + 1e-8 for r in roots):
                    roots.append(root)
            except Exception:
                pass

    return np.array(sorted(roots))

# =============================================================================
# Helper function to recover normalized eigenfunction eta
# =============================================================================
def get_eigenfunction(omega, m, N, r_grid, D1_mat, D2_mat):
    L = build_matrix(omega, m, N, r_grid, D1_mat, D2_mat)
    if L is None:
        return None
    _, _, Vh = la.svd(L)
    eta = Vh[-1, :].conj()
    eta /= np.max(np.abs(eta))
    return eta

# =============================================================================
# 11. Validation and Numerical Tests (Printed to Terminal)
# =============================================================================
def run_validation_checks():
    print("==========================================================")
    print("               CRITICAL NUMERICAL TESTS                   ")
    print("==========================================================")

    # Test A: Depth consistency
    Heq_r0 = H_eq(r0)
    print(f"Test A [Depth Consistency]: H_eq(r0) = {Heq_r0:.6f} m (Expected: {H0:.6f} m)")
    assert np.abs(Heq_r0 - H0) < 1e-10, "Depth consistency failed!"

    # Test B: Derivative consistency
    # Compare analytic derivative against second-order central finite difference
    dr = 1.0 # 1 meter step
    analytic_deriv = dH_eq_dr(r0)
    fd_deriv = (H_eq(r0 + dr) - H_eq(r0 - dr)) / (2.0 * dr)
    diff = np.abs(analytic_deriv - fd_deriv)
    print(f"Test B [Derivative Consistency]: Analytic H_eq'(r0) = {analytic_deriv:.6e}, FD = {fd_deriv:.6e}, Diff = {diff:.6e}")
    assert diff < 1e-6, "Derivative consistency failed!"

    # Test C: Magnetic derivative (if B0 > 0)
    if B0 > 0.0:
        w_test = 1e-4
        analytic_wsp = domega_star_dr(w_test, r0)
        fd_wsp = (omega_star(w_test, r0 + dr) - omega_star(w_test, r0 - dr)) / (2.0 * dr)
        diff_ws = np.abs(analytic_wsp - fd_wsp)
        print(f"Test C [Magnetic Derivative]: domega*/dr Analytic = {analytic_wsp:.6e}, FD = {fd_wsp:.6e}, Diff = {diff_ws:.6e}")
        assert diff_ws < 1e-6, "Magnetic derivative consistency failed!"
    else:
        print("Test C [Magnetic Derivative]: B0 = 0.0, skipped or trivially verified.")

    # Test D: Boundary condition & Test E: ODE residual
    # Perform check for a sample eigenvalue of m=1
    N_sample = 64
    r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_sample, r1, r2)
    roots = find_eigenvalues(1, N_sample, r_grid, D1_mat, D2_mat)

    if len(roots) > 0:
        # Choose a fast mode to verify
        idx_test = -1 # Largest positive eigenvalue
        w_test = roots[idx_test]
        eta = get_eigenfunction(w_test, 1, N_sample, r_grid, D1_mat, D2_mat)

        # Test D: Boundary condition residual
        # BC: ws * D1 * eta - (m * f / rb) * eta = 0
        for b_name, b_idx in [("inner", 0), ("outer", -1)]:
            rb = r_grid[b_idx]
            ws_b = omega_star(w_test, rb)
            val = ws_b * (D1_mat[b_idx, :] @ eta) - (1.0 * f / rb) * eta[b_idx]
            print(f"Test D [Boundary Condition at {b_name}]: Robin BC residual = {np.abs(val):.6e}")

        # Test E: ODE residual
        C2, C1, C0 = radial_coefficients(w_test, 1, r_grid)
        # Check interior points (indices 1 to N_sample-2)
        ode_res = C2 * (D2_mat @ eta) + C1 * (D1_mat @ eta) + C0 * eta
        # L2 norm of residual at interior points
        interior_res = ode_res[1:-1]
        l2_res = np.sqrt(np.sum(np.abs(interior_res)**2) / (N_sample - 2))
        print(f"Test E [ODE Residual]: L2 norm of interior ODE residual = {l2_res:.6e}")
    else:
        print("No roots found for Test D/E.")

    # Test F: Constant-depth limit stability under increasing N_col
    global USE_VARIABLE_DEPTH
    USE_VARIABLE_DEPTH = False
    print("\nRunning Test F: Constant-depth stability scan...")
    for N_test in [32, 64]:
        rg, d1, d2 = chebyshev_lobatto(N_test, r1, r2)
        rts_test = find_eigenvalues(1, N_test, rg, d1, d2)
        print(f"  N = {N_test}: found {len(rts_test)} roots. Sample roots (hat_omega):")
        for rt in rts_test[:4]:
            print(f"    {rt / f:.6f}")

    # Restore variable depth
    USE_VARIABLE_DEPTH = True
    print()

    # Test G: Resolution convergence check for m = 1 (Variable Depth)
    print("==========================================================")
    print("          TEST G: RESOLUTION CONVERGENCE (m=1)            ")
    print("==========================================================")
    resolutions = [32, 48, 64, 80]
    results_conv = {}
    for N_res in resolutions:
        rg, d1, d2 = chebyshev_lobatto(N_res, r1, r2)
        rts = find_eigenvalues(1, N_res, rg, d1, d2)
        results_conv[N_res] = rts
        print(f"  N = {N_res:2d}: found {len(rts)} real eigenfrequencies.")

    print("\nConvergence table for selected large-scale physical wave modes (hat_omega):")
    # We select reference roots (N=80) that correspond to the main physical gravity-inertial modes (e.g. magnitude > 3*f)
    ref_roots_phys = sorted([r for r in results_conv[80] if np.abs(r) > 3.0 * f], key=lambda x: np.abs(x))
    print(f"{'N':>4} | " + " | ".join(f"Mode {i+1}" for i in range(min(5, len(ref_roots_phys)))))
    print("-" * 75)
    for N_res in resolutions:
        row_str = f"{N_res:4d} |"
        rts = results_conv[N_res]
        for i in range(min(5, len(ref_roots_phys))):
            ref_rt = ref_roots_phys[i]
            if len(rts) > 0:
                best_idx = np.argmin(np.abs(rts - ref_rt))
                best_rt = rts[best_idx]
                # Check distance
                dist = np.abs(best_rt - ref_rt) / f
                if dist < 0.05:
                    row_str += f" {best_rt/f:+.6f} |"
                else:
                    row_str += f" {'-':^10} |"
            else:
                row_str += f" {'-':^10} |"
        print(row_str)
    print("==========================================================\n")

# =============================================================================
# 9. Spectrum & Dispersion Curves over m = 1..30
# =============================================================================
def compute_dispersion_relation():
    print("==========================================================")
    print("        COMPUTING GLOBAL DISPERSION RELATION              ")
    print("==========================================================")

    M_max = 30
    m_arr = np.arange(1, M_max + 1)
    N_col = 64

    r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, r1, r2)

    dispersion_data = {}

    for m in m_arr:
        eigs = find_eigenvalues(m, N_col, r_grid, D1_mat, D2_mat, omega_range=(-50 * f, 50 * f))
        dispersion_data[m] = eigs

        # Lightweight print to terminal
        pos = sorted([e for e in eigs if e > 0])
        neg = sorted([e for e in eigs if e < 0], reverse=True)
        if m <= 3 or m % 5 == 0:
            print(f"  m={m:2d} (found {len(eigs)} roots): "
                  f"+{[f'{x/f:.3f}' for x in pos[:4]]}... "
                  f"-{[f'{abs(x)/f:.3f}' for x in neg[:4]]}...")

    return m_arr, dispersion_data

# =============================================================================
# 10. Plotting
# =============================================================================
def plot_dispersion_relation(m_arr, dispersion_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.6, 1]})

    fig.suptitle('SWMHD Global Dispersion Relation (Variable-Depth)\n', fontsize=16, fontweight='bold')

    # Full spectrum plot
    for m in m_arr:
        eigs = dispersion_data[m]
        # Plot positive and negative frequencies as scatter points
        ax1.scatter([m] * len(eigs), eigs / f, color='black', marker='o', s=30, alpha=0.85, zorder=5)

    ax1.axhline(0, color='grey', lw=0.8, ls=':')
    ax1.axhline(+1, color='grey', lw=0.9, ls='--', alpha=0.5)
    ax1.axhline(-1, color='grey', lw=0.9, ls='--', alpha=0.5)

    ax1.set_xlabel('Azimuthal wavenumber $m$', fontsize=15)
    ax1.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
    ax1.set_xlim(0.5, len(m_arr) + 0.5)
    ax1.set_ylim(-50.0, 50.0)
    ax1.set_xticks(m_arr[::2])
    ax1.grid(True, alpha=0.22)
    ax1.set_title('Full Spectrum', fontsize=14)

    # Slow branches zoom plot
    for m in m_arr:
        eigs = dispersion_data[m]
        slow_eigs = eigs[np.abs(eigs / f) < 1.0]
        ax2.scatter([m] * len(slow_eigs), slow_eigs / f, color='black', marker='o', s=45, alpha=0.85, zorder=5)

    ax2.axhline(0, color='grey', lw=0.8, ls=':')
    ax2.set_xlabel('Azimuthal wavenumber $m$', fontsize=15)
    ax2.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
    ax2.set_xlim(0.5, len(m_arr) + 0.5)
    ax2.set_ylim(-0.8, 0.2)
    ax2.set_xticks(m_arr[::2])
    ax2.grid(True, alpha=0.22)
    ax2.set_title('Slow Branches Zoom', fontsize=14)

    # Legend for collocation dots
    col_dot = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Global eigenvalues (collocation)')
    ax1.legend(handles=[col_dot], loc='upper left', fontsize=12)
    ax2.legend(handles=[col_dot], loc='lower left', fontsize=12)

    # Parameter box
    pbox = (
        rf"$\Omega = {Omega:.1e}$ rad/s,  $H_0 = {H0}$ m" + "\n"
        rf"$r_1 = {r1:.1e}$ m,  $r_2 = {r2:.1e}$ m" + "\n"
        rf"$\omega_A = 0$ s$^{{-1}}$ (since $B_0=0$)" + "\n"
        rf"$N = 64$"
    )
    fig.text(0.52, 0.015, pbox, fontsize=11, va='bottom', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))

    fig.subplots_adjust(bottom=0.15)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    plt.savefig("outputs/swmhd_variable_depth_global_dispersion.png", dpi=180, bbox_inches='tight')
    plt.savefig("outputs/swmhd_variable_depth_global_dispersion.pdf", dpi=180, bbox_inches='tight')
    print("Saved plots:")
    print("  - outputs/swmhd_variable_depth_global_dispersion.png")
    print("  - outputs/swmhd_variable_depth_global_dispersion.pdf")

def main():
    run_validation_checks()
    m_arr, dispersion_data = compute_dispersion_relation()
    plot_dispersion_relation(m_arr, dispersion_data)

if __name__ == "__main__":
    main()
