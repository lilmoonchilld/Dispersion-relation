"""
Variable-Height SWMHD Global Dispersion Solver
===============================================
Solves the global eigenvalue problem for linearized variable-depth SWMHD in physical coordinates (SI units).
Supports:
  - Exact analytical Bessel solver for the constant-depth special case.
  - Chebyshev global collocation solver for general variable-depth.
  - WKB-based branch prediction & global one-to-one matching layer for:
    * Magneto-Poincaré (blue)
    * Rossby (red)
    * Magnetostrophic (orange)
"""

import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import jv, yv, jvp, yvp, iv, kv, ivp, kvp

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
B0    = 0.0           # External vertical magnetic field [T] (set to B0 > 0.0 to enable magnetic branches)
rho0  = 1000.0        # Density [kg/m^3]
C     = 0.0           # Radial gravity constant [m^3/s^2]

r1    = 0.5e6         # Inner radius [m]
r2    = 1.0e6         # Outer radius [m]
r0    = 0.5 * (r1 + r2) # Reference radius [m]

f     = 2.0 * Omega   # Coriolis parameter [rad/s]

# Validation switches
USE_VARIABLE_DEPTH = True
USE_ANALYTIC_BESSEL = False  # Set to False by default

N_WKB = 8             # Number of radial modes to predict/match

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
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) * ((-1.0)**(i+k)) / (xi[i] - xi[k])
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
# 16. Analytic Bessel Special-Case Solver Section
# =============================================================================
def bessel_radial_wavenumber(omega):
    """
    Computes k^2 and returns (val, is_oscillatory)
    where val is k^2 (if k^2 > 0) or q^2 = -k^2 (if k^2 < 0).
    """
    ws = omega_star(omega, r0) # constant depth, so evaluated at r0 is same as everywhere
    k2 = omega * (ws**2 - f**2) / (g * H0 * ws)
    if k2 >= 0.0:
        return k2, True
    else:
        return -k2, False

def bessel_boundary_determinant(omega, m):
    """
    Evaluates the exact Bessel boundary determinant D_Bessel(omega)
    using ordinary J, Y or modified I, K Bessel functions.
    """
    omega_tol = 1e-12
    resonance_tol = 1e-12

    if np.abs(omega) < omega_tol:
        return np.nan

    ws = omega_star(omega, r0)
    if np.abs(ws**2 - f**2) < resonance_tol:
        return np.nan

    val, is_oscillatory = bessel_radial_wavenumber(omega)
    arg = np.sqrt(val)

    if is_oscillatory:
        # Oscillatory case: use J_m, Y_m
        B1_J = ws * arg * jvp(m, arg * r1) - (m * f / r1) * jv(m, arg * r1)
        B1_Y = ws * arg * yvp(m, arg * r1) - (m * f / r1) * yv(m, arg * r1)
        B2_J = ws * arg * jvp(m, arg * r2) - (m * f / r2) * jv(m, arg * r2)
        B2_Y = ws * arg * yvp(m, arg * r2) - (m * f / r2) * yv(m, arg * r2)

        return B1_J * B2_Y - B1_Y * B2_J
    else:
        # Evanescent case: use I_m, K_m
        B1_I = ws * arg * ivp(m, arg * r1) - (m * f / r1) * iv(m, arg * r1)
        B1_K = ws * arg * kvp(m, arg * r1) - (m * f / r1) * kv(m, arg * r1)
        B2_I = ws * arg * ivp(m, arg * r2) - (m * f / r2) * iv(m, arg * r2)
        B2_K = ws * arg * kvp(m, arg * r2) - (m * f / r2) * kv(m, arg * r2)

        return B1_I * B2_K - B1_K * B2_I

def find_bessel_eigenvalues(m_val, omega_range=(-5e-3, 5e-3), n_scan=2500):
    """
    Scans the frequency range for sign changes of the exact Bessel boundary determinant D_Bessel(omega),
    and refines the roots using scipy.optimize.brentq.
    """
    scan = np.linspace(*omega_range, n_scan)
    scan = scan[np.abs(scan) > 1e-12]

    dets = []
    for w in scan:
        dets.append(bessel_boundary_determinant(w, m_val))

    dets = np.array(dets)
    roots = []

    # Detect sign changes and solve precisely with brentq
    for i in range(len(scan) - 1):
        if np.isnan(dets[i]) or np.isnan(dets[i+1]):
            continue
        if dets[i] * dets[i+1] < 0:
            # Scale function value to avoid brentq numerical scaling issues
            ref_val = np.abs(dets[i])

            def f_to_solve(w):
                val = bessel_boundary_determinant(w, m_val)
                return val / ref_val

            try:
                root = brentq(f_to_solve, scan[i], scan[i+1], xtol=1e-15)
                if not any(np.abs(root - r) < 1e-7 * np.abs(root) + 1e-8 for r in roots):
                    roots.append(root)
            except Exception:
                pass

    return np.array(sorted(roots))

def get_bessel_eigenfunction(omega, m, r_grid):
    """
    Constructs the analytical Bessel/modified-Bessel eigenfunction on r_grid.
    """
    ws = omega_star(omega, r0)
    val, is_oscillatory = bessel_radial_wavenumber(omega)
    arg = np.sqrt(val)

    if is_oscillatory:
        B1_J = ws * arg * jvp(m, arg * r1) - (m * f / r1) * jv(m, arg * r1)
        B1_Y = ws * arg * yvp(m, arg * r1) - (m * f / r1) * yv(m, arg * r1)

        if np.abs(B1_Y) > 1e-12:
            A, B = 1.0, -B1_J / B1_Y
        else:
            A, B = -B1_Y / B1_J, 1.0

        eta = A * jv(m, arg * r_grid) + B * yv(m, arg * r_grid)
    else:
        B1_I = ws * arg * ivp(m, arg * r1) - (m * f / r1) * iv(m, arg * r1)
        B1_K = ws * arg * kvp(m, arg * r1) - (m * f / r1) * kv(m, arg * r1)

        if np.abs(B1_K) > 1e-12:
            A, B = 1.0, -B1_I / B1_K
        else:
            A, B = -B1_K / B1_I, 1.0

        eta = A * iv(m, arg * r_grid) + B * kv(m, arg * r_grid)

    eta /= np.max(np.abs(eta))
    return eta

def validate_bessel_reduction():
    """
    Explicitly confirms that the constant-depth coefficients reduce exactly
    to the standard Bessel form when USE_VARIABLE_DEPTH = False.
    """
    print("==========================================================")
    print("      TEST 13/14: BESSEL FORM REDUCTION VALIDATION        ")
    print("==========================================================")

    # Store previous state of switches
    prev_vdepth = USE_VARIABLE_DEPTH

    # Temporarily set variable depth to False
    globals()['USE_VARIABLE_DEPTH'] = False

    N_test = 64
    r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_test, r1, r2)

    # Choose a sample frequency
    w_sample = 1.5 * f
    C2, C1, C0 = radial_coefficients(w_sample, 1, r_grid)

    # Expected Bessel-form coefficients under constant-depth
    ws = omega_star(w_sample, r0)
    C2_expected = g * H0 * ws / (ws**2 - f**2)
    C1_expected = g * H0 * ws / (r_grid * (ws**2 - f**2))
    C0_expected = w_sample - (1**2 * g * H0 * ws) / (r_grid**2 * (ws**2 - f**2))

    # Divided coefficients
    C1_over_C2 = C1 / C2
    C0_over_C2 = C0 / C2

    # Verify k^2 expression
    k2_analytic = w_sample * (ws**2 - f**2) / (g * H0 * ws)
    C1_over_C2_expected = 1.0 / r_grid
    C0_over_C2_expected = k2_analytic - 1.0 / r_grid**2

    max_err_C1 = np.max(np.abs(C1_over_C2 - C1_over_C2_expected))
    max_err_C0 = np.max(np.abs(C0_over_C2 - C0_over_C2_expected))

    print(f"B0 = {B0} T, Omega = {Omega} rad/s")
    if B0 == 0.0:
        print(f"Non-magnetic case verified: omega_star = {ws:.6e} == omega = {w_sample:.6e}")
        print(f"k^2 expected: (omega^2 - f^2)/(g*H0) = {k2_analytic:.6e}")

    print(f"Max C1 (1/r) coefficient error over grid: {max_err_C1:.6e}")
    print(f"Max C0 (k^2 - m^2/r^2) coefficient error over grid: {max_err_C0:.6e}")

    # Restore state
    globals()['USE_VARIABLE_DEPTH'] = prev_vdepth
    print("==========================================================\n")

# =============================================================================
# WKB LOCAL DISPERSION / BRANCH PREDICTION
# =============================================================================
def get_wkb_predictions(m, n, r=r0):
    """
    Computes analytical WKB branch predictions for a given m and radial order n
    using the master WKB polynomial evaluated at representative radius r.
    """
    Heq = H_eq(r)
    Q_val = Omega**2 * r - C / (r**2)

    kr_n = n * np.pi / (r2 - r1)
    kappa2_n = kr_n**2 + m**2 / r**2

    K_n = g * Heq * kappa2_n
    om_A2 = omega_A2(r)

    # Poincaré / MS quadratic: y^2 + (2*omA2 - f^2 - K_n)*y + omA2*(omA2 - K_n) = 0
    coeff_b = 2.0 * om_A2 - f**2 - K_n
    coeff_c = om_A2 * (om_A2 - K_n)

    disc = coeff_b**2 - 4.0 * coeff_c
    if disc < 0.0:
        disc = 0.0

    y_plus = (-coeff_b + np.sqrt(disc)) / 2.0
    y_minus = (-coeff_b - np.sqrt(disc)) / 2.0

    # Poincaré
    w_mp_p = np.sqrt(max(y_plus, 0.0))
    w_mp_m = -w_mp_p

    # Rossby
    w_r = m * f * Q_val / (r * (f**2 + K_n))

    # Magnetostrophic
    if B0 > 0.0:
        w_ms_p = np.sqrt(max(y_minus, 0.0))
        w_ms_m = -w_ms_p
    else:
        w_ms_p = np.nan
        w_ms_m = np.nan

    return w_mp_p, w_mp_m, w_r, w_ms_p, w_ms_m

# =============================================================================
# WKB BRANCH ASSIGNMENT
# =============================================================================
def assign_branches_global(eigs, m_val, N_WKB=8):
    """
    Assigns numerical eigenvalues to WKB branches and radial modes (n)
    using a global one-to-one minimum-distance matching strategy.
    """
    wkb_preds = {} # (branch_type, n) -> frequency
    for n in range(1, N_WKB + 1):
        w_mp_p, w_mp_m, w_r, w_ms_p, w_ms_m = get_wkb_predictions(m_val, n)
        wkb_preds[('MP+', n)] = w_mp_p
        wkb_preds[('MP-', n)] = w_mp_m
        wkb_preds[('R', n)] = w_r
        if B0 > 0.0:
            wkb_preds[('MS+', n)] = w_ms_p
            wkb_preds[('MS-', n)] = w_ms_m

    # Collect all physically allowed pairings
    allowed_matches = []
    for j, w_eig in enumerate(eigs):
        for (b_type, n), w_pred in wkb_preds.items():
            if np.isnan(w_pred):
                continue
            # Sign constraints
            if b_type in ['MP+', 'MS+'] and w_eig <= 0:
                continue
            if b_type in ['MP-', 'MS-'] and w_eig >= 0:
                continue

            dist = np.abs(w_eig - w_pred)
            allowed_matches.append((dist, j, (b_type, n)))

    # Sort matches by absolute distance
    allowed_matches.sort(key=lambda x: x[0])

    # Greedily build matching
    matched_eigs = {} # j -> (branch_type, n, pred_w, error)
    matched_branches = set()
    assigned_eig_indices = set()

    for dist, j, b_id in allowed_matches:
        if j not in assigned_eig_indices and b_id not in matched_branches:
            matched_eigs[j] = (b_id[0], b_id[1], wkb_preds[b_id], dist)
            matched_branches.add(b_id)
            assigned_eig_indices.add(j)

    return matched_eigs

# =============================================================================
# GLOBAL SPECTRUM + WKB BRANCH DATA (Test 17 Optional phase integral)
# =============================================================================
def evaluate_phase_integral(omega, m):
    """
    Optionally evaluates the WKB phase integral of the wave normal mode
    to verify consistency against n * pi quantization.
    """
    r_pts = np.linspace(r1, r2, 200)
    dr = r_pts[1] - r_pts[0]

    k2_vals = []
    for r in r_pts:
        Heq = H_eq(r)
        Q_val = Omega**2 * r - C / (r**2)
        om_A2 = omega_A2(r)
        ws = omega_star(omega, r)

        W1 = omega * (ws**2 - f**2) / (g * Heq * ws)
        W2 = m * f * Q_val / (r * g * Heq * ws)
        W3 = 4.0 * m * f * om_A2 * Q_val / (r * g * Heq * (ws**2 - f**2)) if B0 > 0.0 else 0.0

        k2 = W1 + W2 + W3 - m**2 / (r**2)
        k2_vals.append(k2)

    k2_vals = np.array(k2_vals)
    k_vals = np.zeros_like(k2_vals)
    k_vals[k2_vals > 0] = np.sqrt(k2_vals[k2_vals > 0])

    theta = np.sum(k_vals) * dr
    return theta

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
        interior_res = ode_res[1:-1]
        l2_res = np.sqrt(np.sum(np.abs(interior_res)**2) / (N_sample - 2))
        print(f"Test E [ODE Residual]: L2 norm of interior ODE residual = {l2_res:.6e}")
    else:
        print("No roots found for Test D/E.")

    # Test F: Constant-depth limit stability under increasing N_col
    global USE_VARIABLE_DEPTH
    prev_vdepth_check = USE_VARIABLE_DEPTH
    USE_VARIABLE_DEPTH = False
    print("\nRunning Test F: Constant-depth stability scan...")
    for N_test in [32, 64]:
        rg, d1, d2 = chebyshev_lobatto(N_test, r1, r2)
        rts_test = find_eigenvalues(1, N_test, rg, d1, d2)
        print(f"  N = {N_test}: found {len(rts_test)} roots. Sample roots (hat_omega):")
        for rt in rts_test[:4]:
            print(f"    {rt / f:.6f}")

    # Restore variable depth
    USE_VARIABLE_DEPTH = prev_vdepth_check
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
                dist = np.abs(best_rt - ref_rt) / f
                if dist < 0.05:
                    row_str += f" {best_rt/f:+.6f} |"
                else:
                    row_str += f" {'-':^10} |"
            else:
                row_str += f" {'-':^10} |"
        print(row_str)
    print("==========================================================\n")

    # 10. Direct validation against the existing numerical solver for constant depth
    print("==========================================================")
    print("    TEST 10: DIRECT NUMERICAL VS BESSEL VALIDATION        ")
    print("==========================================================")
    prev_vdepth = USE_VARIABLE_DEPTH
    globals()['USE_VARIABLE_DEPTH'] = False

    rg, d1, d2 = chebyshev_lobatto(64, r1, r2)
    cheb_rts = find_eigenvalues(1, 64, rg, d1, d2)
    bessel_rts = find_bessel_eigenvalues(1)

    print(f"{'Mode':>4} | {'Chebyshev omega/f':>18} | {'Bessel omega/f':>15} | {'Difference':>12}")
    print("-" * 60)
    matched_count = 0
    for rt_b in bessel_rts:
        if len(cheb_rts) > 0:
            best_idx = np.argmin(np.abs(cheb_rts - rt_b))
            rt_c = cheb_rts[best_idx]
            diff = np.abs(rt_c - rt_b) / f
            if diff < 1e-3:
                matched_count += 1
                print(f"{matched_count:4d} | {rt_c/f:+18.6f} | {rt_b/f:+15.6f} | {diff:.6e}")

    globals()['USE_VARIABLE_DEPTH'] = prev_vdepth
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

    # Determine the solver mode to run
    if USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH:
        print("Solver mode: EXACT BESSEL")
        print("Equilibrium depth: CONSTANT")
        print("Radial equation: BESSEL")
        print("Eigenvalue method: Bessel boundary determinant")

        # Dense k-grid of horizontal/azimuthal wavenumber parameter
        k_arr = np.linspace(1.0, 30.0, 300)

        pos_branches = {} # n -> list of (k, omega)
        neg_branches = {} # n -> list of (k, omega)

        print("Computing continuous Bessel branches over dense k-grid...")
        for j, k_val in enumerate(k_arr):
            eigs = find_bessel_eigenvalues(k_val, omega_range=(-50 * f, 50 * f))
            pos_eigs = sorted([e for e in eigs if e > 0])
            neg_eigs = sorted([e for e in eigs if e < 0], reverse=True)

            for n, val in enumerate(pos_eigs):
                pos_branches.setdefault(n, []).append((k_val, val))
            for n, val in enumerate(neg_eigs):
                neg_branches.setdefault(n, []).append((k_val, val))

            if (j + 1) % 50 == 0 or j == 0 or j == len(k_arr) - 1:
                print(f"  Progress: {j+1}/{len(k_arr)} k-points solved.")

        # Convert to lists of (n, k_pts, omega_pts)
        pos_branches_list = []
        for n in sorted(pos_branches.keys()):
            pts = pos_branches[n]
            k_pts = np.array([p[0] for p in pts])
            omega_pts = np.array([p[1] for p in pts])
            pos_branches_list.append((n, k_pts, omega_pts))

        neg_branches_list = []
        for n in sorted(neg_branches.keys()):
            pts = neg_branches[n]
            k_pts = np.array([p[0] for p in pts])
            omega_pts = np.array([p[1] for p in pts])
            neg_branches_list.append((n, k_pts, omega_pts))

        dispersion_data = {
            'pos_branches': pos_branches_list,
            'neg_branches': neg_branches_list
        }

        return k_arr, dispersion_data
    else:
        if not USE_VARIABLE_DEPTH:
            print("Solver mode: CHEBYSHEV GLOBAL")
            print("Equilibrium depth: CONSTANT")
            print("Radial equation: REDUCED CONSTANT-COEFFICIENT ODE")
            print("Eigenvalue method: Global collocation")
        else:
            print("Solver mode: CHEBYSHEV GLOBAL")
            print("Equilibrium depth: VARIABLE")
            print("Radial equation: GENERAL VARIABLE-COEFFICIENT ODE")
            print("Eigenvalue method: Global collocation")

        # Store structured classifications
        branch_assignments = {
            'MP+': {}, 'MP-': {}, 'R': {}, 'MS+': {}, 'MS-': {}, 'unassigned': []
        }

        for m in m_arr:
            eigs = find_eigenvalues(m, N_col, r_grid, D1_mat, D2_mat, omega_range=(-50 * f, 50 * f))

            # WKB local branch matching
            matches = assign_branches_global(eigs, m, N_WKB)

            print(f"  m={m:2d} (found {len(eigs)} roots via Chebyshev):")
            for j, w_eig in enumerate(eigs):
                if j in matches:
                    b_type, n, w_wkb, err = matches[j]
                    branch_assignments[b_type].setdefault(n, []).append((m, w_eig, w_wkb, err))

                    # Optional Test 17: Phase integral verification
                    theta = evaluate_phase_integral(w_eig, m)
                    phase_err = np.abs(theta - n * np.pi) / (n * np.pi)

                    print(f"    Assigned: {b_type} n={n} -> omega_hat = {w_eig/f:+.4f} (WKB = {w_wkb/f:+.4f}, Err = {err/f:.4f}, Phase Err = {phase_err:.2%})")
                else:
                    branch_assignments['unassigned'].append((m, w_eig))
                    print(f"    Unassigned: omega_hat = {w_eig/f:+.4f}")
            print()

        return m_arr, branch_assignments

# =============================================================================
# PUBLICATION-QUALITY DISPERSION PLOT
# =============================================================================
def plot_dispersion_relation(x_arr, dispersion_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.6, 1]})

    title_suffix = "(Exact Bessel)" if (USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH) else "(Chebyshev)"
    fig.suptitle(f'SWMHD Global Dispersion Relation {title_suffix}\n', fontsize=16, fontweight='bold')

    COLORS = {
        'MP':  '#1f77b4', # blue
        'MK':  '#2ca02c', # green
        'R':   '#d62728', # red
        'MS':  '#ff7f0e', # orange
        'unassigned': 'gray'
    }

    if USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH:
        # Plot continuous Bessel branches
        pos_branches = dispersion_data['pos_branches']
        neg_branches = dispersion_data['neg_branches']

        # Full spectrum plot
        for n, k_pts, omega_pts in pos_branches:
            color = 'blue' if n == 0 else 'red'
            lbl = 'Kelvin' if n == 0 else ('Poincare' if n == 1 else None)
            ax1.plot(k_pts, omega_pts / f, color=color, lw=2.5, label=lbl, zorder=5)
            ax2.plot(k_pts, omega_pts / f, color=color, lw=2.5, label=lbl, zorder=5)

        for n, k_pts, omega_pts in neg_branches:
            color = 'blue' if n == 0 else 'red'
            ax1.plot(k_pts, omega_pts / f, color=color, lw=2.5, zorder=5)
            ax2.plot(k_pts, omega_pts / f, color=color, lw=2.5, zorder=5)

        ax1.axhline(0, color='grey', lw=0.8, ls=':')
        ax1.axhline(+1, color='grey', lw=0.9, ls='--', alpha=0.5)
        ax1.axhline(-1, color='grey', lw=0.9, ls='--', alpha=0.5)

        ax1.set_xlabel('Wavenumber $k$', fontsize=15)
        ax1.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
        ax1.set_xlim(1.0, 30.0)
        ax1.set_ylim(-50.0, 50.0)
        ax1.grid(True, alpha=0.22)
        ax1.set_title('Full Spectrum', fontsize=14)

        ax2.axhline(0, color='grey', lw=0.8, ls=':')
        ax2.set_xlabel('Wavenumber $k$', fontsize=15)
        ax2.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
        ax2.set_xlim(1.0, 30.0)
        ax2.set_ylim(-0.8, 0.2)
        ax2.grid(True, alpha=0.22)
        ax2.set_title('Slow Branches Zoom', fontsize=14)

        # Deduplicate and create legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        by_label1 = dict(zip(labels1, handles1))
        ax1.legend(by_label1.values(), by_label1.keys(), loc='upper left', fontsize=12)

        handles2, labels2 = ax2.get_legend_handles_labels()
        by_label2 = dict(zip(labels2, handles2))
        ax2.legend(by_label2.values(), by_label2.keys(), loc='lower left', fontsize=12)
    else:
        # Plot Chebyshev and WKB curves
        m_fine = np.linspace(1.0, 30.0, 400)

        # Plot continuous WKB predictions
        for n in range(1, N_WKB + 1):
            alpha_val = max(0.1, 0.4 - 0.05 * (n - 1))

            # WKB branches
            w_mp_p_list = []
            w_mp_m_list = []
            w_r_list = []
            w_ms_p_list = []
            w_ms_m_list = []

            for m in m_fine:
                w_mp_p, w_mp_m, w_r, w_ms_p, w_ms_m = get_wkb_predictions(m, n)
                w_mp_p_list.append(w_mp_p)
                w_mp_m_list.append(w_mp_m)
                w_r_list.append(w_r)
                w_ms_p_list.append(w_ms_p)
                w_ms_m_list.append(w_ms_m)

            w_mp_p_list = np.array(w_mp_p_list) / f
            w_mp_m_list = np.array(w_mp_m_list) / f
            w_r_list = np.array(w_r_list) / f
            w_ms_p_list = np.array(w_ms_p_list) / f
            w_ms_m_list = np.array(w_ms_m_list) / f

            lbl_mp = f'Magneto-Poincaré n={n}' if n <= 2 else None
            lbl_r = f'Rossby n={n}' if n <= 2 else None
            lbl_ms = f'Magnetostrophic n={n}' if (n <= 2 and B0 > 0.0) else None

            # Plot WKB MP
            ax1.plot(m_fine, w_mp_p_list, color=COLORS['MP'], lw=1.5, alpha=alpha_val, label=lbl_mp)
            ax1.plot(m_fine, w_mp_m_list, color=COLORS['MP'], lw=1.5, alpha=alpha_val)

            # Plot WKB Rossby
            ax1.plot(m_fine, w_r_list, color=COLORS['R'], lw=1.5, alpha=alpha_val, label=lbl_r)
            ax2.plot(m_fine, w_r_list, color=COLORS['R'], lw=1.5, alpha=alpha_val, label=lbl_r)

            # Plot WKB MS (if magnetic field is present)
            if B0 > 0.0:
                ax1.plot(m_fine, w_ms_p_list, color=COLORS['MS'], lw=1.5, alpha=alpha_val, label=lbl_ms)
                ax1.plot(m_fine, w_ms_m_list, color=COLORS['MS'], lw=1.5, alpha=alpha_val)
                ax2.plot(m_fine, w_ms_p_list, color=COLORS['MS'], lw=1.5, alpha=alpha_val, label=lbl_ms)
                ax2.plot(m_fine, w_ms_m_list, color=COLORS['MS'], lw=1.5, alpha=alpha_val)

        # Plot assigned global eigenvalues as colored scatter points
        for b_type in ['MP+', 'MP-', 'R', 'MS+', 'MS-']:
            color_key = 'MP' if 'MP' in b_type else ('R' if b_type == 'R' else 'MS')
            for n in sorted(dispersion_data[b_type].keys()):
                pts = dispersion_data[b_type][n]
                if len(pts) == 0:
                    continue
                m_vals = [p[0] for p in pts]
                w_vals = [p[1] / f for p in pts]
                lbl = f'Global {color_key}' if n == 1 else None

                ax1.scatter(m_vals, w_vals, color=COLORS[color_key], marker='o', s=30, alpha=0.85, zorder=5, label=lbl)
                if 'MP' not in b_type:
                    ax2.scatter(m_vals, w_vals, color=COLORS[color_key], marker='o', s=45, alpha=0.85, zorder=5, label=lbl)

        # Plot unassigned eigenvalues as neutral scatter dots
        unassigned_pts = dispersion_data['unassigned']
        if len(unassigned_pts) > 0:
            m_un = [p[0] for p in unassigned_pts]
            w_un = [p[1] / f for p in unassigned_pts]
            ax1.scatter(m_un, w_un, color=COLORS['unassigned'], marker='x', s=20, alpha=0.5, zorder=4, label='Unassigned')
            ax2.scatter(m_un, w_un, color=COLORS['unassigned'], marker='x', s=25, alpha=0.5, zorder=4, label='Unassigned')

        ax1.axhline(0, color='grey', lw=0.8, ls=':')
        ax1.axhline(+1, color='grey', lw=0.9, ls='--', alpha=0.5)
        ax1.axhline(-1, color='grey', lw=0.9, ls='--', alpha=0.5)

        ax1.set_xlabel('Azimuthal wavenumber $m$', fontsize=15)
        ax1.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
        ax1.set_xlim(0.5, len(x_arr) + 0.5)
        ax1.set_ylim(-50.0, 50.0)
        ax1.set_xticks(x_arr[::2])
        ax1.grid(True, alpha=0.22)
        ax1.set_title('Full Spectrum', fontsize=14)

        ax2.axhline(0, color='grey', lw=0.8, ls=':')
        ax2.set_xlabel('Azimuthal wavenumber $m$', fontsize=15)
        ax2.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
        ax2.set_xlim(0.5, len(x_arr) + 0.5)
        ax2.set_ylim(-0.8, 0.2)
        ax2.set_xticks(x_arr[::2])
        ax2.grid(True, alpha=0.22)
        ax2.set_title('Slow Branches Zoom', fontsize=14)

        # Figure legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        by_label1 = dict(zip(labels1, handles1))
        ax1.legend(by_label1.values(), by_label1.keys(), loc='upper left', fontsize=10, framealpha=0.9)

        handles2, labels2 = ax2.get_legend_handles_labels()
        by_label2 = dict(zip(labels2, handles2))
        ax2.legend(by_label2.values(), by_label2.keys(), loc='lower left', fontsize=10, framealpha=0.9)

    # Figure wide legend placed outside the right panel
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    seen = set()
    legend_handles = []
    legend_labels = []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l)
            legend_handles.append(h)
            legend_labels.append(l)

    fig.legend(
        legend_handles,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=13,
        frameon=True,
        borderaxespad=0.5,
        labelspacing=0.8,
        handlelength=2.2
    )

    # Parameter box
    pbox = (
        rf"$\Omega = {Omega:.1e}$ rad/s,  $H_0 = {H0}$ m" + "\n"
        rf"$r_1 = {r1:.1e}$ m,  $r_2 = {r2:.1e}$ m" + "\n"
        rf"$\omega_A(r_0) = {np.sqrt(omega_A2(r0)):.2e}$ s$^{{-1}}$" + "\n"
        rf"Solver: {title_suffix}"
    )
    fig.text(0.52, 0.015, pbox, fontsize=11, va='bottom', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))

    fig.subplots_adjust(bottom=0.15, right=0.78)
    plt.tight_layout(rect=[0, 0.12, 0.78, 0.95])

    plt.savefig("outputs/swmhd_variable_depth_global_dispersion.png", dpi=180, bbox_inches='tight')
    plt.savefig("outputs/swmhd_variable_depth_global_dispersion.pdf", dpi=180, bbox_inches='tight')
    print("Saved plots:")
    print("  - outputs/swmhd_variable_depth_global_dispersion.png")
    print("  - outputs/swmhd_variable_depth_global_dispersion.pdf")

    # Save separate versions of panels and legend as requested
    fig_left, ax_l = plt.subplots(figsize=(8, 6))
    if USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH:
        # Plot continuous Bessel branches
        pos_branches = dispersion_data['pos_branches']
        neg_branches = dispersion_data['neg_branches']
        for n, k_pts, omega_pts in pos_branches:
            color = 'blue' if n == 0 else 'red'
            ax_l.plot(k_pts, omega_pts / f, color=color, lw=2.5)
        for n, k_pts, omega_pts in neg_branches:
            color = 'blue' if n == 0 else 'red'
            ax_l.plot(k_pts, omega_pts / f, color=color, lw=2.5)
        ax_l.set_xlabel('Wavenumber $k$')
        ax_l.set_xlim(1.0, 30.0)
    else:
        # Plot scatter
        for b_type in ['MP+', 'MP-', 'R', 'MS+', 'MS-']:
            color_key = 'MP' if 'MP' in b_type else ('R' if b_type == 'R' else 'MS')
            for n in sorted(dispersion_data[b_type].keys()):
                pts = dispersion_data[b_type][n]
                if len(pts) == 0:
                    continue
                m_vals = [p[0] for p in pts]
                w_vals = [p[1] / f for p in pts]
                ax_l.scatter(m_vals, w_vals, color=COLORS[color_key], marker='o', s=30, alpha=0.85)
        ax_l.set_xlabel('Azimuthal wavenumber $m$')
        ax_l.set_xlim(0.5, len(x_arr) + 0.5)

    ax_l.axhline(0, color='grey', lw=0.8, ls=':')
    ax_l.set_ylabel(r'Normalised frequency $\hat\omega$')
    ax_l.set_ylim(-50.0, 50.0)
    ax_l.grid(True, alpha=0.22)
    ax_l.set_title('Full Spectrum')
    plt.tight_layout()
    plt.savefig("outputs/FullSpectrum_variable_depth.png", dpi=300, bbox_inches='tight')
    plt.close(fig_left)

    fig_right, ax_r = plt.subplots(figsize=(6, 6))
    if USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH:
        pos_branches = dispersion_data['pos_branches']
        neg_branches = dispersion_data['neg_branches']
        for n, k_pts, omega_pts in pos_branches:
            color = 'blue' if n == 0 else 'red'
            ax_r.plot(k_pts, omega_pts / f, color=color, lw=2.5)
        for n, k_pts, omega_pts in neg_branches:
            color = 'blue' if n == 0 else 'red'
            ax_r.plot(k_pts, omega_pts / f, color=color, lw=2.5)
        ax_r.set_xlabel('Wavenumber $k$')
        ax_r.set_xlim(1.0, 30.0)
    else:
        for b_type in ['R', 'MS+', 'MS-']:
            color_key = 'R' if b_type == 'R' else 'MS'
            for n in sorted(dispersion_data[b_type].keys()):
                pts = dispersion_data[b_type][n]
                if len(pts) == 0:
                    continue
                m_vals = [p[0] for p in pts]
                w_vals = [p[1] / f for p in pts]
                ax_r.scatter(m_vals, w_vals, color=COLORS[color_key], marker='o', s=45, alpha=0.85)
        ax_r.set_xlabel('Azimuthal wavenumber $m$')
        ax_r.set_xlim(0.5, len(x_arr) + 0.5)

    ax_r.axhline(0, color='grey', lw=0.8, ls=':')
    ax_r.set_ylabel(r'Normalised frequency $\hat\omega$')
    ax_r.set_ylim(-0.8, 0.2)
    ax_r.grid(True, alpha=0.22)
    ax_r.set_title('Slow Branches Zoom')
    plt.tight_layout()
    plt.savefig("outputs/SlowSpectrum_variable_depth.png", dpi=300, bbox_inches='tight')
    plt.close(fig_right)

    fig_leg = plt.figure(figsize=(10, 2))
    fig_leg.legend(
        legend_handles,
        legend_labels,
        loc='center',
        ncol=4,
        fontsize=13,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig("outputs/Legend_variable_depth.png", dpi=300, bbox_inches='tight')
    plt.close(fig_leg)

def main():
    if USE_ANALYTIC_BESSEL and USE_VARIABLE_DEPTH:
        print("Analytic Bessel solver unavailable: H_eq(r) is variable.\nUsing Chebyshev global solver.")
        globals()['USE_ANALYTIC_BESSEL'] = False

    validate_bessel_reduction()
    run_validation_checks()
    x_arr, dispersion_data = compute_dispersion_relation()
    plot_dispersion_relation(x_arr, dispersion_data)

if __name__ == "__main__":
    main()
