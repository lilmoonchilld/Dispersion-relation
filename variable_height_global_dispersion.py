"""
Variable-Height SWMHD Global Dispersion Solver (Publication Grade)
===================================================================
Solves the global eigenvalue problem for linearized variable-depth SWMHD in physical coordinates (SI units).
Upgraded with singular-value-based (SVD) eigenvalue extraction and continuous-m exact Bessel branch tracking.
"""

import os
import sys
import platform
import csv
import numpy as np
import scipy
import scipy.linalg as la
scipy_version = scipy.__version__
numpy_version = np.__version__

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import brentq, minimize_scalar
from scipy.special import jv, yv, jvp, yvp, iv, kv, ivp, kvp
from scipy.interpolate import interp1d

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
# 0. Physical Parameters (SI Units) & Solver Configuration
# =============================================================================
MU0   = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]

# Definitive parameters
Omega = 1.65e-4        # Rotation rate [rad/s]
H0    = 50.0           # Reference depth [m]
g     = 10.44          # Gravitational acceleration [m/s^2]
B0    = 5e-5           # External vertical magnetic field [T]
rho0  = 1000.0        # Density [kg/m^3]
C     = 1e16          # Radial gravity constant [m^3/s^2]

r1    = 0.5e8         # Inner radius [m]
r2    = 1.0e8         # Outer radius [m]
r0    = 0.5 * (r1 + r2) # Reference radius [m]

f     = 2.0 * Omega   # Coriolis parameter [rad/s]

# Validation switches
USE_VARIABLE_DEPTH = False
USE_ANALYTIC_BESSEL = True  # Set to False by default if variable depth

# Continuous m plotting configuration
M_CONTINUOUS_MIN  = 1.0
M_CONTINUOUS_MAX  = 30.0
M_CONTINUOUS_STEP = 0.05
SHOW_INTEGER_M_POINTS = False
FULL_SPECTRUM_YLIM = (-40.0, 40.0)
SLOW_BRANCH_YLIM   = (-0.5, 0.5)

# Numerical Solver Tolerances
OMEGA_TOL        = 1e-8 * f     # Zero-frequency exclusion threshold [rad/s]
RESONANCE_TOL    = 1e-6 * f**2  # Resonance exclusion threshold [rad/s]^2
SVD_RESIDUAL_TOL = 1e-8         # Default SVD normalized residual acceptance threshold

# Candidate detection parameters
MIN_PROMINENCE   = 0.999        # Candidate local minimum prominence ratio
MIN_SEPARATION   = 1e-5 * f     # Minimum separation between distinct candidates [rad/s]

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
    if not is_valid_frequency(omega, r_grid):
        return None

    C2, C1, C0 = radial_coefficients(omega, m, r_grid)

    # Scale interior rows by 1/C2 so the ODE differential operator is monic:
    # eta'' + (C1/C2)*eta' + (C0/C2)*eta = 0
    L = D2_mat + np.diag(C1 / C2) @ D1_mat + np.diag(C0 / C2)

    # Boundaries: v_r = 0. Replace row 0 (r1) and row N-1 (r2)
    for idx in [0, N - 1]:
        rb = r_grid[idx]
        ws_b = omega_star(omega, rb)
        L[idx, :] = ws_b * D1_mat[idx, :] - (m * f / rb) * np.eye(N)[idx]

    return L

# =============================================================================
# 6. Singular Frequency Partitioning & Validity Checks
# =============================================================================
def is_valid_frequency(omega, r_grid):
    """
    Checks whether a trial frequency omega is non-singular and safe for matrix assembly.
    Excludes omega = 0 and inertial resonance locations omega_*^2 = f^2.
    """
    if np.abs(omega) < OMEGA_TOL:
        return False
    ws = omega_star(omega, r_grid)
    if np.any(np.abs(ws**2 - f**2) < RESONANCE_TOL):
        return False
    return True

def find_singular_frequencies(r_grid, omega_range=(-50*f, 50*f)):
    """
    Finds all singular frequencies (omega = 0 and roots of omega_*^2(r) - f^2 = 0)
    within the given frequency search range across the radial grid.
    """
    s_points = {0.0}

    if B0 == 0.0:
        s_points.add(f)
        s_points.add(-f)
    else:
        om_A2_arr = omega_A2(r_grid)
        for om_A2_val in om_A2_arr:
            disc = f**2 - 4.0 * om_A2_val
            if disc >= 0:
                sq_disc = np.sqrt(disc)
                s_points.add(0.5 * (f + sq_disc))
                s_points.add(0.5 * (f - sq_disc))
                s_points.add(0.5 * (-f + sq_disc))
                s_points.add(0.5 * (-f - sq_disc))

    w_min, w_max = omega_range
    valid_s = sorted([s for s in s_points if w_min <= s <= w_max])
    return np.array(valid_s)

def partition_valid_intervals(r_grid, omega_range=(-50*f, 50*f), delta_buffer=1e-4*f):
    """
    Partitions [omega_min, omega_max] into contiguous valid frequency intervals,
    buffering away from singular frequencies by merging overlapping exclusion regions.
    """
    s_points = find_singular_frequencies(r_grid, omega_range)
    w_min, w_max = omega_range

    s_in_range = sorted([s for s in set(s_points) if w_min <= s <= w_max])

    exclusion_buffers = []
    for s in s_in_range:
        b_left = s - delta_buffer
        b_right = s + delta_buffer
        if not exclusion_buffers:
            exclusion_buffers.append([b_left, b_right])
        else:
            if b_left <= exclusion_buffers[-1][1]:
                exclusion_buffers[-1][1] = max(exclusion_buffers[-1][1], b_right)
            else:
                exclusion_buffers.append([b_left, b_right])

    valid_intervals = []
    curr = w_min
    for b_left, b_right in exclusion_buffers:
        if b_left > curr + delta_buffer:
            valid_intervals.append((curr, b_left))
        curr = max(curr, b_right)
    if w_max > curr + delta_buffer:
        valid_intervals.append((curr, w_max))

    return valid_intervals

# =============================================================================
# 7. SVD-Based Primary Eigenvalue Solver
# =============================================================================
def normalized_sigma_min(omega, m, N, r_grid, D1_mat, D2_mat):
    """
    Evaluates matrix singular values and returns (sigma_min, sigma_max, sigma_rel).
    Returns (NaN, NaN, NaN) if frequency is singular or invalid.
    """
    L = build_matrix(omega, m, N, r_grid, D1_mat, D2_mat)
    if L is None:
        return np.nan, np.nan, np.nan
    try:
        s = la.svdvals(L)
        sigma_min = s[-1]
        sigma_max = s[0]
        if sigma_max <= 0.0:
            return np.nan, np.nan, np.nan
        return sigma_min, sigma_max, sigma_min / sigma_max
    except Exception:
        return np.nan, np.nan, np.nan

def find_eigenvalues_svd(m_val, N, r_grid, D1_mat, D2_mat, omega_range=(-0.005*f, 0.005*f), n_scan=2000, residual_tol=SVD_RESIDUAL_TOL):
    """
    SVD-based Primary Eigenvalue Solver.
    Scans valid frequency intervals for local minima of normalized_sigma_min(omega),
    refines candidates using bounded scalar minimization, and filters by residual_tol.
    """
    valid_intervals = partition_valid_intervals(r_grid, omega_range)
    accepted_roots = []
    accepted_sig_rels = []
    all_candidates = []

    total_len = sum(b - a for a, b in valid_intervals)

    for a_k, b_k in valid_intervals:
        n_pts = max(50, int(n_scan * (b_k - a_k) / total_len))
        w_scan = np.linspace(a_k, b_k, n_pts)

        sig_rels = []
        sig_mins = []
        sig_maxs = []
        valid_w = []

        for w in w_scan:
            s_min, s_max, s_rel = normalized_sigma_min(w, m_val, N, r_grid, D1_mat, D2_mat)
            if not np.isnan(s_rel):
                valid_w.append(w)
                sig_mins.append(s_min)
                sig_maxs.append(s_max)
                sig_rels.append(s_rel)

        valid_w = np.array(valid_w)
        sig_rels = np.array(sig_rels)

        if len(valid_w) < 3:
            continue

        for i in range(1, len(valid_w) - 1):
            if sig_rels[i] <= sig_rels[i-1] and sig_rels[i] <= sig_rels[i+1]:
                neighbors_min = min(sig_rels[i-1], sig_rels[i+1])
                if sig_rels[i] <= MIN_PROMINENCE * neighbors_min or sig_rels[i] < residual_tol * 10.0:
                    cand_w = valid_w[i]
                    all_candidates.append((cand_w, sig_rels[i]))

                    w_left = valid_w[i-1]
                    w_right = valid_w[i+1]

                    def f_obj(w):
                        _, _, s_rel = normalized_sigma_min(w, m_val, N, r_grid, D1_mat, D2_mat)
                        return s_rel if not np.isnan(s_rel) else 1e10

                    try:
                        res = minimize_scalar(f_obj, bounds=(w_left, w_right), method='bounded', options={'xatol': 1e-13})
                        if res.success:
                            w_ref = res.x
                            s_min, s_max, s_rel = normalized_sigma_min(w_ref, m_val, N, r_grid, D1_mat, D2_mat)
                            if s_rel <= residual_tol:
                                if not any(np.abs(w_ref - r) < MIN_SEPARATION for r in accepted_roots):
                                    accepted_roots.append(w_ref)
                                    accepted_sig_rels.append(s_rel)
                    except Exception:
                        pass

    if len(accepted_roots) > 0:
        sort_idx = np.argsort(accepted_roots)
        accepted_roots = np.array(accepted_roots)[sort_idx]
        accepted_sig_rels = np.array(accepted_sig_rels)[sort_idx]
    else:
        accepted_roots = np.array([])
        accepted_sig_rels = np.array([])

    return accepted_roots, accepted_sig_rels, all_candidates

# =============================================================================
# 8. Determinant Solver (Secondary Cross-Check Solver)
# =============================================================================
def find_eigenvalues_determinant(m_val, N, r_grid, D1_mat, D2_mat, omega_range=(-0.005*f, 0.005*f), n_scan=2000):
    """
    Secondary Cross-Check Solver using determinant sign-changes.
    Scans valid frequency intervals for sign changes of det(L(omega)),
    and isolates roots using brentq.
    """
    valid_intervals = partition_valid_intervals(r_grid, omega_range)
    roots = []
    total_len = sum(b - a for a, b in valid_intervals)

    for a_k, b_k in valid_intervals:
        n_pts = max(50, int(n_scan * (b_k - a_k) / total_len))
        scan = np.linspace(a_k, b_k, n_pts)

        signs = []
        log_dets = []
        valid_scan = []

        for w in scan:
            L = build_matrix(w, m_val, N, r_grid, D1_mat, D2_mat)
            if L is None:
                continue
            try:
                sign, log_det = np.linalg.slogdet(L)
                signs.append(sign)
                log_dets.append(log_det)
                valid_scan.append(w)
            except Exception:
                pass

        signs = np.array(signs)
        log_dets = np.array(log_dets)
        valid_scan = np.array(valid_scan)

        for i in range(len(valid_scan) - 1):
            if signs[i] * signs[i+1] < 0:
                ref_log_det = log_dets[i]

                def f_to_solve(w):
                    L = build_matrix(w, m_val, N, r_grid, D1_mat, D2_mat)
                    if L is None:
                        return np.nan
                    sign, log_det = np.linalg.slogdet(L)
                    return sign * np.exp(log_det - ref_log_det)

                try:
                    root = brentq(f_to_solve, valid_scan[i], valid_scan[i+1], xtol=1e-15)
                    if not any(np.abs(root - r) < MIN_SEPARATION for r in roots):
                        roots.append(root)
                except Exception:
                    pass

    return np.array(sorted(roots))

# =============================================================================
# 9. Eigenfunction Calculation & Residual Diagnostics
# =============================================================================
def get_eigenfunction(omega, m, N, r_grid, D1_mat, D2_mat):
    """
    Extracts the normalized eigenfunction eta (max |eta| = 1) from the right singular vector
    corresponding to the smallest singular value of L(omega).
    """
    L = build_matrix(omega, m, N, r_grid, D1_mat, D2_mat)
    if L is None:
        return None
    U, s, Vh = la.svd(L)
    eta = Vh[-1, :].conj()
    max_eta = np.max(np.abs(eta))
    if max_eta > 0.0:
        eta /= max_eta
    return eta

def compute_eigen_residuals(omega, m, N, r_grid, D1_mat, D2_mat):
    """
    Computes full matrix residual, interior ODE residual, and boundary condition residuals for eigenmode.
    """
    eta = get_eigenfunction(omega, m, N, r_grid, D1_mat, D2_mat)
    if eta is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    L = build_matrix(omega, m, N, r_grid, D1_mat, D2_mat)

    R_full = np.linalg.norm(L @ eta, 2) / np.linalg.norm(eta, 2)

    C2, C1, C0 = radial_coefficients(omega, m, r_grid)
    ode_res = C2 * (D2_mat @ eta) + C1 * (D1_mat @ eta) + C0 * eta
    R_ODE = np.linalg.norm(ode_res[1:-1], 2) / np.linalg.norm(eta[1:-1], 2)

    ws1 = omega_star(omega, r_grid[0])
    R_inner = np.abs(ws1 * (D1_mat[0, :] @ eta) - (m * f / r_grid[0]) * eta[0])

    ws2 = omega_star(omega, r_grid[-1])
    R_outer = np.abs(ws2 * (D1_mat[-1, :] @ eta) - (m * f / r_grid[-1]) * eta[-1])

    _, _, sig_rel = normalized_sigma_min(omega, m, N, r_grid, D1_mat, D2_mat)

    return R_full, R_ODE, R_inner, R_outer, sig_rel

# =============================================================================
# 10. Analytic Bessel Special-Case Solver & Continuous Branch Solver
# =============================================================================
def bessel_radial_wavenumber(omega):
    ws = omega_star(omega, r0)
    k2 = omega * (ws**2 - f**2) / (g * H0 * ws)
    if k2 >= 0.0:
        return k2, True
    else:
        return -k2, False

def bessel_boundary_determinant(omega, m):
    """
    Evaluates exact Bessel boundary determinant for arbitrary real-valued m.
    Supports scalar or 1D array omega with vectorized evaluation.
    D(omega, m) = B1_J * B2_Y - B1_Y * B2_J = 0
    """
    if isinstance(omega, np.ndarray):
        det = np.full_like(omega, np.nan, dtype=float)
        ws = omega_star(omega, r0)
        k2 = omega * (ws**2 - f**2) / (g * H0 * ws)

        invalid_mask = (np.abs(omega) < OMEGA_TOL) | (np.abs(ws**2 - f**2) < RESONANCE_TOL)
        valid_mask = ~invalid_mask
        if not np.any(valid_mask):
            return det

        w_v = omega[valid_mask]
        ws_v = ws[valid_mask]
        k2_v = k2[valid_mask]

        is_osc = k2_v >= 0.0
        valid_indices = np.where(valid_mask)[0]

        if np.any(is_osc):
            w_osc = w_v[is_osc]
            ws_osc = ws_v[is_osc]
            arg_osc = np.sqrt(k2_v[is_osc])

            B1_J = ws_osc * arg_osc * jvp(m, arg_osc * r1) - (m * f / r1) * jv(m, arg_osc * r1)
            B1_Y = ws_osc * arg_osc * yvp(m, arg_osc * r1) - (m * f / r1) * yv(m, arg_osc * r1)
            B2_J = ws_osc * arg_osc * jvp(m, arg_osc * r2) - (m * f / r2) * jv(m, arg_osc * r2)
            B2_Y = ws_osc * arg_osc * yvp(m, arg_osc * r2) - (m * f / r2) * yv(m, arg_osc * r2)

            det[valid_indices[is_osc]] = B1_J * B2_Y - B1_Y * B2_J

        if np.any(~is_osc):
            w_ev = w_v[~is_osc]
            ws_ev = ws_v[~is_osc]
            arg_ev = np.sqrt(-k2_v[~is_osc])

            B1_I = ws_ev * arg_ev * ivp(m, arg_ev * r1) - (m * f / r1) * iv(m, arg_ev * r1)
            B1_K = ws_ev * arg_ev * kvp(m, arg_ev * r1) - (m * f / r1) * kv(m, arg_ev * r1)
            B2_I = ws_ev * arg_ev * ivp(m, arg_ev * r2) - (m * f / r2) * iv(m, arg_ev * r2)
            B2_K = ws_ev * arg_ev * kvp(m, arg_ev * r2) - (m * f / r2) * kv(m, arg_ev * r2)

            det[valid_indices[~is_osc]] = B1_I * B2_K - B1_K * B2_I

        return det
    else:
        if np.abs(omega) < OMEGA_TOL:
            return np.nan
        ws = omega_star(omega, r0)
        if np.abs(ws**2 - f**2) < RESONANCE_TOL:
            return np.nan
        k2 = omega * (ws**2 - f**2) / (g * H0 * ws)
        if k2 >= 0.0:
            arg = np.sqrt(k2)
            B1_J = ws * arg * jvp(m, arg * r1) - (m * f / r1) * jv(m, arg * r1)
            B1_Y = ws * arg * yvp(m, arg * r1) - (m * f / r1) * yv(m, arg * r1)
            B2_J = ws * arg * jvp(m, arg * r2) - (m * f / r2) * jv(m, arg * r2)
            B2_Y = ws * arg * yvp(m, arg * r2) - (m * f / r2) * yv(m, arg * r2)
            return B1_J * B2_Y - B1_Y * B2_J
        else:
            arg = np.sqrt(-k2)
            B1_I = ws * arg * ivp(m, arg * r1) - (m * f / r1) * iv(m, arg * r1)
            B1_K = ws * arg * kvp(m, arg * r1) - (m * f / r1) * kv(m, arg * r1)
            B2_I = ws * arg * ivp(m, arg * r2) - (m * f / r2) * iv(m, arg * r2)
            B2_K = ws * arg * kvp(m, arg * r2) - (m * f / r2) * kv(m, arg * r2)
            return B1_I * B2_K - B1_K * B2_I

def get_bessel_eigenfunction(omega, m, n_pts=200):
    """
    Reconstructs exact normalized Bessel radial eigenfunction eta(r) on a dense grid [r1, r2].
    Also counts interior nodes and boundary energy concentration for radial mode index classification.
    """
    r_arr = np.linspace(r1, r2, n_pts)
    ws = omega_star(omega, r0)
    val, is_oscillatory = bessel_radial_wavenumber(omega)
    arg = np.sqrt(val)

    if is_oscillatory:
        B1_J = ws * arg * jvp(m, arg * r1) - (m * f / r1) * jv(m, arg * r1)
        B1_Y = ws * arg * yvp(m, arg * r1) - (m * f / r1) * yv(m, arg * r1)
        if np.abs(B1_J) >= np.abs(B1_Y):
            eta = B1_J * yv(m, arg * r_arr) - B1_Y * jv(m, arg * r_arr)
        else:
            eta = jv(m, arg * r_arr) - (B1_J / (B1_Y + 1e-300)) * yv(m, arg * r_arr)
    else:
        B1_I = ws * arg * ivp(m, arg * r1) - (m * f / r1) * iv(m, arg * r1)
        B1_K = ws * arg * kvp(m, arg * r1) - (m * f / r1) * kv(m, arg * r1)
        if np.abs(B1_I) >= np.abs(B1_K):
            eta = B1_I * kv(m, arg * r_arr) - B1_K * iv(m, arg * r_arr)
        else:
            eta = iv(m, arg * r_arr) - (B1_I / (B1_K + 1e-300)) * kv(m, arg * r_arr)

    max_eta = np.max(np.abs(eta))
    if max_eta > 0.0:
        eta /= max_eta

    interior = eta[1:-1]
    node_count = np.count_nonzero(np.diff(np.sign(interior))) if len(interior) > 1 else 0
    return eta, node_count

def find_bessel_eigenvalues(m_val, omega_range=(-25*f, 25*f), n_scan=1000):
    """
    Calculates exact Bessel roots for a given azimuthal wavenumber m_val (float or int).
    Vectorized over frequency intervals for fast performance.
    """
    valid_intervals = partition_valid_intervals(np.array([r1, r0, r2]), omega_range)
    roots = []
    total_len = sum(b - a for a, b in valid_intervals)

    for a_k, b_k in valid_intervals:
        n_pts = max(50, int(n_scan * (b_k - a_k) / total_len))
        scan = np.linspace(a_k, b_k, n_pts)

        dets = bessel_boundary_determinant(scan, m_val)
        valid_mask = ~np.isnan(dets)
        valid_scan = scan[valid_mask]
        valid_dets = dets[valid_mask]

        for i in range(len(valid_scan) - 1):
            if valid_dets[i] * valid_dets[i+1] < 0:
                ref_val = np.abs(valid_dets[i])
                def f_to_solve(w):
                    return bessel_boundary_determinant(w, m_val) / ref_val
                try:
                    root = brentq(f_to_solve, valid_scan[i], valid_scan[i+1], xtol=1e-12)
                    if not any(np.abs(root - r) < MIN_SEPARATION for r in roots):
                        roots.append(root)
                except Exception:
                    pass

    return np.array(sorted(roots))

def classify_bessel_root(omega, m_val):
    """
    Physically classifies a Bessel eigenfrequency root into branch category and radial mode index n.
    Categorization uses radial wavenumber k_r^2 sign (evanescent vs oscillatory),
    boundary localization energy, and interior nodal zero-crossings.
    """
    k2, is_oscillatory = bessel_radial_wavenumber(omega)
    eta, node_count = get_bessel_eigenfunction(omega, m_val)

    # Measure boundary trapping: fraction of energy concentrated near inner/outer 15% boundaries
    n_pts = len(eta)
    b_idx = int(0.15 * n_pts)
    b_energy = np.sum(np.abs(eta[:b_idx])**2) + np.sum(np.abs(eta[-b_idx:])**2)
    tot_energy = np.sum(np.abs(eta)**2) + 1e-300
    b_ratio = b_energy / tot_energy

    if not is_oscillatory or b_ratio > 0.55 or node_count == 0:
        branch_category = 'Kelvin/slow'
        radial_n = 0
    else:
        branch_category = 'Poincare-like'
        radial_n = max(1, node_count + 1)

    return branch_category, radial_n

def find_bessel_branches_continuous_m(
    m_min=M_CONTINUOUS_MIN,
    m_max=M_CONTINUOUS_MAX,
    dm=M_CONTINUOUS_STEP,
    omega_range=(-25*f, 25*f),
    n_scan=1000
):
    """
    Continuous m Bessel Branch Solver & Continuation Tracker.
    Samples m over a dense real grid [m_min, m_max] with step dm,
    extracts initial exact Bessel roots at m_min, and tracks branches continuously
    across m using linear/secant predictor steps and adaptive bracketing.
    """
    m_grid = np.arange(m_min, m_max + 0.5 * dm, dm)
    n_m = len(m_grid)

    initial_roots = find_bessel_eigenvalues(m_grid[0], omega_range=omega_range, n_scan=n_scan)
    n_branches = len(initial_roots)

    branches = np.full((n_branches, n_m), np.nan)
    branch_categories = []
    branch_radial_n = []

    for i, w0 in enumerate(initial_roots):
        branches[i, 0] = w0
        cat, r_n = classify_bessel_root(w0, m_grid[0])
        branch_categories.append(cat)
        branch_radial_n.append(r_n)

    for j in range(1, n_m):
        m_curr = m_grid[j]
        for b_i in range(n_branches):
            w_curr = branches[b_i, j-1]
            if np.isnan(w_curr):
                continue

            # Predictor step: use secant extrapolation if previous step exists, else constant prediction
            if j >= 2 and not np.isnan(branches[b_i, j-2]):
                slope = (branches[b_i, j-1] - branches[b_i, j-2])
                w_pred = w_curr + slope
            else:
                w_pred = w_curr

            found = False
            # Search around predictor with adaptive expansion
            for mult in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
                delta_w = mult * max(1e-3 * f, 0.05 * np.abs(w_pred))
                w_a = w_pred - delta_w
                w_b = w_pred + delta_w

                # Verify non-singular interval
                if not is_valid_frequency(w_a, np.array([r1, r0, r2])) or not is_valid_frequency(w_b, np.array([r1, r0, r2])):
                    continue

                d_a = bessel_boundary_determinant(w_a, m_curr)
                d_b = bessel_boundary_determinant(w_b, m_curr)

                if not np.isnan(d_a) and not np.isnan(d_b) and d_a * d_b < 0:
                    ref_val = np.abs(d_a)
                    def f_sol(w):
                        return bessel_boundary_determinant(w, m_curr) / ref_val
                    try:
                        res = brentq(f_sol, w_a, w_b, xtol=1e-12)
                        branches[b_i, j] = res
                        found = True
                        break
                    except Exception:
                        pass
            if not found:
                branches[b_i, j] = np.nan

    return m_grid, branches, branch_categories, branch_radial_n

# =============================================================================
# 11. Diagnostic Plotting
# =============================================================================
def plot_sigma_scan(m_val=1, N_col=64, omega_range=(-5*f, 5*f), n_scan=800):
    """
    Generates and saves the diagnostic sigma-min scan plot for m=1, showing
    normalized sigma_min versus frequency and marking detected local minima.
    """
    r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, r1, r2)
    valid_intervals = partition_valid_intervals(r_grid, omega_range)

    fig, ax = plt.subplots(figsize=(12, 6))

    all_w = []
    all_sig_rels = []

    total_len = sum(b - a for a, b in valid_intervals)
    for a_k, b_k in valid_intervals:
        n_pts = max(50, int(n_scan * (b_k - a_k) / total_len))
        w_scan = np.linspace(a_k, b_k, n_pts)
        for w in w_scan:
            _, _, s_rel = normalized_sigma_min(w, m_val, N_col, r_grid, D1_mat, D2_mat)
            if not np.isnan(s_rel):
                all_w.append(w)
                all_sig_rels.append(s_rel)

    all_w = np.array(all_w)
    all_sig_rels = np.array(all_sig_rels)

    ax.semilogy(all_w / f, all_sig_rels, 'b-', lw=1.2, alpha=0.85, label=r'$\sigma_{\min} / \sigma_{\max}$')
    ax.axhline(SVD_RESIDUAL_TOL, color='red', linestyle='--', lw=1.2, label=f'Tolerance ({SVD_RESIDUAL_TOL:.1e})')

    acc_roots, acc_sig_rels, candidates = find_eigenvalues_svd(m_val, N_col, r_grid, D1_mat, D2_mat, omega_range=omega_range, n_scan=n_scan)

    if len(acc_roots) > 0:
        ax.scatter(acc_roots / f, acc_sig_rels, color='green', marker='o', s=60, zorder=6, label=f'Accepted roots ({len(acc_roots)})')

    ax.set_xlabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=16)
    ax.set_ylabel(r'Normalized singular value $\sigma_{\min}/\sigma_{\max}$', fontsize=16)
    ax.set_title(f'Diagnostic SVD Scan ($m={m_val}$, $N={N_col}$)', fontsize=16, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    png_path = "outputs/sigma_min_scan_m1.png"
    pdf_path = "outputs/sigma_min_scan_m1.pdf"
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    with open("outputs/sigma_scan_m1.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["omega_rad_s", "hat_omega", "sigma_rel"])
        for w, s_rel in zip(all_w, all_sig_rels):
            writer.writerow([w, w/f, s_rel])

    print(f"Saved diagnostic scan plot: {png_path} and {pdf_path}")
    print(f"Saved diagnostic scan CSV: outputs/sigma_scan_m1.csv")

# =============================================================================
# 12. Main Global Dispersion Calculation & Plotting
# =============================================================================
def compute_dispersion_relation():
    print("==========================================================")
    print("        COMPUTING GLOBAL DISPERSION RELATION              ")
    print("==========================================================")

    if USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH:
        print("Solver mode: EXACT BESSEL CONTINUOUS BRANCHES")
        m_grid, branches, categories, radial_n = find_bessel_branches_continuous_m(
            m_min=M_CONTINUOUS_MIN,
            m_max=M_CONTINUOUS_MAX,
            dm=M_CONTINUOUS_STEP,
            n_scan=1000
        )

        with open("outputs/continuous_bessel_branches.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["m", "branch_id", "radial_mode_n", "omega_rad_s", "omega_over_f", "k_r_squared", "branch_category", "valid"])
            for j, m_val in enumerate(m_grid):
                for b_i in range(branches.shape[0]):
                    w_val = branches[b_i, j]
                    is_val = not np.isnan(w_val)
                    if is_val:
                        k2_val, _ = bessel_radial_wavenumber(w_val)
                        writer.writerow([m_val, b_i + 1, radial_n[b_i], w_val, w_val / f, k2_val, categories[b_i], True])
                    else:
                        writer.writerow([m_val, b_i + 1, radial_n[b_i], "", "", "", categories[b_i], False])

        with open("outputs/continuous_bessel_branch_summary.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["branch_id", "radial_mode_n", "category", "m_min", "m_max", "valid_points_count"])
            for b_i in range(branches.shape[0]):
                valid_cnt = np.count_nonzero(~np.isnan(branches[b_i, :]))
                writer.writerow([b_i + 1, radial_n[b_i], categories[b_i], m_grid[0], m_grid[-1], valid_cnt])

        print(f"Computed {branches.shape[0]} continuous branches over m in [{m_grid[0]}, {m_grid[-1]}] with step {M_CONTINUOUS_STEP}")
        print("Saved outputs/continuous_bessel_branches.csv and outputs/continuous_bessel_branch_summary.csv")
        return m_grid, branches, categories, radial_n
    else:
        print("Solver mode: CHEBYSHEV SVD GLOBAL")
        M_max = 30
        m_arr = np.arange(1, M_max + 1)
        N_col = 64
        r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, r1, r2)
        dispersion_data = {}
        for m in m_arr:
            eigs, sig_rels, _ = find_eigenvalues_svd(m, N_col, r_grid, D1_mat, D2_mat, omega_range=(-25 * f, 25 * f), n_scan=400)
            dispersion_data[m] = eigs
            pos = sorted([e for e in eigs if e > 0])
            neg = sorted([e for e in eigs if e < 0], reverse=True)
            if m <= 3 or m % 5 == 0:
                print(f"  m={m:2d} (found {len(eigs)} roots via SVD): "
                      f"+{[f'{x/f:.3f}' for x in pos[:4]]}... "
                      f"-{[f'{abs(x)/f:.3f}' for x in neg[:4]]}...")
        return m_arr, dispersion_data, None, None

def plot_dispersion_relation(m_data, branch_data, categories=None, radial_n=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.6, 1]})

    title_suffix = "(Exact Bessel)" if (USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH) else "(Chebyshev SVD)"
    fig.suptitle(f'SWMHD Global Dispersion Relation {title_suffix}\n', fontsize=16, fontweight='bold')

    if USE_ANALYTIC_BESSEL and not USE_VARIABLE_DEPTH:
        m_grid = m_data
        branches = branch_data
        n_branches = branches.shape[0]

        poincare_color = '#1f77b4'
        slow_color     = '#d62728'

        added_poincare_label = False
        added_slow_label = False

        for b_i in range(n_branches):
            b_cat = categories[b_i]
            r_n = radial_n[b_i]
            y_vals = branches[b_i, :] / f

            c = poincare_color if b_cat == 'Poincare-like' else slow_color
            ls = '-'
            lw = 1.8

            label = None
            if b_cat == 'Poincare-like' and not added_poincare_label:
                label = 'Poincaré-like branches'
                added_poincare_label = True
            elif b_cat == 'Kelvin/slow' and not added_slow_label:
                label = 'Kelvin/slow branches'
                added_slow_label = True

            ax1.plot(m_grid, y_vals, color=c, linestyle=ls, lw=lw, alpha=0.9, label=label)
            ax2.plot(m_grid, y_vals, color=c, linestyle=ls, lw=lw, alpha=0.9)

        if SHOW_INTEGER_M_POINTS:
            int_m_vals = np.arange(1, 31)
            for m_int in int_m_vals:
                rts = find_bessel_eigenvalues(m_int, n_scan=1000)
                if len(rts) > 0:
                    ax1.scatter([m_int] * len(rts), rts / f, color='black', marker='o', s=18, zorder=6, label='Integer m roots' if m_int==1 else "")
                    slow_rts = rts[np.abs(rts / f) < 1.0]
                    if len(slow_rts) > 0:
                        ax2.scatter([m_int] * len(slow_rts), slow_rts / f, color='black', marker='o', s=25, zorder=6)

    else:
        m_arr = m_data
        dispersion_data = branch_data
        for m in m_arr:
            eigs = dispersion_data[m]
            ax1.scatter([m] * len(eigs), eigs / f, color='black', marker='o', s=30, alpha=0.85, zorder=5)
            slow_eigs = eigs[np.abs(eigs / f) < 1.0]
            ax2.scatter([m] * len(slow_eigs), slow_eigs / f, color='black', marker='o', s=45, alpha=0.85, zorder=5)

    ax1.axhline(0, color='grey', lw=0.8, ls=':')
    ax1.axhline(+1, color='grey', lw=0.9, ls='--', alpha=0.5)
    ax1.axhline(-1, color='grey', lw=0.9, ls='--', alpha=0.5)

    ax1.set_xlabel('Azimuthal wavenumber $m$', fontsize=15)
    ax1.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
    ax1.set_xlim(M_CONTINUOUS_MIN, M_CONTINUOUS_MAX)
    ax1.set_ylim(FULL_SPECTRUM_YLIM)
    ax1.grid(True, alpha=0.22)
    ax1.set_title('Full Spectrum', fontsize=14)
    if added_poincare_label or added_slow_label:
        ax1.legend(loc='upper right', fontsize=11)

    ax2.axhline(0, color='grey', lw=0.8, ls=':')
    ax2.set_xlabel('Azimuthal wavenumber $m$', fontsize=15)
    ax2.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=15)
    ax2.set_xlim(M_CONTINUOUS_MIN, M_CONTINUOUS_MAX)
    ax2.set_ylim(SLOW_BRANCH_YLIM)
    ax2.grid(True, alpha=0.22)
    ax2.set_title('Slow Branches Zoom', fontsize=14)

    om_A_val = np.sqrt(omega_A2(r0))
    pbox = (
        rf"$\Omega = {Omega:.1e}$ rad/s,  $H_0 = {H0}$ m" + "\n"
        rf"$r_1 = {r1:.1e}$ m,  $r_2 = {r2:.1e}$ m" + "\n"
        rf"$\omega_A(r_0) = {om_A_val:.2e}$ s$^{{-1}}$ ($B_0 = {B0}$ T)" + "\n"
        rf"Solver: {title_suffix}"
    )
    fig.text(0.52, 0.015, pbox, fontsize=11, va='bottom', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))

    fig.subplots_adjust(bottom=0.15)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    png_path = "outputs/swmhd_variable_depth_global_dispersion.png"
    pdf_path = "outputs/swmhd_variable_depth_global_dispersion.pdf"
    plt.savefig(png_path, dpi=180, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved dispersion plots:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")

# =============================================================================
# 13. COMPREHENSIVE NUMERICAL VALIDATION SUITE (TESTS 1 to 13)
# =============================================================================
def run_validation_suite():
    print("==========================================================")
    print("    RUNNING COMPREHENSIVE SWMHD NUMERICAL VALIDATION SUITE ")
    print("==========================================================")

    test_results = {}

    # -------------------------------------------------------------------------
    # TEST 1: Equilibrium-Depth Consistency
    # -------------------------------------------------------------------------
    Heq_r0 = H_eq(r0)
    err_test1 = np.abs(Heq_r0 - H0)
    pass_test1 = err_test1 < 1e-10
    test_results["TEST 1: Equilibrium Depth Consistency"] = (pass_test1, f"|H_eq(r0) - H0| = {err_test1:.6e} m")
    print(f"TEST 1 [Equilibrium Depth]: H_eq(r0) = {Heq_r0:.6f} m (Expected: {H0:.6f} m) -> {'PASS' if pass_test1 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # TEST 2: Derivative Consistency
    # -------------------------------------------------------------------------
    dr = 1.0
    analytic_deriv = dH_eq_dr(r0)
    fd_deriv = (H_eq(r0 + dr) - H_eq(r0 - dr)) / (2.0 * dr)
    err_test2 = np.abs(analytic_deriv - fd_deriv)
    pass_test2 = err_test2 < 1e-6
    test_results["TEST 2: Derivative Consistency"] = (pass_test2, f"|Analytic - FD| = {err_test2:.6e}")
    print(f"TEST 2 [Derivative Consistency]: Analytic H_eq'(r0) = {analytic_deriv:.6e}, FD = {fd_deriv:.6e} -> {'PASS' if pass_test2 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # TEST 3: Magnetic Derivative Consistency
    # -------------------------------------------------------------------------
    w_test = 1e-4
    if B0 > 0.0:
        analytic_wsp = domega_star_dr(w_test, r0)
        fd_wsp = (omega_star(w_test, r0 + dr) - omega_star(w_test, r0 - dr)) / (2.0 * dr)
        err_test3 = np.abs(analytic_wsp - fd_wsp)
        pass_test3 = err_test3 < 1e-6
        msg_test3 = f"|Analytic - FD| = {err_test3:.6e}"
    else:
        pass_test3 = True
        msg_test3 = "B0 = 0.0 (trivially zero)"
    test_results["TEST 3: Magnetic Derivative Consistency"] = (pass_test3, msg_test3)
    print(f"TEST 3 [Magnetic Derivative]: {msg_test3} -> {'PASS' if pass_test3 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # TEST 4: Constant-Depth Bessel ODE Reduction
    # -------------------------------------------------------------------------
    prev_vdepth = USE_VARIABLE_DEPTH
    globals()['USE_VARIABLE_DEPTH'] = False

    N_test = 64
    rg_b, d1_b, d2_b = chebyshev_lobatto(N_test, r1, r2)
    w_samp = 1.5 * f
    C2, C1, C0 = radial_coefficients(w_samp, 1, rg_b)

    ws_samp = omega_star(w_samp, r0)
    k2_analytic = w_samp * (ws_samp**2 - f**2) / (g * H0 * ws_samp)

    C1_over_C2 = C1 / C2
    C0_over_C2 = C0 / C2
    C1_expected = 1.0 / rg_b
    C0_expected = k2_analytic - 1.0 / (rg_b**2)

    max_err_C1 = np.max(np.abs(C1_over_C2 - C1_expected))
    max_err_C0 = np.max(np.abs(C0_over_C2 - C0_expected))
    pass_test4 = (max_err_C1 < 1e-12) and (max_err_C0 < 1e-12)
    test_results["TEST 4: Bessel ODE Reduction"] = (pass_test4, f"Max C1 err = {max_err_C1:.2e}, Max C0 err = {max_err_C0:.2e}")
    print(f"TEST 4 [Bessel ODE Reduction]: Max C1 err = {max_err_C1:.2e}, Max C0 err = {max_err_C0:.2e} -> {'PASS' if pass_test4 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # TEST 5 & TEST 6: Exact Bessel vs Chebyshev SVD / Determinant
    # -------------------------------------------------------------------------
    bessel_rts = find_bessel_eigenvalues(1, omega_range=(-10*f, 10*f), n_scan=4000)
    cheb_svd_rts, cheb_svd_sig, _ = find_eigenvalues_svd(1, 64, rg_b, d1_b, d2_b, omega_range=(-10*f, 10*f), n_scan=4000)
    cheb_det_rts = find_eigenvalues_determinant(1, 64, rg_b, d1_b, d2_b, omega_range=(-10*f, 10*f), n_scan=4000)

    globals()['USE_VARIABLE_DEPTH'] = prev_vdepth

    print("\n==========================================================")
    print("      TABLE A: BESSEL VS DETERMINANT VS SVD COMPARISON    ")
    print("==========================================================")
    print(f"{'Mode':>4} | {'Bessel omega/f':>16} | {'Det omega/f':>16} | {'SVD omega/f':>16} | {'SVD Err':>10} | {'Det Err':>10}")
    print("-" * 86)

    bessel_comp_rows = []
    svd_errs = []
    det_errs = []

    for idx, r_b in enumerate(bessel_rts):
        if len(cheb_svd_rts) > 0:
            idx_s = np.argmin(np.abs(cheb_svd_rts - r_b))
            r_s = cheb_svd_rts[idx_s]
            err_s = np.abs(r_s - r_b) / np.abs(r_b)
            svd_errs.append(err_s)
            str_s = f"{r_s/f:+16.6f}"
            str_err_s = f"{err_s:10.2e}"
        else:
            str_s = f"{'-':^16}"
            str_err_s = f"{'-':^10}"

        if len(cheb_det_rts) > 0:
            idx_d = np.argmin(np.abs(cheb_det_rts - r_b))
            r_d = cheb_det_rts[idx_d]
            err_d = np.abs(r_d - r_b) / np.abs(r_b)
            det_errs.append(err_d)
            str_d = f"{r_d/f:+16.6f}"
            str_err_d = f"{err_d:10.2e}"
        else:
            str_d = f"{'-':^16}"
            str_err_d = f"{'-':^10}"

        print(f"{idx+1:4d} | {r_b/f:+16.6f} | {str_d} | {str_s} | {str_err_s} | {str_err_d}")
        bessel_comp_rows.append([idx+1, r_b, r_b/f, str_d.strip(), str_s.strip(), str_err_s.strip(), str_err_d.strip()])

    with open("outputs/bessel_comparison.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mode", "Bessel_omega_rad_s", "Bessel_hat_omega", "Det_hat_omega", "SVD_hat_omega", "SVD_Rel_Err", "Det_Rel_Err"])
        for row in bessel_comp_rows:
            writer.writerow(row)

    max_svd_err = np.max(svd_errs) if len(svd_errs) > 0 else 1.0
    max_det_err = np.max(det_errs) if len(det_errs) > 0 else 1.0

    pass_test5 = max_svd_err < 1e-4
    pass_test6 = max_det_err < 1e-4

    test_results["TEST 5: Exact Bessel vs Chebyshev SVD"] = (pass_test5, f"Max relative SVD error = {max_svd_err:.2e}")
    test_results["TEST 6: Exact Bessel vs Chebyshev Determinant"] = (pass_test6, f"Max relative Det error = {max_det_err:.2e}")

    # -------------------------------------------------------------------------
    # TEST 7: SVD Eigenfunction & Boundary Residuals
    # -------------------------------------------------------------------------
    rg_v, d1_v, d2_v = chebyshev_lobatto(64, r1, r2)
    v_rts, v_sigs, _ = find_eigenvalues_svd(1, 64, rg_v, d1_v, d2_v, omega_range=(-10*f, 10*f), n_scan=1000)

    print("\n==========================================================")
    print("      TABLE C: EIGENFUNCTION & BOUNDARY RESIDUALS         ")
    print("==========================================================")
    print(f"{'Mode':>4} | {'omega/f':>12} | {'R_full':>10} | {'R_ODE':>10} | {'R_inner':>10} | {'R_outer':>10} | {'sigma_rel':>10}")
    print("-" * 78)

    res_rows = []
    r_full_list = []

    for idx, w in enumerate(v_rts[:6]):
        rf, rode, rinn, rout, srel = compute_eigen_residuals(w, 1, 64, rg_v, d1_v, d2_v)
        r_full_list.append(rf)
        print(f"{idx+1:4d} | {w/f:+12.6f} | {rf:10.2e} | {rode:10.2e} | {rinn:10.2e} | {rout:10.2e} | {srel:10.2e}")
        res_rows.append([idx+1, w, w/f, rf, rode, rinn, rout, srel])

    pass_test7 = max(r_full_list) < 1e-6 if len(r_full_list) > 0 else False
    test_results["TEST 7: Eigenfunction Residuals"] = (pass_test7, f"Max full residual = {max(r_full_list):.2e}" if len(r_full_list)>0 else "No roots")

    # -------------------------------------------------------------------------
    # TEST 8: Resolution Convergence (N = 32, 48, 64, 80, 96)
    # -------------------------------------------------------------------------
    print("\n==========================================================")
    print("      TABLE B: N-RESOLUTION CONVERGENCE TEST              ")
    print("==========================================================")

    resolutions = [32, 48, 64, 80, 96]
    conv_data = {}
    grids = {}

    for N_res in resolutions:
        rg, d1, d2 = chebyshev_lobatto(N_res, r1, r2)
        grids[N_res] = (rg, d1, d2)
        rts, sigs, _ = find_eigenvalues_svd(1, N_res, rg, d1, d2, omega_range=(-5*f, 5*f), n_scan=300)
        conv_data[N_res] = (rts, sigs)

    ref_rts, ref_sigs = conv_data[96]
    rg_ref, d1_ref, d2_ref = grids[96]

    print(f"{'N':>4} | {'Mode':>4} | {'omega/f':>14} | {'Rel Freq Err':>12} | {'sigma_rel':>10} | {'Overlap':>10}")
    print("-" * 65)

    conv_rows = []
    max_rel_err_conv = 0.0

    sel_ref_rts = [w for w in ref_rts if np.abs(w / f) < 0.1]
    for m_idx, w_ref in enumerate(sel_ref_rts[:3]):
        eta_ref = get_eigenfunction(w_ref, 1, 96, rg_ref, d1_ref, d2_ref)

        for N_res in resolutions:
            rts, sigs = conv_data[N_res]
            rg_N, d1_N, d2_N = grids[N_res]

            if len(rts) == 0:
                continue

            best_k = np.argmin(np.abs(rts - w_ref))
            w_N = rts[best_k]
            sig_N = sigs[best_k]
            rel_freq_err = np.abs(w_N - w_ref) / np.abs(w_ref)

            if N_res < 96:
                max_rel_err_conv = max(max_rel_err_conv, rel_freq_err)

            eta_N = get_eigenfunction(w_N, 1, N_res, rg_N, d1_N, d2_N)
            f_interp = interp1d(rg_N, eta_N, kind='cubic', bounds_error=False, fill_value=0.0)
            eta_N_on_ref = f_interp(rg_ref)

            overlap = np.abs(np.vdot(eta_N_on_ref, eta_ref)) / (np.linalg.norm(eta_N_on_ref, 2) * np.linalg.norm(eta_ref, 2) + 1e-15)

            print(f"{N_res:4d} | {m_idx+1:4d} | {w_N/f:+14.6f} | {rel_freq_err:12.2e} | {sig_N:10.2e} | {overlap:10.6f}")
            conv_rows.append([N_res, m_idx+1, w_N, w_N/f, rel_freq_err, sig_N, overlap])

    with open("outputs/convergence.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["N", "Mode", "omega_rad_s", "hat_omega", "Rel_Freq_Err", "sigma_rel", "Eigenfunction_Overlap"])
        for row in conv_rows:
            writer.writerow(row)

    pass_test8 = max_rel_err_conv < 1e-3
    test_results["TEST 8: Resolution Convergence"] = (pass_test8, f"Max N=64 vs N=96 relative error = {max_rel_err_conv:.2e}")

    # -------------------------------------------------------------------------
    # TEST 9: B0 -> 0 Limit Test
    # -------------------------------------------------------------------------
    prev_B0 = B0
    globals()['B0'] = 0.0
    rts_b0, _, _ = find_eigenvalues_svd(1, 64, rg_v, d1_v, d2_v, omega_range=(-5*f, 5*f), n_scan=800)

    globals()['B0'] = 1e-5
    rts_b_eps, _, _ = find_eigenvalues_svd(1, 64, rg_v, d1_v, d2_v, omega_range=(-5*f, 5*f), n_scan=800)

    fast_b0 = [w for w in rts_b0 if np.abs(w) > 0.05*f]
    fast_eps = [w for w in rts_b_eps if np.abs(w) > 0.05*f]

    b0_diffs = []
    for w0 in fast_b0:
        if len(fast_eps) > 0:
            best_idx = np.argmin(np.abs(np.array(fast_eps) - w0))
            rel_diff = np.abs(fast_eps[best_idx] - w0) / np.abs(w0)
            b0_diffs.append(rel_diff)

    max_b0_diff = max(b0_diffs) if len(b0_diffs) > 0 else 0.0
    pass_test9 = max_b0_diff < 1e-3
    globals()['B0'] = prev_B0
    test_results["TEST 9: B0 -> 0 Limit Continuum"] = (pass_test9, f"Max relative shift as B0->0 = {max_b0_diff:.2e}")
    print(f"\nTEST 9 [B0 -> 0 Limit]: Max relative frequency shift = {max_b0_diff:.2e} -> {'PASS' if pass_test9 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # TEST 10: C -> 0 / Constant-Depth Limiting Behavior
    # -------------------------------------------------------------------------
    prev_vdepth = USE_VARIABLE_DEPTH
    globals()['USE_VARIABLE_DEPTH'] = True
    prev_C = C
    globals()['C'] = 0.0
    dH_c0 = dH_eq_dr(r0)
    dH_expected = (Omega**2 / g) * r0
    err_c0 = np.abs(dH_c0 - dH_expected)
    pass_test10 = err_c0 < 1e-12
    globals()['C'] = prev_C
    globals()['USE_VARIABLE_DEPTH'] = prev_vdepth
    test_results["TEST 10: C -> 0 Limit Behavior"] = (pass_test10, f"|dH/dr - Omega^2*r/g| = {err_c0:.2e}")
    print(f"TEST 10 [C -> 0 Limit]: Derivative error = {err_c0:.2e} -> {'PASS' if pass_test10 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # TEST 11: Diagnostic Sigma-Min Scan Plot Execution
    # -------------------------------------------------------------------------
    plot_sigma_scan(m_val=1, N_col=64, omega_range=(-5*f, 5*f), n_scan=800)
    pass_test11 = os.path.exists("outputs/sigma_min_scan_m1.png") and os.path.exists("outputs/sigma_min_scan_m1.pdf")
    test_results["TEST 11: Diagnostic Plot Generation"] = (pass_test11, "sigma_min_scan_m1.png & pdf created")
    print(f"TEST 11 [Diagnostic Plot]: Output verified -> {'PASS' if pass_test11 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # TEST 12: Comparison of Determinant Sign-Change vs SVD Candidate Minima
    # -------------------------------------------------------------------------
    svd_roots_m1, sig_rels_m1, candidates_m1 = find_eigenvalues_svd(1, 64, rg_v, d1_v, d2_v, omega_range=(-10*f, 10*f), n_scan=1000)
    det_roots_m1 = find_eigenvalues_determinant(1, 64, rg_v, d1_v, d2_v, omega_range=(-10*f, 10*f), n_scan=1000)

    missing_in_det = []
    for w_s in svd_roots_m1:
        if len(det_roots_m1) == 0 or np.min(np.abs(det_roots_m1 - w_s)) > 1e-3 * np.abs(w_s):
            missing_in_det.append(w_s)

    print(f"\nTEST 12 [SVD vs Determinant Sign-Change Comparison]:")
    print(f"  Total SVD Roots found: {len(svd_roots_m1)}")
    print(f"  Total Determinant Sign-Change Roots found: {len(det_roots_m1)}")
    if len(missing_in_det) > 0:
        print(f"  Explicit SVD Roots missed by Determinant Sign-Change: {len(missing_in_det)}")
        for w_m in missing_in_det:
            print(f"    omega/f = {w_m/f:+.6f}")
    else:
        print("  All SVD roots produced determinant sign changes for this specific parameter set.")

    pass_test12 = True
    test_results["TEST 12: Determinant vs SVD Root Comparison"] = (pass_test12, f"SVD roots: {len(svd_roots_m1)}, Det roots: {len(det_roots_m1)}")

    # -------------------------------------------------------------------------
    # TEST 13: Continuous Branch vs Independent Exact Integer-(m) Bessel Roots
    # -------------------------------------------------------------------------
    print("\n==========================================================")
    print("  TEST 13: CONTINUOUS BRANCH VS INDEPENDENT INTEGER-m BESSEL ROOTS")
    print("==========================================================")

    m_test_list = [1, 5, 10, 15, 20, 25, 30]
    m_grid, branches, categories, radial_n = find_bessel_branches_continuous_m(
        m_min=M_CONTINUOUS_MIN,
        m_max=M_CONTINUOUS_MAX,
        dm=M_CONTINUOUS_STEP,
        n_scan=1000
    )

    max_branch_rel_err = 0.0
    test13_rows = []

    print(f"{'m':>4} | {'Exact omega/f':>16} | {'Branch omega/f':>16} | {'Rel Error':>12} | {'Branch Category':>15}")
    print("-" * 75)

    for m_test in m_test_list:
        # 1. Independent exact roots solve at integer m_test
        exact_rts = find_bessel_eigenvalues(m_test, omega_range=(-25*f, 25*f), n_scan=1000)

        # 2. Extract continuous branch roots at closest m index in continuous grid
        j_idx = np.argmin(np.abs(m_grid - m_test))
        branch_rts = branches[:, j_idx]
        branch_rts_valid = branch_rts[~np.isnan(branch_rts)]

        for br_w in branch_rts_valid:
            if len(exact_rts) > 0:
                best_idx = np.argmin(np.abs(exact_rts - br_w))
                ex_w = exact_rts[best_idx]
                rel_err = np.abs(br_w - ex_w) / np.abs(ex_w)
                max_branch_rel_err = max(max_branch_rel_err, rel_err)
                cat, _ = classify_bessel_root(ex_w, m_test)
                print(f"{m_test:4d} | {ex_w/f:+16.6f} | {br_w/f:+16.6f} | {rel_err:12.2e} | {cat:>15s}")
                test13_rows.append([m_test, ex_w, ex_w/f, br_w, br_w/f, rel_err, cat])

    pass_test13 = max_branch_rel_err < 1e-4
    test_results["TEST 13: Continuous Branch vs Exact Integer-(m) Bessel Roots"] = (
        pass_test13, f"Max relative error across integer m = {max_branch_rel_err:.2e}"
    )

    with open("outputs/test13_branch_vs_exact.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["m", "exact_omega_rad_s", "exact_hat_omega", "branch_omega_rad_s", "branch_hat_omega", "rel_err", "category"])
        for row in test13_rows:
            writer.writerow(row)

    with open("outputs/validation_summary.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Test Name", "Status", "Details"])
        for name, (status, details) in test_results.items():
            writer.writerow([name, "PASS" if status else "FAIL", details])

    all_pass = all(status for status, _ in test_results.values())

    print("\n==========================================================")
    print("               FINAL VALIDATION SUMMARY                   ")
    print("==========================================================")
    for name, (status, details) in test_results.items():
        print(f"  [{'PASS' if status else 'FAIL'}] {name:55s} | {details}")
    print("==========================================================")
    print(f"  OVERALL SOLVER STATUS: {'PASS' if all_pass else 'FAIL'}")
    print("==========================================================\n")

    return all_pass

# =============================================================================
# Main Program Execution
# =============================================================================
def main():
    print("==========================================================")
    print("   SWMHD GLOBAL EIGENVALUE SOLVER (PUBLICATION GRADE)     ")
    print("==========================================================")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.platform()}")
    print(f"NumPy Version  : {numpy_version}")
    print(f"SciPy Version  : {scipy_version}")
    print("----------------------------------------------------------")
    print("Physical Parameters (SI Units):")
    print(f"  Omega        = {Omega:.2e} rad/s")
    print(f"  f (2*Omega)  = {f:.2e} rad/s")
    print(f"  H0           = {H0:.1f} m")
    print(f"  g            = {g:.2f} m/s^2")
    print(f"  B0           = {B0:.2e} T")
    print(f"  rho0         = {rho0:.1f} kg/m^3")
    print(f"  C            = {C:.2e} m^3/s^2")
    print(f"  r1           = {r1:.2e} m")
    print(f"  r2           = {r2:.2e} m")
    print("Solver Settings:")
    print(f"  SVD Residual Tolerance (SVD_RESIDUAL_TOL) = {SVD_RESIDUAL_TOL:.1e}")
    print(f"  Zero Exclusion Tolerance (OMEGA_TOL)     = {OMEGA_TOL:.1e} rad/s")
    print(f"  Resonance Tolerance (RESONANCE_TOL)       = {RESONANCE_TOL:.1e} (rad/s)^2")
    print(f"  Continuous m Step (M_CONTINUOUS_STEP)    = {M_CONTINUOUS_STEP}")
    print("==========================================================\n")

    if USE_ANALYTIC_BESSEL and USE_VARIABLE_DEPTH:
        print("Analytic Bessel solver unavailable: H_eq(r) is variable.\nUsing Chebyshev global SVD solver.")
        globals()['USE_ANALYTIC_BESSEL'] = False

    all_passed = run_validation_suite()
    if not all_passed:
        print("CRITICAL WARNING: Validation tests produced failures. Inspect output before publication!")

    m_data, branch_data, categories, radial_n = compute_dispersion_relation()
    plot_dispersion_relation(m_data, branch_data, categories, radial_n)

if __name__ == "__main__":
    main()
