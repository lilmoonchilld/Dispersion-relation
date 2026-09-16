import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals
from scipy.optimize import minimize_scalar
import warnings
import os

warnings.filterwarnings("ignore")

# ==========================================================
# 1. Imports / Style
# ==========================================================
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

# ==========================================================
# 2. Parameter Configuration
# ==========================================================

class Params:
    MU0 = 4 * np.pi * 1e-7
    Omega = 0.5e-4
    H0 = 500.0
    g = 9.81
    B0 = 5e-4
    rho0 = 1000.0
    C = 9.375e9
    r1 = 0.5e6
    r2 = 1.0e6

    # Derived parameters initialized below
    r0 = 0.0
    f = 0.0
    r_WKB = 0.0

    # Configurations
    constant_depth_override = False

# ==========================================================
# 3. Case Configuration
# ==========================================================

def set_case_parameters(case_num, p: Params, r_WKB_override=None):
    # Base resets
    p.C = 9.375e9
    p.B0 = 5e-4
    p.constant_depth_override = False

    if case_num == 1:
        p.B0 = 0.0
        p.C = 0.0
    elif case_num == 2:
        p.B0 = 0.0
    elif case_num == 3:
        p.C = 0.0
    elif case_num == 4:
        pass # default variable depth magnetic
    elif case_num == 5:
        p.B0 = 0.0
        p.C = 0.0
        p.constant_depth_override = True
    else:
        raise ValueError(f"Unknown case number {case_num}")

    p.r0 = (p.r1 + p.r2) / 2.0
    p.f = 2.0 * p.Omega
    p.r_WKB = p.r0 if r_WKB_override is None else r_WKB_override

p = Params()

# ==========================================================
# 4. Equilibrium Depth
# ==========================================================

def H_eq(r, p: Params):
    if p.constant_depth_override:
        return np.full_like(r, p.H0, dtype=float) if isinstance(r, np.ndarray) else p.H0
    return p.H0 + (p.Omega**2 / (2.0 * p.g)) * (r**2 - p.r0**2) + (p.C / p.g) * (1.0/r - 1.0/p.r0)

def dH_eq_dr(r, p: Params):
    if p.constant_depth_override:
        return np.zeros_like(r, dtype=float) if isinstance(r, np.ndarray) else 0.0
    return (p.Omega**2 * r) / p.g - p.C / (p.g * r**2)

def Q(r, p: Params):
    return p.g * dH_eq_dr(r, p)

# ==========================================================
# 5. Magnetic Functions
# ==========================================================

def omega_A_sq(r, p: Params):
    return p.B0**2 / (p.MU0 * p.rho0 * H_eq(r, p)**2)

def omega_star(r, omega, p: Params):
    return omega + omega_A_sq(r, p) / omega

def domega_star_dr(r, omega, p: Params):
    if p.B0 == 0.0:
        return np.zeros_like(r, dtype=float) if isinstance(r, np.ndarray) else 0.0
    return - (2.0 * omega_A_sq(r, p) / omega) * (dH_eq_dr(r, p) / H_eq(r, p))

# ==========================================================
# Dynamic Scan Range Calculator
# ==========================================================

def calculate_scan_range(p: Params, M_max=10):
    # Dynamic bounds based on f, omega_A_max, max(sqrt(gH) * kappa), and omega_A^2/f
    r_test = np.linspace(p.r1, p.r2, 100)
    oma2_vals = omega_A_sq(r_test, p)
    oma_max = np.sqrt(np.max(oma2_vals))
    h_max = np.max(H_eq(r_test, p))

    kr_max = 5 * np.pi / (p.r2 - p.r1) # assume up to n=5 for bounds
    kappa_max = np.sqrt(kr_max**2 + (M_max / p.r1)**2)

    poincare_scale = np.sqrt(p.f**2 + p.g * h_max * kappa_max**2)
    ms_scale = np.max(oma2_vals) / p.f if p.f != 0 else 0

    bounds = [
        1.5 * p.f,
        2.5 * oma_max,
        1.2 * poincare_scale,
        5.0 * ms_scale
    ]

    bound_max = max(bounds)
    if np.isnan(bound_max) or bound_max == 0:
        bound_max = 10 * p.Omega

    return (-bound_max, bound_max)


# ==========================================================
# 6. Chebyshev Collocation
# ==========================================================

def chebyshev_lobatto(N, r_min, r_max):
    j = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))

    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0

    xi_col = xi[:, None]
    xi_row = xi[None, :]
    dX = xi_col - xi_row

    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) * (-1.0)**(i + k) / dX[i, k]

    D -= np.diag(D.sum(axis=1))

    sc = 2.0 / (r_max - r_min)
    D1 = sc * D
    D2 = D1 @ D1

    rp = 0.5 * (r_min + r_max) + 0.5 * (r_max - r_min) * xi

    # Permute matrices to align with an ascending radial grid
    rp = rp[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]

    return rp, D1, D2

# ==========================================================
# 7. Numerical Validations
# ==========================================================

def run_numerical_validations(p: Params, N_col=40):
    r_grid, D1, D2 = chebyshev_lobatto(N_col, p.r1, p.r2)

    # A. Depth reference: H_eq(r0) = H0
    h_r0 = H_eq(p.r0, p)
    assert np.isclose(h_r0, p.H0, atol=1e-8, rtol=1e-8), f"Depth at r0 = {h_r0} != {p.H0}"

    # B. Derivative consistency
    h_eq = H_eq(r_grid, p)
    dh_eq_analytic = dH_eq_dr(r_grid, p)
    dh_eq_fd = D1 @ h_eq
    err_dh = np.max(np.abs(dh_eq_fd[1:-1] - dh_eq_analytic[1:-1]))
    assert err_dh < 1e-4, f"H_eq' analytic vs numerical error {err_dh} too large"

    # C. Q identity
    q_vals = Q(r_grid, p)
    err_q = np.max(np.abs(p.g * dh_eq_analytic - q_vals))
    assert err_q < 1e-12, "Q(r) does not match g * H_eq'(r)"

    # D. Magnetic derivative
    if p.B0 != 0.0:
        omega_test = p.Omega
        omega_s = omega_star(r_grid, omega_test, p)
        domega_s_analytic = domega_star_dr(r_grid, omega_test, p)
        domega_s_fd = D1 @ omega_s
        err_domega_s = np.max(np.abs(domega_s_fd[1:-1] - domega_s_analytic[1:-1]))
        assert err_domega_s < 1e-4, f"omega_*' error {err_domega_s} too large"

    # E. Derivative matrix tests
    d1_r = D1 @ r_grid
    err_d1_r = np.max(np.abs(d1_r[1:-1] - 1.0))
    assert err_d1_r < 1e-10, f"D1 @ r != 1, error = {err_d1_r}"

    d1_r2 = D1 @ (r_grid**2)
    err_d1_r2 = np.max(np.abs(d1_r2[1:-1] - 2.0 * r_grid[1:-1]))
    assert err_d1_r2 < 1e-6, f"D1 @ r^2 != 2r, error = {err_d1_r2}"

    # F. Positivity
    assert np.all(h_eq > 0), "H_eq(r) <= 0 found in the computational domain!"

    # Grid ascending test
    assert np.all(np.diff(r_grid) > 0), "Grid is not monotonically ascending."


# ==========================================================
# 8. Variable-depth ODE coefficients & Assembly
# ==========================================================

def get_A(r, omega, p: Params):
    h = H_eq(r, p)
    dh = dH_eq_dr(r, p)
    om_s = omega_star(r, omega, p)
    dom_s = domega_star_dr(r, omega, p)

    term1 = 1.0 / r
    term2 = dh / h
    term3 = dom_s / om_s
    term4 = - (2.0 * om_s * dom_s) / (om_s**2 - p.f**2)

    return term1 + term2 + term3 + term4

def get_B(r, omega, m, p: Params):
    h = H_eq(r, p)
    dh = dH_eq_dr(r, p)
    om_s = omega_star(r, omega, p)
    dom_s = domega_star_dr(r, omega, p)

    part1 = (omega * (om_s**2 - p.f**2)) / (p.g * h * om_s)

    bracket = dh - (2.0 * h * om_s * dom_s) / (om_s**2 - p.f**2)
    part2 = (m * p.f / (r * h * om_s)) * bracket

    part3 = - (m**2) / (r**2)

    return part1 + part2 + part3

def build_operator(omega, m, r_grid, D1, D2, p: Params):
    A_r = get_A(r_grid, omega, p)
    B_r = get_B(r_grid, omega, m, p)

    L = D2 + np.diag(A_r) @ D1 + np.diag(B_r)

    om_s_r1 = omega_star(r_grid[0], omega, p)
    om_s_r2 = omega_star(r_grid[-1], omega, p)

    L[0, :] = om_s_r1 * D1[0, :]
    L[0, 0] += (m * p.f) / r_grid[0]

    L[-1, :] = om_s_r2 * D1[-1, :]
    L[-1, -1] += (m * p.f) / r_grid[-1]

    return L

def extract_eigenfunction(L):
    _, singular_values, Vh = svd(L, full_matrices=False, overwrite_a=True, check_finite=False)
    eta = Vh[-1, :].copy()
    norm_val = np.max(np.abs(eta))
    if norm_val > 0:
        eta /= norm_val
    return eta, float(singular_values[-1])

def reconstruct_vr(r, eta, D1, omega, m, p: Params):
    eta_prime = D1 @ eta
    om_s = omega_star(r, omega, p)

    numerator = om_s * eta_prime + (m * p.f / r) * eta
    denominator = om_s**2 - p.f**2

    vr = -1j * p.g * numerator / denominator
    return vr

def evaluate_mode_diagnostics(omega, m, eta, r_grid, D1, D2, p: Params):
    eta_prime = D1 @ eta

    om_s_r1 = omega_star(r_grid[0], omega, p)
    Rb_1 = om_s_r1 * eta_prime[0] + (m * p.f / r_grid[0]) * eta[0]

    om_s_r2 = omega_star(r_grid[-1], omega, p)
    Rb_2 = om_s_r2 * eta_prime[-1] + (m * p.f / r_grid[-1]) * eta[-1]

    vr = reconstruct_vr(r_grid, eta, D1, omega, m, p)
    vr_r1 = np.abs(vr[0])
    vr_r2 = np.abs(vr[-1])

    A_r = get_A(r_grid, omega, p)
    B_r = get_B(r_grid, omega, m, p)

    eta_double_prime = D2 @ eta
    ODE_residual = eta_double_prime + A_r * eta_prime + B_r * eta

    interior_res = ODE_residual[1:-1]
    interior_eta = eta[1:-1]
    L2_res = np.linalg.norm(interior_res)
    rel_L2 = L2_res / max(np.linalg.norm(interior_eta), 1e-15)

    # Internal operator relative norm check L_eta / eta
    L_matrix = build_operator(omega, m, r_grid, D1, D2, p)
    L_eta = L_matrix @ eta
    interior_L_eta = L_eta[1:-1]
    rel_L = np.linalg.norm(interior_L_eta) / max(np.linalg.norm(interior_eta), 1e-15)

    return {
        "Rb_1": np.abs(Rb_1),
        "Rb_2": np.abs(Rb_2),
        "vr_1": vr_r1,
        "vr_2": vr_r2,
        "L2_res": L2_res,
        "rel_L2": rel_L2,
        "rel_L": rel_L
    }

# ==========================================================
# 9. SVD Nonlinear Frequency Search
# ==========================================================

def get_singular_values(omega, m, r_grid, D1, D2, p: Params):
    if not np.isfinite(omega) or abs(omega) < 1e-12:
        return None
    try:
        om_s = omega_star(r_grid, omega, p)
        if np.any(np.abs(om_s**2 - p.f**2) < 1e-12):
            return None

        L = build_operator(omega, m, r_grid, D1, D2, p)
        return svdvals(L, overwrite_a=True, check_finite=False)
    except Exception:
        return None

def split_scan_segments(scan_points, r_grid, p: Params, zero_tol=1e-8, res_tol=1e-7):
    valid_scan = []

    for w in scan_points:
        if abs(w) < zero_tol:
            continue
        om_s = omega_star(r_grid, w, p)
        s_res = np.min(np.abs(om_s**2 - p.f**2))
        if s_res < res_tol:
            continue
        valid_scan.append(w)

    valid_scan = np.array(valid_scan)
    if len(valid_scan) < 3:
        return []

    base_step = abs(scan_points[1] - scan_points[0]) if len(scan_points) > 1 else 1.0
    jumps = np.where(np.diff(valid_scan) > 1.5 * base_step)[0] + 1
    segments = np.split(valid_scan, jumps)

    return [seg for seg in segments if len(seg) >= 3]

def find_eigenmodes(
    m, r_grid, D1, D2, p: Params,
    omega_range=(-3.0, 3.0),
    n_scan=5000,
    sigma_tol=1e-6,
    sigma_rtol=1e-5,
    duplicate_tol=1e-5
):
    sigma_cache = {}

    def objective(w):
        w = float(w)
        if w not in sigma_cache:
            svals = get_singular_values(w, m, r_grid, D1, D2, p)
            if svals is None:
                sigma_cache[w] = (np.inf, np.inf)
            else:
                sigma_cache[w] = (float(svals[-1]), float(svals[0]))
        return sigma_cache[w][0]

    scan_points = np.linspace(omega_range[0], omega_range[1], n_scan)
    segments = split_scan_segments(scan_points, r_grid, p)

    candidates = []
    for seg in segments:
        s_vals = np.array([objective(w) for w in seg])
        for k in range(1, len(seg) - 1):
            if not np.isfinite(s_vals[k]):
                continue
            if s_vals[k] < s_vals[k-1] and s_vals[k] < s_vals[k+1]:
                candidates.append((s_vals[k], seg[k-1], seg[k], seg[k+1]))

    candidates.sort(key=lambda item: item[0])
    modes = []

    for _, left, center, right in candidates:
        try:
            res = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-11, "maxiter": 150}
            )
            w_opt = float(res.x) if (res.success and np.isfinite(res.x)) else float(center)
        except Exception:
            w_opt = float(center)

        s_min, s_max = sigma_cache.get(w_opt, (np.inf, np.inf))
        if not np.isfinite(s_min):
            continue

        rel_s = s_min / max(s_max, 1e-15)
        if s_min > sigma_tol and rel_s > sigma_rtol:
            continue

        is_duplicate = False
        for i, existing in enumerate(modes):
            if abs(w_opt - existing["omega"]) < duplicate_tol:
                is_duplicate = True
                if s_min < existing["sigma_min"]:
                    L = build_operator(w_opt, m, r_grid, D1, D2, p)
                    eta, _ = extract_eigenfunction(L)
                    modes[i] = {
                        "omega": w_opt,
                        "eta": eta,
                        "sigma_min": s_min,
                        "relative_sigma": rel_s
                    }
                break

        if not is_duplicate:
            L = build_operator(w_opt, m, r_grid, D1, D2, p)
            eta, _ = extract_eigenfunction(L)
            modes.append({
                "omega": w_opt,
                "eta": eta,
                "sigma_min": s_min,
                "relative_sigma": rel_s
            })

    modes.sort(key=lambda item: item["omega"])
    return modes


# ==========================================================
# 10. Local WKB Dispersion Relations
# ==========================================================

def get_wkb_predictions(m, n, p: Params):
    kr = n * np.pi / (p.r2 - p.r1)
    kappa2 = kr**2 + (m / p.r_WKB)**2

    h_wkb = H_eq(p.r_WKB, p)
    q_wkb = Q(p.r_WKB, p)
    oma2_wkb = omega_A_sq(p.r_WKB, p)

    preds = {
        "MP_p": np.nan, "MP_m": np.nan,
        "MK_in": np.nan, "MK_out": np.nan,
        "R": np.nan, "MS": np.nan
    }

    # --------------------------------------------------------
    # A. Hydrodynamic uniform-depth limit (B0=0, constant depth)
    # --------------------------------------------------------
    if p.B0 == 0.0 and p.constant_depth_override:
        om_p = np.sqrt(p.f**2 + p.g * h_wkb * kappa2)
        preds["MP_p"] = om_p
        preds["MP_m"] = -om_p

    # --------------------------------------------------------
    # B. Hydrodynamic variable-depth limit (B0=0, Q!=0)
    # --------------------------------------------------------
    elif p.B0 == 0.0 and not p.constant_depth_override:
        # omega^3 - (f^2 + gH kappa^2) omega + (m f Q)/r = 0
        c1 = 1.0
        c0 = 0.0
        cm1 = - (p.f**2 + p.g * h_wkb * kappa2)
        cm2 = (m * p.f * q_wkb) / p.r_WKB

        roots = np.roots([c1, c0, cm1, cm2])
        real_roots = np.sort(roots[np.isreal(roots)].real)

        if len(real_roots) == 3:
            preds["MP_m"] = real_roots[0]
            preds["R"] = real_roots[1]
            preds["MP_p"] = real_roots[2]

    # --------------------------------------------------------
    # C. Magnetic uniform-depth limit (Q=0, B0!=0, constant depth)
    # --------------------------------------------------------
    elif p.B0 != 0.0 and p.constant_depth_override:
        # (omega^2 + omega_A^2)^2 - f^2 omega^2 - gH kappa^2 (omega^2 + omega_A^2) = 0
        # Let x = omega^2
        # x^2 + (2 oma2 - f^2 - gH kappa^2) x + oma2(oma2 - gH kappa^2) = 0
        A = 1.0
        B_coef = 2.0 * oma2_wkb - p.f**2 - p.g * h_wkb * kappa2
        C_coef = oma2_wkb * (oma2_wkb - p.g * h_wkb * kappa2)

        disc = B_coef**2 - 4.0 * A * C_coef
        if disc >= 0:
            x1 = (-B_coef + np.sqrt(disc)) / 2.0
            x2 = (-B_coef - np.sqrt(disc)) / 2.0

            if x1 > 0:
                preds["MP_p"] = np.sqrt(x1)
                preds["MP_m"] = -np.sqrt(x1)

            if x2 > 0:
                # The prompt instructs: "Magnetostrophic branches are tracked continuously rather than forced to one sign."
                # However, for constant depth, the slow mode is purely symmetrical +- roots. We assign positive to MS for mapping,
                # but will track actual signs during general branch continuation.
                preds["MS"] = -np.sqrt(x2)
                preds["R"] = np.sqrt(x2) # It has both signs symmetrically for uniform depth

    # --------------------------------------------------------
    # D. Variable-depth Magnetic (General)
    # --------------------------------------------------------
    elif p.B0 != 0.0 and not p.constant_depth_override:
        # kappa^2 = [omega*(omega_*^2 - f^2)] / [gH omega_*] + [mfQ] / [r g H omega_*] + [4mf oma2 Q omega] / [r g H N(omega)]

        c_N = [1.0, 0.0, 2.0*oma2_wkb - p.f**2, 0.0, oma2_wkb**2]
        c_N2 = np.polymul(c_N, c_N)

        c_O2_oma2 = [1.0, 0.0, oma2_wkb]
        c_LHS_part = np.polymul(c_O2_oma2, c_N)
        K_val = kappa2 * p.g * h_wkb * p.r_WKB
        c_LHS = [K_val * x for x in c_LHS_part]

        c_R1 = [p.r_WKB * x for x in c_N2]

        c_O = [1.0, 0.0]
        c_R2_part = np.polymul(c_O, c_N)
        c_R2 = [m * p.f * q_wkb * x for x in c_R2_part]

        c_R3_part = np.polymul(c_O, c_O2_oma2)
        c_R3 = [4.0 * m * p.f * oma2_wkb * q_wkb * x for x in c_R3_part]

        def pad_add(p1, p2):
            length = max(len(p1), len(p2))
            return np.pad(p1, (length - len(p1), 0)) + np.pad(p2, (length - len(p2), 0))

        poly = pad_add(c_LHS, [-x for x in c_R1])
        poly = pad_add(poly, [-x for x in c_R2])
        poly = pad_add(poly, [-x for x in c_R3])

        roots = np.roots(poly)
        real_roots = np.sort(roots[np.abs(np.imag(roots)) < 1e-8].real)

        valid_roots = []
        for r in real_roots:
            if abs(r) > 1e-12:
                os = r + oma2_wkb/r
                if abs(os**2 - p.f**2) > 1e-7:
                    # Rational equation consistency check
                    O = r
                    N_O = O**4 + (2*oma2_wkb - p.f**2)*O**2 + oma2_wkb**2
                    R1 = (O*(os**2 - p.f**2))/(p.g*h_wkb*os)
                    R2 = (m*p.f*q_wkb)/(p.r_WKB*p.g*h_wkb*os)
                    R3 = (4*m*p.f*oma2_wkb*q_wkb*O)/(p.r_WKB*p.g*h_wkb*N_O)
                    resid = abs(R1 + R2 + R3 - kappa2)

                    if resid < 1e-3:
                        valid_roots.append(r)

        valid_roots = np.sort(valid_roots)

        if len(valid_roots) >= 2:
            preds["MP_p"] = valid_roots[-1]
            preds["MP_m"] = valid_roots[0]

            # Identify slow branches using approximations as starting locators
            om_R_approx = (m * p.f * q_wkb) / (p.r_WKB * (p.f**2 + p.g * h_wkb * kappa2))
            # Note: MS approximation is ~ omega_A^2 / |f| in magnitude, keeping sign flexible
            om_MS_scale = oma2_wkb / p.f if p.f != 0 else 0.0

            middle_roots = valid_roots[1:-1]
            if len(middle_roots) > 0:
                idx_R = np.argmin(np.abs(middle_roots - om_R_approx))
                preds["R"] = middle_roots[idx_R]

                remaining = np.delete(middle_roots, idx_R)
                if len(remaining) > 0:
                    # Find MS as nearest to +/- om_MS_scale
                    best_ms = None
                    best_dist = np.inf
                    for mr in remaining:
                        dist = min(abs(mr - om_MS_scale), abs(mr + om_MS_scale))
                        if dist < best_dist:
                            best_dist = dist
                            best_ms = mr
                    preds["MS"] = best_ms

    # --------------------------------------------------------
    # E. Kelvin / Magneto-Kelvin boundary approximation
    # --------------------------------------------------------
    h_r1 = H_eq(p.r1, p)
    oma2_r1 = omega_A_sq(p.r1, p)
    RHS_1 = (m**2 * p.g * h_r1) / (p.r1**2)
    if RHS_1 >= oma2_r1:
        preds["MK_in"] = -np.sqrt(RHS_1 - oma2_r1)

    h_r2 = H_eq(p.r2, p)
    oma2_r2 = omega_A_sq(p.r2, p)
    RHS_2 = (m**2 * p.g * h_r2) / (p.r2**2)
    if RHS_2 >= oma2_r2:
        preds["MK_out"] = np.sqrt(RHS_2 - oma2_r2)

    return preds


# ==========================================================
# 11. Branch Assignment & Continuation
# ==========================================================

def calculate_boundary_localization(eta, r_grid, p: Params, dr_wall_ratio=0.1):
    dr_wall = (p.r2 - p.r1) * dr_wall_ratio

    eta_sq = np.abs(eta)**2

    # Simple trapezoidal integration approximation on Chebyshev grid
    # For a robust integral, we just sum weighted by local grid spacing
    dr = np.abs(np.diff(r_grid))
    dr_mid = np.zeros(len(r_grid))
    dr_mid[0] = dr[0] / 2.0
    dr_mid[-1] = dr[-1] / 2.0
    dr_mid[1:-1] = (dr[:-1] + dr[1:]) / 2.0

    integral_total = np.sum(eta_sq * dr_mid)

    mask_in = r_grid <= (p.r1 + dr_wall)
    integral_in = np.sum(eta_sq[mask_in] * dr_mid[mask_in])

    mask_out = r_grid >= (p.r2 - dr_wall)
    integral_out = np.sum(eta_sq[mask_out] * dr_mid[mask_out])

    L_in = integral_in / max(integral_total, 1e-15)
    L_out = integral_out / max(integral_total, 1e-15)

    return L_in, L_out

def assign_branches_global(modes, m, r_grid, p: Params, N_max=10):
    wkb_preds = {}

    # 1. Collect all WKB predictions
    k_preds = get_wkb_predictions(m, 1, p)
    if not np.isnan(k_preds["MK_in"]):
        wkb_preds[('MK_in', 0)] = k_preds["MK_in"]
    if not np.isnan(k_preds["MK_out"]):
        wkb_preds[('MK_out', 0)] = k_preds["MK_out"]

    for n in range(1, N_max + 1):
        preds = get_wkb_predictions(m, n, p)
        for b_name in ["MP_p", "MP_m", "R", "MS"]:
            if not np.isnan(preds[b_name]):
                wkb_preds[(b_name, n)] = preds[b_name]

    # 2. Match global eigenvalues to WKB predictions greedily
    distances = []
    for i, mode in enumerate(modes):
        eig = mode["omega"]
        for branch_id, w_val in wkb_preds.items():
            dist = abs(eig - w_val)
            distances.append((dist, i, branch_id))

    distances.sort(key=lambda x: x[0])

    assigned_eigs = set()
    assigned_branches = set()

    matches = {
        'MK_in': None,
        'MK_out': None,
        'MP_p': {},
        'MP_m': {},
        'R': {},
        'MS': {}
    }

    for dist, eig_idx, branch_id in distances:
        if eig_idx not in assigned_eigs and branch_id not in assigned_branches:
            assigned_eigs.add(eig_idx)
            assigned_branches.add(branch_id)

            mode = modes[eig_idx]
            b_name, n = branch_id

            # Additional constraint for Kelvin branches: check localization
            if b_name in ['MK_in', 'MK_out']:
                L_in, L_out = calculate_boundary_localization(mode["eta"], r_grid, p)

                # We require a basic threshold for localization to accept Kelvin assignment, e.g., > 20%
                if b_name == 'MK_in' and L_in < 0.15:
                    assigned_eigs.remove(eig_idx)
                    assigned_branches.remove(branch_id)
                    continue
                elif b_name == 'MK_out' and L_out < 0.15:
                    assigned_eigs.remove(eig_idx)
                    assigned_branches.remove(branch_id)
                    continue

                matches[b_name] = mode
            else:
                matches[b_name][n] = mode

    return matches


# ==========================================================
# 12. Convergence Testing
# ==========================================================

def run_convergence_test(m_val, p: Params, N_list=[32, 48, 64, 80]):
    print(f"\n--- Convergence Test for m = {m_val} ---")
    results = {}

    scan_range = calculate_scan_range(p, m_val)

    for N in N_list:
        r_grid, D1, D2 = chebyshev_lobatto(N, p.r1, p.r2)
        modes = find_eigenmodes(m_val, r_grid, D1, D2, p, n_scan=2000, omega_range=scan_range)
        freqs = np.array([m["omega"] for m in modes])
        results[N] = freqs
        print(f"N = {N:2d}: Found {len(freqs)} modes.")

    if len(N_list) < 2:
        return

    ref_freqs = results[N_list[-1]]
    if len(ref_freqs) == 0:
        print("No modes found at reference resolution.")
        return

    print("\nTracking convergence of selected reference modes:")
    print("Mode Index | " + " | ".join([f"N={N:2d}" for N in N_list]))
    print("-" * (13 + 9 * len(N_list)))

    for i, ref_w in enumerate(ref_freqs[:5]):
        row = f"  Mode {i:2d}  |"
        for N in N_list:
            freqs = results[N]
            if len(freqs) == 0:
                row += "   ---   |"
                continue
            idx_closest = np.argmin(np.abs(freqs - ref_w))
            closest_w = freqs[idx_closest]
            norm_w = closest_w / (2.0 * p.Omega)
            row += f" {norm_w:7.4f} |"
        print(row)
    print("-" * (13 + 9 * len(N_list)))

# ==========================================================
# 13. Spectrum Loop & Output Formats
# ==========================================================

COLORS = {
    'MP':  '#1f77b4',
    'MK':  '#2ca02c',
    'R':   '#d62728',
    'MS':  '#ff7f0e',
}

def compute_spectrum(p: Params, M_max=10, N_col=40):
    m_arr = np.arange(1, M_max + 1)
    r_grid, D1, D2 = chebyshev_lobatto(N_col, p.r1, p.r2)

    global_data = {
        'MK_in': [], 'MK_out': [],
        'MP_p': {}, 'MP_m': {}, 'R': {}, 'MS': {}
    }

    om_range = calculate_scan_range(p, M_max)
    print(f"\nComputing spectrum for N={N_col}, m=1..{M_max}")
    print(f"Scan Range: [{om_range[0]:.2e}, {om_range[1]:.2e}] rad/s")
    print(f"Norm Range: [{om_range[0]/(2*p.Omega):.2f}, {om_range[1]/(2*p.Omega):.2f}]")

    all_assigned_modes = []

    for m in m_arr:
        modes = find_eigenmodes(m, r_grid, D1, D2, p, omega_range=om_range, n_scan=4000)

        assigned = assign_branches_global(modes, m, r_grid, p, N_max=10)

        # Log mode details for final report
        for b_name, item in assigned.items():
            if item is None:
                continue
            if b_name in ['MK_in', 'MK_out']:
                diag = evaluate_mode_diagnostics(item["omega"], m, item["eta"], r_grid, D1, D2, p)
                all_assigned_modes.append({
                    "m": m, "branch": b_name, "omega": item["omega"], "diag": diag, "sigma_min": item["sigma_min"]
                })
                global_data[b_name].append((m, item["omega"]))
            else:
                for n_idx, m_dict in item.items():
                    diag = evaluate_mode_diagnostics(m_dict["omega"], m, m_dict["eta"], r_grid, D1, D2, p)
                    all_assigned_modes.append({
                        "m": m, "branch": f"{b_name}_{n_idx}", "omega": m_dict["omega"], "diag": diag, "sigma_min": m_dict["sigma_min"]
                    })
                    global_data[b_name].setdefault(n_idx, []).append((m, m_dict["omega"]))

        pos = sorted([e["omega"] for e in modes if e["omega"] > 0])
        neg = sorted([e["omega"] for e in modes if e["omega"] < 0], reverse=True)
        if m <= 3 or m % 5 == 0:
            print(f"  m={m:2d}: +{[f'{x/(2*p.Omega):.3f}' for x in pos[:3]]}... "
                  f"-{[f'{abs(x)/(2*p.Omega):.3f}' for x in neg[:3]]}...")

    return global_data, all_assigned_modes, om_range

def plot_spectrum(global_data, m_arr, p: Params, case_num, om_range):
    os.makedirs("outputs", exist_ok=True)
    m_fine = np.linspace(1, max(m_arr), 400)

    WKB_MK_in = []
    WKB_MK_out = []
    WKB_MP_p = {n: [] for n in range(1, 10)}
    WKB_MP_m = {n: [] for n in range(1, 10)}
    WKB_R = {n: [] for n in range(1, 10)}
    WKB_MS = {n: [] for n in range(1, 10)}

    for m in m_fine:
        k_preds = get_wkb_predictions(m, 1, p)
        WKB_MK_in.append(k_preds["MK_in"])
        WKB_MK_out.append(k_preds["MK_out"])

        for n in range(1, 10):
            preds = get_wkb_predictions(m, n, p)
            WKB_MP_p[n].append(preds["MP_p"])
            WKB_MP_m[n].append(preds["MP_m"])
            WKB_R[n].append(preds["R"])
            WKB_MS[n].append(preds["MS"])

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[5, 1.4], hspace=0.08, wspace=0.10)

    ax = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])

    def draw_panel(ax_obj, zoom=False):
        if not zoom:
            ax_obj.plot(m_fine, np.array(WKB_MK_in)/(2*p.Omega), color=COLORS['MK'], lw=2.5, label='Magneto-Kelvin (WKB)', zorder=4)
            ax_obj.plot(m_fine, np.array(WKB_MK_out)/(2*p.Omega), color=COLORS['MK'], lw=2.5, zorder=4)

        for n in range(1, 5):
            lbl_MP = 'Magneto-Poincaré (WKB)' if n == 1 else None
            lbl_R  = 'Rossby (WKB)' if n == 1 else None
            lbl_MS = 'Magnetostrophic (WKB)' if n == 1 else None
            alpha_val = max(0.1, 0.4 - 0.05*(n-1))

            ax_obj.plot(m_fine, np.array(WKB_MP_p[n])/(2*p.Omega), color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MP, zorder=3)
            ax_obj.plot(m_fine, np.array(WKB_MP_m[n])/(2*p.Omega), color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, zorder=3)
            ax_obj.plot(m_fine, np.array(WKB_R[n])/(2*p.Omega), color=COLORS['R'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_R, zorder=3)
            ax_obj.plot(m_fine, np.array(WKB_MS[n])/(2*p.Omega), color=COLORS['MS'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MS, zorder=3)

        if global_data['MK_in']:
            mk_m, mk_w = zip(*global_data['MK_in'])
            ax_obj.scatter(mk_m, np.array(mk_w)/(2*p.Omega), color=COLORS['MK'], marker='s', s=55, zorder=6, label='Magneto-Kelvin (collocation)')
        if global_data['MK_out']:
            mk_m, mk_w = zip(*global_data['MK_out'])
            ax_obj.scatter(mk_m, np.array(mk_w)/(2*p.Omega), color=COLORS['MK'], marker='s', s=55, zorder=6)

        def plot_branch_collocation(data_dict, color, label_prefix):
            for n in sorted(data_dict.keys()):
                if not data_dict[n]: continue
                bp_m, bp_w = zip(*data_dict[n])
                lbl = f'{label_prefix} (collocation)' if n==1 else None
                ax_obj.scatter(bp_m, np.array(bp_w)/(2*p.Omega), color=color, marker='o', s=30, zorder=5, alpha=0.85, label=lbl)

        plot_branch_collocation(global_data['MP_p'], COLORS['MP'], 'Magneto-Poincaré')
        plot_branch_collocation(global_data['MP_m'], COLORS['MP'], 'Magneto-Poincaré')
        plot_branch_collocation(global_data['R'], COLORS['R'], 'Rossby')
        plot_branch_collocation(global_data['MS'], COLORS['MS'], 'Magnetostrophic')

        ax_obj.axhline(0, color='grey', lw=0.8, ls=':')
        ax_obj.axhline(+1, color='grey', lw=0.9, ls='--', alpha=0.5)
        ax_obj.axhline(-1, color='grey', lw=0.9, ls='--', alpha=0.5)

        ax_obj.set_xlabel('Azimuthal wavenumber $m$', fontsize=20)
        ax_obj.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=20)
        ax_obj.set_xlim(0.7, max(m_arr) + 0.5)
        ax_obj.set_xticks(m_arr[::2] if len(m_arr) > 10 else m_arr)
        ax_obj.grid(True, alpha=0.22)

    draw_panel(ax, zoom=False)
    ax.set_title('Full spectrum', fontsize=25)

    # Configure dynamic scale for full spectrum based on dynamic range
    y_full = om_range[1] / (2.0 * p.Omega)
    ax.set_ylim(-y_full, y_full)

    draw_panel(ax2, zoom=True)
    ax2.set_title(r'Slow branches', fontsize=25)

    # Extract slow branch limits
    slow_vals = []
    for k in global_data['R'].keys():
        slow_vals.extend(global_data['R'][k])
    for k in global_data['MS'].keys():
        slow_vals.extend(global_data['MS'][k])

    if slow_vals:
        _, w_vals = zip(*slow_vals)
        max_slow = max(np.abs(np.array(w_vals) / (2.0 * p.Omega)))
        zoom_lim = max(max_slow * 1.5, 0.5)
    else:
        zoom_lim = 1.0

    # Strictly preserve the symmetric negative and positive frequencies for slow modes
    ax2.set_ylim(-zoom_lim, zoom_lim)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    seen = set()
    legend_handles = []
    legend_labels = []
    for h, l in zip(all_handles, all_labels):
        if l not in seen and l is not None:
            seen.add(l)
            legend_handles.append(h)
            legend_labels.append(l)

    fig.legend(
        legend_handles, legend_labels,
        loc='center',
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        fontsize=16,
        frameon=True,
        labelspacing=0.8,
        columnspacing=1.5,
        handlelength=2.2
    )

    base_name = f"outputs/case{case_num}_B0_{p.B0}_C_{p.C}_dispersion"
    plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_name}.pdf", dpi=150, bbox_inches='tight')
    print(f"Saved figure: {base_name}.png/.pdf")
    plt.close(fig)


# ==========================================================
# 14. Terminal Report & Main Execution
# ==========================================================

def print_final_report(case_num, p: Params, N_col, M_max, om_range, all_assigned_modes):
    print(f"\n{'='*60}")
    print(f" FINAL REPORT - Case {case_num}")
    print(f"{'='*60}")
    print(f" Physical Parameters:")
    print(f"   B0        = {p.B0} T")
    print(f"   C         = {p.C} m^3/s")
    print(f"   H0        = {p.H0} m")
    print(f"   Omega     = {p.Omega} rad/s")
    print(f"   f         = {p.f} rad/s")
    print(f" Domain:")
    print(f"   r1        = {p.r1} m")
    print(f"   r2        = {p.r2} m")

    r_test = np.linspace(p.r1, p.r2, 100)
    h_test = H_eq(r_test, p)
    print(f"   H_eq min  = {np.min(h_test):.2f} m")
    print(f"   H_eq max  = {np.max(h_test):.2f} m")

    print(f" WKB Evaluated at r_WKB = {p.r_WKB:.2f} m:")
    print(f"   H_eq(r_WKB)= {H_eq(p.r_WKB, p):.2f} m")
    print(f"   Q(r_WKB)   = {Q(p.r_WKB, p):.4e} m/s^2")
    print(f"   omega_A    = {np.sqrt(omega_A_sq(p.r_WKB, p)):.4e} rad/s")

    print(f"\n Spectral Search:")
    print(f"   Scan Range = [{om_range[0]:.2e}, {om_range[1]:.2e}] rad/s")
    print(f"   Total Modes Assigned: {len(all_assigned_modes)}")

    print("\n Selected Mode Metadata (m=1):")
    print(f"  {'Branch':<10} | {'omega':<15} | {'omega_hat':<10} | {'sigma_min':<10} | {'rel_sigma':<10} | {'Rb_1':<10} | {'L2_res':<10}")
    print("-" * 90)

    # Filter for m=1 to show as a sample
    sample_modes = [m for m in all_assigned_modes if m["m"] == 1]

    for sm in sample_modes:
        w_hat = sm["omega"] / p.f
        print(f"  {sm['branch']:<10} | {sm['omega']:<+15.6e} | {w_hat:<+10.4f} | {sm['sigma_min']:<10.2e} | "
              f"{sm['diag']['rel_L2']:<10.2e} | {sm['diag']['Rb_1']:<10.2e} | {sm['diag']['L2_res']:<10.2e}")

    print(f"{'='*60}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Variable-depth SWMHD Eigenvalue Solver")
    parser.add_argument("--run-all", action="store_true", help="Run all 5 cases sequentially")
    parser.add_argument("--case", type=int, default=4, help="Run specific case 1-5")
    parser.add_argument("--M-max", type=int, default=10, help="Max azimuthal wavenumber")
    parser.add_argument("--N-col", type=int, default=40, help="Collocation points")
    parser.add_argument("--B0", type=float, default=5e-4, help="Override B0 for case 3/4")
    parser.add_argument("--r-WKB", type=float, default=None, help="Evaluate local WKB approximations at specific radius")
    args = parser.parse_args()

    cases_to_run = [1, 2, 3, 4, 5] if args.run_all else [args.case]

    for case_num in cases_to_run:
        print(f"\n\n{'*'*70}\n Starting Execution for Case {case_num}\n{'*'*70}")

        set_case_parameters(case_num, p, r_WKB_override=args.r_WKB)
        if case_num in [3, 4] and args.B0 != 5e-4:
            p.B0 = args.B0

        print(f"Initializing Numerical Grid N={args.N_col}...")
        run_numerical_validations(p, N_col=args.N_col)

        run_convergence_test(1, p, N_list=[32, 48, 64])

        global_data, all_assigned_modes, om_range = compute_spectrum(p, M_max=args.M_max, N_col=args.N_col)

        plot_spectrum(global_data, np.arange(1, args.M_max+1), p, case_num, om_range)

        print_final_report(case_num, p, args.N_col, args.M_max, om_range, all_assigned_modes)

if __name__ == "__main__":
    main()
