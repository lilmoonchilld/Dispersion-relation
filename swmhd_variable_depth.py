import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 0. Publication style settings
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
# 1. Physical parameters & Execution Cases
# ==========================================================

class Params:
    MU0 = 4 * np.pi * 1e-7
    Omega = 0.5e-4
    H0 = 500.0
    g = 9.81
    B0 = 0.0
    rho0 = 1000.0
    C = 9.375e9
    r1 = 0.5e6
    r2 = 1.0e6

    # Derived
    r0 = 0.75e6
    f = 1e-4

    # Flags
    constant_depth_override = False

def set_case_parameters(case_num, p: Params):
    # Reset to defaults before applying case logic
    p.B0 = 0.0
    p.C = 9.375e9
    p.constant_depth_override = False

    if case_num == 1:
        p.B0 = 0.0
        p.C = 0.0
    elif case_num == 2:
        p.B0 = 0.0
        p.C = 9.375e9
    elif case_num == 3:
        p.B0 = 1.0  # Or another reasonable value, e.g. 1 T
        p.C = 0.0
    elif case_num == 4:
        p.B0 = 1.0
        p.C = 9.375e9
    elif case_num == 5:
        p.B0 = 0.0
        p.C = 0.0
        p.constant_depth_override = True
    else:
        raise ValueError(f"Unknown case number {case_num}")

    p.r0 = (p.r1 + p.r2) / 2.0
    p.f = 2.0 * p.Omega

p = Params()

# ==========================================================
# 2. Equilibrium depth
# ==========================================================

def H_eq(r, p: Params):
    if p.constant_depth_override:
        # True constant depth
        # Since H_eq(r0) = H0 must hold, and we want constant depth everywhere:
        return np.full_like(r, p.H0, dtype=float) if isinstance(r, np.ndarray) else p.H0
    return p.H0 + (p.Omega**2 / (2.0 * p.g)) * (r**2 - p.r0**2) + (p.C / p.g) * (1.0/r - 1.0/p.r0)

def dH_eq_dr(r, p: Params):
    if p.constant_depth_override:
        return np.zeros_like(r, dtype=float) if isinstance(r, np.ndarray) else 0.0
    return (p.Omega**2 * r) / p.g - p.C / (p.g * r**2)

def Q(r, p: Params):
    # Q(r) = g * H_eq'(r)
    return p.g * dH_eq_dr(r, p)

# ==========================================================
# 3. Magnetic-frequency functions
# ==========================================================

def omega_A_sq(r, p: Params):
    return p.B0**2 / (p.MU0 * p.rho0 * H_eq(r, p)**2)

def omega_star(r, omega, p: Params):
    if omega == 0:
        return np.inf if isinstance(r, (float, int)) else np.full_like(r, np.inf)
    return omega + omega_A_sq(r, p) / omega

def domega_star_dr(r, omega, p: Params):
    if omega == 0:
        return np.inf if isinstance(r, (float, int)) else np.full_like(r, np.inf)
    if p.B0 == 0.0:
        return np.zeros_like(r, dtype=float) if isinstance(r, np.ndarray) else 0.0
    return - (2.0 * omega_A_sq(r, p) / omega) * (dH_eq_dr(r, p) / H_eq(r, p))

# Simple test to verify compilation
if __name__ == "__main__":
    set_case_parameters(4, p)
    print("Case 4 loaded successfully.")
    r_test = np.linspace(p.r1, p.r2, 10)
    print("H_eq:", H_eq(r_test, p))
    print("dH_eq_dr:", dH_eq_dr(r_test, p))
    print("Q:", Q(r_test, p))
    print("omega_star:", omega_star(r_test, 1e-4, p))
    print("domega_star_dr:", domega_star_dr(r_test, 1e-4, p))

# ==========================================================
# 4. Chebyshev collocation
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

    # Permute to make r ascending
    rp = rp[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]

    return rp, D1, D2

# ==========================================================
# 5. Numerical Validations
# ==========================================================

def run_numerical_validations(p: Params, N_col=40):
    r_grid, D1, _ = chebyshev_lobatto(N_col, p.r1, p.r2)

    # 1. Reference-depth consistency: H_eq(r0) = H0
    h_r0 = H_eq(p.r0, p)
    assert np.isclose(h_r0, p.H0, atol=1e-8, rtol=1e-8), f"Depth at r0 = {h_r0} != {p.H0}"

    # 2. Derivative consistency
    h_eq = H_eq(r_grid, p)
    dh_eq_analytic = dH_eq_dr(r_grid, p)
    dh_eq_fd = D1 @ h_eq

    # Exclude boundaries for finite-difference accuracy checks
    err_dh = np.max(np.abs(dh_eq_fd[1:-1] - dh_eq_analytic[1:-1]))
    assert err_dh < 1e-4, f"H_eq' analytic vs numerical error {err_dh} too large"

    # 3. Verify g * H_eq'(r) = Q(r)
    q_vals = Q(r_grid, p)
    err_q = np.max(np.abs(p.g * dh_eq_analytic - q_vals))
    assert err_q < 1e-12, "Q(r) does not match g * H_eq'(r)"

    # 4. If B0 != 0, compare analytic omega_*' to finite difference
    omega_test = p.Omega  # just a test frequency
    omega_s = omega_star(r_grid, omega_test, p)
    domega_s_analytic = domega_star_dr(r_grid, omega_test, p)
    domega_s_fd = D1 @ omega_s

    if p.B0 != 0.0:
        err_domega_s = np.max(np.abs(domega_s_fd[1:-1] - domega_s_analytic[1:-1]))
        assert err_domega_s < 1e-4, f"omega_*' analytic vs numerical error {err_domega_s} too large"

    # 5. Verify that H_eq(r) > 0 over the computational domain
    assert np.all(h_eq > 0), "H_eq(r) <= 0 found in the computational domain!"

    print("All basic numerical validations passed for current configuration.")

if __name__ == "__main__":
    run_numerical_validations(p)

# ==========================================================
# 6. Variable-depth ODE coefficients & Assembly
# ==========================================================

def get_A(r, omega, p: Params):
    # A(r) = 1/r + H_eq'/H_eq + omega_*'/omega_* - 2*omega_**omega_*' / (omega_*^2 - f^2)
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
    # B(r) = [omega*(omega_*^2 - f^2)] / [g*H_eq*omega_*]
    #      + [m*f / (r*H_eq*omega_*)] * [H_eq' - 2*H_eq*omega_**omega_*' / (omega_*^2 - f^2)]
    #      - m^2/r^2
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

    # Boundary conditions: omega_*(r_b)*eta'(r_b) + (m*f/r_b)*eta(r_b) = 0
    N = len(r_grid)
    om_s_r1 = omega_star(r_grid[0], omega, p)
    om_s_r2 = omega_star(r_grid[-1], omega, p)

    # r1 (inner boundary, index 0)
    L[0, :] = om_s_r1 * D1[0, :]
    L[0, 0] += (m * p.f) / r_grid[0]

    # r2 (outer boundary, index N-1)
    L[-1, :] = om_s_r2 * D1[-1, :]
    L[-1, -1] += (m * p.f) / r_grid[-1]

    return L

def extract_eigenfunction(L):
    # L = U Sigma V^H
    # Smallest singular value is the last one, associated right singular vector is Vh[-1, :]
    _, singular_values, Vh = svd(L, full_matrices=False, overwrite_a=True, check_finite=False)
    eta = Vh[-1, :].copy()
    norm_val = np.max(np.abs(eta))
    if norm_val > 0:
        eta /= norm_val
    return eta, float(singular_values[-1])

def reconstruct_vr(r, eta, D1, omega, m, p: Params):
    # vr = -i*g * [omega_* * eta' + (m*f/r)*eta] / (omega_*^2 - f^2)
    eta_prime = D1 @ eta
    om_s = omega_star(r, omega, p)

    numerator = om_s * eta_prime + (m * p.f / r) * eta
    denominator = om_s**2 - p.f**2

    # Note: Using complex arithmetic since vr has the 'i' term.
    # The physical norm will just be np.abs(vr)
    vr = -1j * p.g * numerator / denominator
    return vr

# ==========================================================
# 7. SVD Nonlinear Frequency Search
# ==========================================================

def get_singular_values(omega, m, r_grid, D1, D2, p: Params):
    if not np.isfinite(omega) or abs(omega) < 1e-12:
        return None
    try:
        # Check if omega_*^2 == f^2 anywhere
        om_s = omega_star(r_grid, omega, p)
        if np.any(np.abs(om_s**2 - p.f**2) < 1e-12):
            return None

        L = build_operator(omega, m, r_grid, D1, D2, p)
        return svdvals(L, overwrite_a=True, check_finite=False)
    except Exception:
        return None

def find_eigenmodes(
    m,
    r_grid,
    D1,
    D2,
    p: Params,
    omega_range=(-3.0, 3.0),
    n_scan=5000,
    sigma_tol=1e-7,
    sigma_rtol=1e-6,
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

    # Masking singular regions: omega=0 and omega_*^2 = f^2
    scan_points = np.linspace(omega_range[0], omega_range[1], n_scan)
    valid_scan = []

    for w in scan_points:
        if abs(w) < 1e-8:
            continue
        om_s = omega_star(r_grid, w, p)
        if np.any(np.abs(om_s**2 - p.f**2) < 1e-8):
            continue
        valid_scan.append(w)

    valid_scan = np.array(valid_scan)
    if len(valid_scan) < 3:
        return []

    # Split into segments based on jumps
    base_step = abs(scan_points[1] - scan_points[0])
    jumps = np.where(np.diff(valid_scan) > 1.5 * base_step)[0] + 1
    segments = np.split(valid_scan, jumps)

    candidates = []
    for seg in segments:
        if len(seg) < 3:
            continue
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
                options={"xatol": 1e-10, "maxiter": 100}
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
# 8. Boundary & ODE Residual Diagnostics
# ==========================================================

def evaluate_mode_diagnostics(omega, m, eta, r_grid, D1, D2, p: Params):
    # Boundary condition residual: R_b = omega_*(r_b)*eta'(r_b) + (m*f/r_b)*eta(r_b)
    eta_prime = D1 @ eta

    # Inner boundary
    om_s_r1 = omega_star(r_grid[0], omega, p)
    Rb_1 = om_s_r1 * eta_prime[0] + (m * p.f / r_grid[0]) * eta[0]

    # Outer boundary
    om_s_r2 = omega_star(r_grid[-1], omega, p)
    Rb_2 = om_s_r2 * eta_prime[-1] + (m * p.f / r_grid[-1]) * eta[-1]

    # Reconstruct vr
    vr = reconstruct_vr(r_grid, eta, D1, omega, m, p)
    vr_r1 = np.abs(vr[0])
    vr_r2 = np.abs(vr[-1])

    # Interior ODE residual
    A_r = get_A(r_grid, omega, p)
    B_r = get_B(r_grid, omega, m, p)

    eta_double_prime = D2 @ eta
    ODE_residual = eta_double_prime + A_r * eta_prime + B_r * eta

    # L2 norm on interior
    interior_res = ODE_residual[1:-1]
    interior_eta = eta[1:-1]
    L2_res = np.linalg.norm(interior_res)
    rel_L2 = L2_res / max(np.linalg.norm(interior_eta), 1e-15)

    return {
        "Rb_1": np.abs(Rb_1),
        "Rb_2": np.abs(Rb_2),
        "vr_1": vr_r1,
        "vr_2": vr_r2,
        "L2_res": L2_res,
        "rel_L2": rel_L2
    }

# ==========================================================
# 9. Local WKB Dispersion Relations & Branch Approximations
# ==========================================================

def get_wkb_predictions(m, n, p: Params, r_WKB=None):
    if r_WKB is None:
        r_WKB = p.r0

    kr = n * np.pi / (p.r2 - p.r1)
    kappa2 = kr**2 + (m / r_WKB)**2

    h_wkb = H_eq(r_WKB, p)
    q_wkb = Q(r_WKB, p)
    oma2_wkb = omega_A_sq(r_WKB, p)

    preds = {
        "MP_p": np.nan, "MP_m": np.nan,
        "MK_in": np.nan, "MK_out": np.nan,
        "R": np.nan, "MS": np.nan
    }

    # --------------------------------------------------------
    # A. Hydrodynamic uniform-depth limit (B0=0, Q=0)
    # --------------------------------------------------------
    if p.B0 == 0.0 and p.constant_depth_override:
        # Poincare waves
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
        cm2 = (m * p.f * q_wkb) / r_WKB

        roots = np.roots([c1, c0, cm1, cm2])
        real_roots = np.sort(roots[np.isreal(roots)].real)

        if len(real_roots) == 3:
            preds["MP_m"] = real_roots[0]
            preds["R"] = real_roots[1]
            preds["MP_p"] = real_roots[2]

    # --------------------------------------------------------
    # C. Magnetic uniform-depth limit (Q=0, B0!=0)
    # --------------------------------------------------------
    elif p.B0 != 0.0 and p.constant_depth_override:
        # (omega^2 + omega_A^2)^2 - f^2 omega^2 - gH kappa^2 (omega^2 + omega_A^2) = 0
        # Let x = omega^2
        # (x + oma2)^2 - f^2 x - gH kappa^2 (x + oma2) = 0
        # x^2 + 2 oma2 x + oma2^2 - f^2 x - gH kappa^2 x - gH kappa^2 oma2 = 0
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

            # The slow magnetic branch
            if x2 > 0:
                preds["MS"] = -np.sqrt(x2) # Standard convention negative for slow Rossby-like modes usually? MS might be pos/neg. We just store the negative one. Let's keep it consistent.

    # --------------------------------------------------------
    # D. Variable-depth Magnetic (General)
    # --------------------------------------------------------
    elif p.B0 != 0.0 and not p.constant_depth_override:
        # Full WKB polynomial from:
        # kappa^2 = [omega*(omega_*^2 - f^2)] / [g*H*omega_*] + [mfQ] / [r g H omega_*] + [4mf oma2 Q omega] / [r g H N(omega)]
        # This is an 8th order polynomial in omega.
        # However, instructions state: "Use the balance omega_* ~ pm f and low frequency scaling omega_MS ~ omega_A^2/f only as an asymptotic estimate. Clearly label it as an approximation."
        # The prompt also says "Implement exact numerical roots of this local cubic for hydro".
        # For the full WKB, we can evaluate the polynomial roots exactly using np.roots

        # Let's construct the polynomial for the full WKB equation:
        # Let O = omega.
        # om_s = O + oma2/O = (O^2 + oma2)/O
        # om_s^2 - f^2 = (O^2 + oma2)^2/O^2 - f^2 = [O^4 + (2oma2 - f^2)O^2 + oma2^2] / O^2 = N(O) / O^2

        # LHS: kappa^2
        # RHS1: O * [N(O)/O^2] / [ g H (O^2 + oma2)/O ] = N(O) / [ g H (O^2 + oma2) ]
        # RHS2: [m f Q] / [ r g H (O^2 + oma2)/O ] = [m f Q O] / [ r g H (O^2 + oma2) ]
        # RHS3: [4 m f oma2 Q O] / [ r g H N(O) ]

        # kappa^2 = RHS1 + RHS2 + RHS3
        # kappa^2 * g H * r = r * N(O)/(O^2 + oma2) + [m f Q O]/(O^2 + oma2) + [4 m f oma2 Q O]/N(O)

        # Multiply both sides by (O^2 + oma2) * N(O):
        # K = kappa^2 * g * h_wkb * r_WKB
        # K * (O^2 + oma2) * N(O) = r_WKB * N(O)^2 + m f q_wkb O * N(O) + 4 m f oma2 q_wkb O * (O^2 + oma2)

        # N(O) = O^4 + (2oma2 - f^2)O^2 + oma2^2

        c_N = [1.0, 0.0, 2.0*oma2_wkb - p.f**2, 0.0, oma2_wkb**2]

        # N(O)^2
        c_N2 = np.polymul(c_N, c_N)

        # K * (O^2 + oma2) * N(O)
        c_O2_oma2 = [1.0, 0.0, oma2_wkb]
        c_LHS = np.polymul(c_O2_oma2, c_N)
        K_val = kappa2 * p.g * h_wkb * r_WKB
        c_LHS = [K_val * x for x in c_LHS]

        # RHS terms
        c_R1 = [r_WKB * x for x in c_N2]

        c_O = [1.0, 0.0]
        c_R2_part = np.polymul(c_O, c_N)
        c_R2 = [m * p.f * q_wkb * x for x in c_R2_part]

        c_R3_part = np.polymul(c_O, c_O2_oma2)
        c_R3 = [4.0 * m * p.f * oma2_wkb * q_wkb * x for x in c_R3_part]

        # Balance: c_LHS - c_R1 - c_R2 - c_R3 = 0
        # Polynomial addition requires same length, pad with zeros
        def pad_add(p1, p2):
            length = max(len(p1), len(p2))
            return np.pad(p1, (length - len(p1), 0)) + np.pad(p2, (length - len(p2), 0))

        poly = pad_add(c_LHS, [-x for x in c_R1])
        poly = pad_add(poly, [-x for x in c_R2])
        poly = pad_add(poly, [-x for x in c_R3])

        roots = np.roots(poly)
        real_roots = np.sort(roots[np.abs(np.imag(roots)) < 1e-8].real)

        if len(real_roots) > 0:
            # Filter roots near 0 or near omega_* = f
            valid_roots = []
            for r in real_roots:
                if abs(r) > 1e-12:
                    os = r + oma2_wkb/r
                    if abs(os**2 - p.f**2) > 1e-7:
                        valid_roots.append(r)

            valid_roots = np.sort(valid_roots)

            # Simple heuristic assignment for the general full WKB case:
            # Largest positive and negative -> MP_p, MP_m
            # Next inner pair (if any) -> slow branches MS and R depending on limits
            if len(valid_roots) >= 2:
                preds["MP_p"] = valid_roots[-1]
                preds["MP_m"] = valid_roots[0]

            # The asymptotic approximations from prompt:
            # Rossby: omega_R ~ m f Q / (r (f^2 + g H kappa^2))
            om_R_approx = (m * p.f * q_wkb) / (r_WKB * (p.f**2 + p.g * h_wkb * kappa2))

            # Identify Rossby root by proximity to approximation
            if len(valid_roots) > 2:
                idx_R = np.argmin(np.abs(valid_roots - om_R_approx))
                preds["R"] = valid_roots[idx_R]

                # Magnetostrophic: omega_MS ~ omega_A^2 / f (approximate branch locator)
                om_MS_approx = oma2_wkb / p.f
                remaining = np.delete(valid_roots, [0, -1, idx_R])
                if len(remaining) > 0:
                    idx_MS = np.argmin(np.abs(remaining + om_MS_approx)) # often negative
                    preds["MS"] = remaining[idx_MS]

    # --------------------------------------------------------
    # E. Kelvin / Magneto-Kelvin boundary approximation
    # --------------------------------------------------------
    # omega^2 + omega_A^2(rb) = m^2 g H_eq(rb) / rb^2
    # Inner boundary r1 (negative root)
    h_r1 = H_eq(p.r1, p)
    oma2_r1 = omega_A_sq(p.r1, p)
    RHS_1 = (m**2 * p.g * h_r1) / (p.r1**2)
    if RHS_1 >= oma2_r1:
        preds["MK_in"] = -np.sqrt(RHS_1 - oma2_r1)

    # Outer boundary r2 (positive root)
    h_r2 = H_eq(p.r2, p)
    oma2_r2 = omega_A_sq(p.r2, p)
    RHS_2 = (m**2 * p.g * h_r2) / (p.r2**2)
    if RHS_2 >= oma2_r2:
        preds["MK_out"] = np.sqrt(RHS_2 - oma2_r2)

    return preds

# ==========================================================
# 10. Branch Assignment
# ==========================================================

def assign_branches_global(eigs, m, p: Params, N_max=10, r_WKB=None):
    wkb_preds = {}

    # Get Kelvin bounds (n=1 is arbitrary, they don't depend on n)
    k_preds = get_wkb_predictions(m, 1, p, r_WKB)
    if not np.isnan(k_preds["MK_in"]):
        wkb_preds[('MK_in', 0)] = k_preds["MK_in"]
    if not np.isnan(k_preds["MK_out"]):
        wkb_preds[('MK_out', 0)] = k_preds["MK_out"]

    for n in range(1, N_max + 1):
        preds = get_wkb_predictions(m, n, p, r_WKB)
        for b_name in ["MP_p", "MP_m", "R", "MS"]:
            if not np.isnan(preds[b_name]):
                wkb_preds[(b_name, n)] = preds[b_name]

    distances = []
    for i, eig in enumerate(eigs):
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

            b_name, n = branch_id
            if b_name in ['MK_in', 'MK_out']:
                matches[b_name] = eigs[eig_idx]
            else:
                matches[b_name][n] = eigs[eig_idx]

    return matches

# ==========================================================
# 11. Convergence Testing
# ==========================================================

def run_convergence_test(m_val, p: Params, N_list=[32, 48, 64, 80]):
    print(f"\n--- Convergence Test for m = {m_val} ---")
    results = {}
    for N in N_list:
        r_grid, D1, D2 = chebyshev_lobatto(N, p.r1, p.r2)
        modes = find_eigenmodes(m_val, r_grid, D1, D2, p, n_scan=5000, omega_range=(-20*p.Omega, 20*p.Omega))
        freqs = np.array([m["omega"] for m in modes])
        results[N] = freqs
        print(f"N = {N:2d}: Found {len(freqs)} modes.")

    if len(N_list) < 2:
        return

    ref_freqs = results[N_list[-1]]
    print("\nTracking convergence of selected reference modes (N_max):")
    print("Mode Index | " + " | ".join([f"N={N:2d}" for N in N_list]))
    print("-" * (13 + 9 * len(N_list)))

    # We will pick a few distinct modes from the highest resolution reference to track
    for i, ref_w in enumerate(ref_freqs[:5]): # just top 5 for brevity
        row = f"  Mode {i:2d}  |"
        for N in N_list:
            freqs = results[N]
            if len(freqs) == 0:
                row += "   ---   |"
                continue
            idx_closest = np.argmin(np.abs(freqs - ref_w))
            closest_w = freqs[idx_closest]
            # Convert to normalized frequency for display
            norm_w = closest_w / (2.0 * p.Omega)
            row += f" {norm_w:7.4f} |"
        print(row)
    print("-" * (13 + 9 * len(N_list)))

# ==========================================================
# 12. Main Spectrum Calculation
# ==========================================================

def compute_spectrum(p: Params, M_max=10, N_col=40):
    m_arr = np.arange(1, M_max + 1)

    r_grid, D1, D2 = chebyshev_lobatto(N_col, p.r1, p.r2)

    print(f"\nComputing spectrum for N={N_col}, m=1..{M_max}")

    global_data = {
        'MK_in': [],
        'MK_out': [],
        'MP_p': {},
        'MP_m': {},
        'R': {},
        'MS': {}
    }

    # Scan range slightly larger than high/low analytical bounds
    om_range = (-15 * p.Omega, 15 * p.Omega)

    for m in m_arr:
        modes = find_eigenmodes(m, r_grid, D1, D2, p, omega_range=om_range, n_scan=3000)
        eigs = []
        for mode in modes:
            diag = evaluate_mode_diagnostics(mode["omega"], m, mode["eta"], r_grid, D1, D2, p)
            print(f"    Mode omega={mode['omega']:+.6e} | R_b1={diag['Rb_1']:.2e}, R_b2={diag['Rb_2']:.2e} | "
                  f"vr_1={diag['vr_1']:.2e}, vr_2={diag['vr_2']:.2e} | "
                  f"L2_res={diag['L2_res']:.2e}, rel_L2={diag['rel_L2']:.2e}")
            eigs.append(mode["omega"])

        assigned = assign_branches_global(eigs, m, p)

        if assigned['MK_in'] is not None:
            global_data['MK_in'].append((m, assigned['MK_in']))
        if assigned['MK_out'] is not None:
            global_data['MK_out'].append((m, assigned['MK_out']))

        for n_idx, val in assigned['MP_p'].items():
            global_data['MP_p'].setdefault(n_idx, []).append((m, val))
        for n_idx, val in assigned['MP_m'].items():
            global_data['MP_m'].setdefault(n_idx, []).append((m, val))
        for n_idx, val in assigned['R'].items():
            global_data['R'].setdefault(n_idx, []).append((m, val))
        for n_idx, val in assigned['MS'].items():
            global_data['MS'].setdefault(n_idx, []).append((m, val))

        pos = sorted([e for e in eigs if e > 0])
        neg = sorted([e for e in eigs if e < 0], reverse=True)
        if m <= 3 or m % 5 == 0:
            print(f"  m={m:2d}: +{[f'{x/(2*p.Omega):.3f}' for x in pos[:3]]}... "
                  f"-{[f'{abs(x)/(2*p.Omega):.3f}' for x in neg[:3]]}...")

    return global_data


# ==========================================================
# 13. Plotting & Final Execution
# ==========================================================

COLORS = {
    'MP':  '#1f77b4',
    'MK':  '#2ca02c',
    'R':   '#d62728',
    'MS':  '#ff7f0e',
}

def plot_spectrum(global_data, m_arr, p: Params, case_num, save_prefix="swmhd_dispersion"):
    m_fine = np.linspace(1, max(m_arr), 400)

    # --------------------------------------------------------
    # Generate continuous WKB curves for m_fine
    # --------------------------------------------------------
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
        # 1. Plot WKB continuous lines
        if not zoom:
            ax_obj.plot(m_fine, np.array(WKB_MK_in)/(2*p.Omega), color=COLORS['MK'], lw=2.5, label='Magneto-Kelvin (WKB)', zorder=4)
            ax_obj.plot(m_fine, np.array(WKB_MK_out)/(2*p.Omega), color=COLORS['MK'], lw=2.5, zorder=4)

        for n in range(1, 5): # Just plotting first few radial modes to avoid clutter
            lbl_MP = 'Magneto-Poincaré (WKB)' if n == 1 else None
            lbl_R  = 'Rossby (WKB)' if n == 1 else None
            lbl_MS = 'Magnetostrophic (WKB)' if n == 1 else None
            alpha_val = max(0.1, 0.4 - 0.05*(n-1))

            ax_obj.plot(m_fine, np.array(WKB_MP_p[n])/(2*p.Omega), color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MP, zorder=3)
            ax_obj.plot(m_fine, np.array(WKB_MP_m[n])/(2*p.Omega), color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, zorder=3)
            ax_obj.plot(m_fine, np.array(WKB_R[n])/(2*p.Omega), color=COLORS['R'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_R, zorder=3)
            ax_obj.plot(m_fine, np.array(WKB_MS[n])/(2*p.Omega), color=COLORS['MS'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MS, zorder=3)

        # 2. Plot Global Collocation points
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
    ax.set_ylim(-15, 15)

    draw_panel(ax2, zoom=True)
    ax2.set_title(r'Slow branches', fontsize=25)
    ax2.set_ylim(0, 1.5)

    # Legend deduplication and bottom placement
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

    fname = f"{save_prefix}_case{case_num}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {fname}")
    plt.close(fig)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Variable-depth SWMHD Eigenvalue Solver")
    parser.add_argument("--run-all", action="store_true", help="Run all 5 cases")
    parser.add_argument("--case", type=int, default=4, help="Run specific case 1-5")
    parser.add_argument("--M-max", type=int, default=10, help="Max azimuthal wavenumber")
    parser.add_argument("--N-col", type=int, default=40, help="Collocation points")
    parser.add_argument("--B0", type=float, default=5e-4, help="Override B0 for case 3/4")
    args = parser.parse_args()

    cases_to_run = [1, 2, 3, 4, 5] if args.run_all else [args.case]

    for case_num in cases_to_run:
        print(f"\n{'='*50}\n Running Case {case_num}\n{'='*50}")
        set_case_parameters(case_num, p)
        if case_num in [3, 4]:
            p.B0 = args.B0

        print(f"B0 = {p.B0} T, C = {p.C}, Const Depth = {p.constant_depth_override}")

        run_numerical_validations(p, N_col=args.N_col)
        run_convergence_test(1, p, N_list=[32, 48, 64])

        global_data = compute_spectrum(p, M_max=args.M_max, N_col=args.N_col)
        plot_spectrum(global_data, np.arange(1, args.M_max+1), p, case_num)

if __name__ == "__main__":
    main()
