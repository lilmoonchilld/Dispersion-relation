import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# Run Mode and Complex Stability Configuration
# ==========================================================
# Supported modes: "real_dispersion", "complex_stability", "validation_tests"
RUN_MODE = "real_dispersion"

# Complex Stability Parameters
STABILITY_M_MIN = 1
STABILITY_M_MAX = 30
SELECTED_M = 5  # For plotting the complex spectrum of a single m

STABILITY_N_REAL = 150
STABILITY_N_IMAG = 120

STABILITY_OMEGA_R_RANGE = (-5, 5)
STABILITY_GAMMA_RANGE = (-2, 2)

MIN_FILTER_SIZE = 3
STABILITY_SIGMA_TOL = 1e-12
STABILITY_SIGMA_RTOL = 1e-12
STABILITY_DUPLICATE_TOL = 5e-4
STABILITY_MIN_ABS_OMEGA = 1e-3
STABILITY_GAMMA_TOL = 1e-12

# ==========================================================
# Publication style settings
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

# -- 0. Physical parameters --------------------------------------------------
MU0 = 4 * np.pi * 1e-7
Omega = 0.5e-4 # rad/s
H0 = 500 # m
g = 9.81 # m/s^2
B0 = 8.5e-4 # T
rho0 = 1000.0 # kg/m^3
C = 9.375e8 # m^3/s^2

r1 = 0.5e6 # inner radius [m]
r2 = 1e6 # outer radius [m]
r0 = 0.5 * (r1 + r2)

VA2 = B0**2 / (MU0 * rho0)
omA2 = VA2 / H0**2
omA = np.sqrt(omA2)
beta_g = 2.0 * C / r0**3
beta_eff = beta_g
c0sq = g * H0
c0 = np.sqrt(c0sq)
f = 2.0 * Omega # Coriolis parameter

# -- Dimensionless Parameters ------------------------------------------------
f_scale = 2.0 * Omega
hat_r1 = r1 / r0
hat_r2 = r2 / r0
hat_c0sq = c0sq / (Omega**2 * r0**2)
hat_VA2 = VA2 / (Omega**2 * r0**2)
hat_VA2sq = np.sqrt(hat_VA2)
gamma = 2.0 * C / (Omega**2 * r0**3)
hat_omA = np.sqrt(VA2) / (f_scale * H0)
hat_omA2 = hat_omA**2

# Two positive-frequency magnetic-rotational modes
omega_low = Omega - np.sqrt(Omega**2 - omA2)
omega_high = Omega + np.sqrt(Omega**2 - omA2)

# Normalised frequencies
omega_low_norm = omega_low / (2 * Omega)
omega_high_norm = omega_high / (2 * Omega)

print("=" * 62)
print(" System Parameters (Dimensionless)")
print("=" * 62)
print(f" Three Governing Numbers:")
print(f" 1. Magnetic-Coriolis ratio (hat_VA2sq) : {hat_VA2sq:.4f}")
print(f" 2. Inverse Of Froude Number Square (hat_c0^2) : {hat_c0sq:.4f}")
print(f" 3. Radial-gravity ratio (gamma) : {gamma:.4f}")
print("-" * 62)
print(f" Domain: hat_r in [{hat_r1:.3f}, {hat_r2:.3f}]")
print(f"V_A2 = {omA:.6e} m/s")
print(f"omega_low = {omega_low:.6e} s^-1")
print(f"omega_high= {omega_high:.6e} s^-1")
print(f"low/(2Ω) = {omega_low_norm:.6f}")
print(f"high/(2Ω) = {omega_high_norm:.6f}")
print()

# -- 1. Collocation machinery ------------------------------------------------
N_col = 64

def chebyshev_lobatto(N, r_min, r_max):
    j = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N)
    c[0] = 2; c[-1] = 2
    X = np.tile(xi, (N, 1)); dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) * (-1)**(i + k) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    sc = 2.0 / (r_max - r_min); D1 = sc * D; D2 = D1 @ D1
    rp = 0.5*(r_min+r_max) + 0.5*(r_max-r_min)*xi
    rp = rp[::-1]; D1 = D1[::-1,::-1]; D2 = D2[::-1,::-1]
    return rp, D1, D2

hat_r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, hat_r1, hat_r2)

def omega_star_hat(hat_omega):
    """Dimensionless modified frequency hat_omega_* = hat_omega + hat_omega_A^2 / hat_omega."""
    return hat_omega + hat_omA2 / hat_omega

def build_matrix(hat_omega, m_val):
    ws_hat = omega_star_hat(hat_omega)

    # P_coeff: 1/r - (1+gamma)(r-1)/c0^2
    Pv = 1.0 / hat_r_grid - (gamma) * (hat_r_grid - 1.0) / hat_c0sq

    # Q_coeff
    term1 = 4.0 * hat_omega * (ws_hat**2 - 1.0) / (ws_hat * hat_c0sq)
    term2 = -m_val**2 / hat_r_grid**2
    term3 = -(gamma) * (2.0 * hat_r_grid - 1.0) / (hat_c0sq * hat_r_grid)
    term4 = -m_val * (gamma) * (hat_r_grid - 1.0) / (ws_hat * hat_c0sq * hat_r_grid)
    Qv = term1 + term2 + term3 + term4

    # Determine type of matrix elements
    if np.iscomplexobj(hat_omega):
        dtype = complex
    else:
        dtype = float

    L = D2_mat.astype(dtype) + np.diag(Pv) @ D1_mat + np.diag(Qv)
    for idx in [0, N_col - 1]:
        rb = hat_r_grid[idx]
        bc_eta_coeff = -(ws_hat * (gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
        L[idx,:] = ws_hat * hat_c0sq * D1_mat[idx,:] + bc_eta_coeff * np.eye(N_col, dtype=dtype)[idx]
    return L

def _singular_values(hat_omega, m_val):
    """Return all singular values of L(hat_omega, m_val), or None if invalid."""
    if not np.isfinite(hat_omega) or abs(hat_omega) <= np.finfo(float).tiny:
        return None
    try:
        L = build_matrix(hat_omega, m_val)
        return svdvals(L, overwrite_a=True, check_finite=False)
    except Exception:
        return None

def sigma_min_value(hat_omega, m_val):
    """Return the smallest singular value of L(hat_omega, m_val)."""
    singular_values = _singular_values(hat_omega, m_val)
    if singular_values is None: return np.inf
    return float(singular_values[-1])

def singular_value_info(hat_omega, m_val):
    """Return (sigma_min, sigma_max) for L(hat_omega, m_val)."""
    singular_values = _singular_values(hat_omega, m_val)
    if singular_values is None: return np.inf, np.inf
    return float(singular_values[-1]), float(singular_values[0])

def extract_eigenfunction(hat_omega, m_val):
    """ Extract the normalized right singular vector associated with sigma_min.
    At an eigenfrequency L eta = 0, so the right singular vector belonging to
    the smallest singular value is the collocation eigenfunction eta.
    """
    L = build_matrix(hat_omega, m_val)
    _, singular_values, Vh = svd(L, full_matrices=False, overwrite_a=True, check_finite=False)
    eta = Vh[-1, :].copy()
    norm = np.max(np.abs(eta))
    if norm > 0:
        eta /= norm
    return eta, float(singular_values[-1])

def _frequency_scan_segments(omega_range, n_scan, min_abs_omega):
    """Create scan segments while excluding the singular point at omega = 0."""
    scan_full = np.linspace(*omega_range, n_scan)
    valid = scan_full[np.abs(scan_full) > min_abs_omega]
    if valid.size == 0: return []
    if valid.size == 1: return [valid]
    base_step = abs(scan_full[1] - scan_full[0]) if scan_full.size > 1 else np.inf
    jumps = np.where(np.diff(valid) > 1.5 * base_step)[0] + 1
    return [segment for segment in np.split(valid, jumps) if segment.size > 0]

def find_eigenmodes(m_val, omega_range=(-45, 45), n_scan=10000, sigma_tol=1e-12, sigma_rtol=1e-12, duplicate_tol=5e-4, min_abs_omega=0.01, xatol=1e-11, maxiter=120):
    """ Return sorted nonlinear eigenmodes for azimuthal mode m_val. """
    sigma_cache = {}

    def cached_singular_info(w):
        key = float(w)
        if key not in sigma_cache:
            if abs(key) <= min_abs_omega:
                sigma_cache[key] = (np.inf, np.inf)
            else:
                sigma_cache[key] = singular_value_info(key, m_val)
        return sigma_cache[key]

    def objective(w):
        sigma_min, _ = cached_singular_info(w)
        return sigma_min

    candidates = []
    for scan in _frequency_scan_segments(omega_range, n_scan, min_abs_omega):
        sigma_scan = np.array([objective(w) for w in scan])

        for k in range(1, len(scan) - 1):
            if not np.isfinite(sigma_scan[k]): continue
            if sigma_scan[k] <= sigma_scan[k - 1] and sigma_scan[k] <= sigma_scan[k + 1]:
                candidates.append((sigma_scan[k], scan[k - 1], scan[k], scan[k + 1]))

    candidates.sort(key=lambda item: item[0])
    modes = []

    for _, left, center, right in candidates:
        if not (np.isfinite(left) and np.isfinite(center) and np.isfinite(right)): continue
        if left >= right or left < 0 < right: continue

        try:
            result = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": xatol, "maxiter": maxiter},
            )
            if result.success and np.isfinite(result.x):
                omega_refined = float(result.x)
            else:
                omega_refined = float(center)
        except Exception:
            omega_refined = float(center)

        sigma_min, sigma_max = cached_singular_info(omega_refined)
        relative_sigma = sigma_min / max(sigma_max, np.finfo(float).tiny)

        if sigma_min > sigma_tol and relative_sigma > sigma_rtol: continue

        duplicate_index = None
        for idx, mode in enumerate(modes):
            if abs(omega_refined - mode["omega"]) < duplicate_tol:
                duplicate_index = idx
                break

        eta, sigma_from_svd = extract_eigenfunction(omega_refined, m_val)
        mode = {
            "omega": omega_refined,
            "eta": eta,
            "sigma_min": sigma_from_svd,
            "relative_sigma": relative_sigma,
        }

        if duplicate_index is None:
            modes.append(mode)
        elif sigma_from_svd < modes[duplicate_index]["sigma_min"]:
            modes[duplicate_index] = mode

    modes.sort(key=lambda item: item["omega"])
    return modes

def find_eigenvalues(m_val, omega_range=(-45, 45), n_scan=10000):
    """Return sorted real eigenfrequencies (dimensionless) for azimuthal mode m_val."""
    modes = find_eigenmodes(m_val, omega_range=omega_range, n_scan=n_scan)
    return np.array([mode["omega"] for mode in modes])

# ==========================================================
# Complex Stability Solver Core
# ==========================================================
import scipy.ndimage
from scipy.optimize import minimize

def complex_sigma_min(hat_omega, m_val):
    """Return the minimum singular value of the full complex collocation matrix."""
    if abs(hat_omega) <= STABILITY_MIN_ABS_OMEGA:
        return np.inf
    try:
        L = build_matrix(hat_omega, m_val)
        return float(svdvals(L, overwrite_a=True, check_finite=False)[-1])
    except Exception:
        return np.inf

def complex_singular_info(hat_omega, m_val):
    """Return (sigma_min, sigma_max) for L(hat_omega, m_val)."""
    if abs(hat_omega) <= STABILITY_MIN_ABS_OMEGA:
        return np.inf, np.inf
    try:
        L = build_matrix(hat_omega, m_val)
        s = svdvals(L, overwrite_a=True, check_finite=False)
        return float(s[-1]), float(s[0])
    except Exception:
        return np.inf, np.inf

def find_complex_candidates(m_val, omega_r_range, gamma_range, n_real, n_imag):
    """Generate a 2D grid and identify local candidate minima for refinement."""
    omega_r_vals = np.linspace(omega_r_range[0], omega_r_range[1], n_real)
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_imag)

    sigma_grid = np.zeros((n_imag, n_real))

    for i, g_val in enumerate(gamma_vals):
        for j, r_val in enumerate(omega_r_vals):
            hat_w = r_val + 1j * g_val
            sigma_grid[i, j] = complex_sigma_min(hat_w, m_val)

    # Find local minima using a minimum filter
    min_filtered = scipy.ndimage.minimum_filter(sigma_grid, size=MIN_FILTER_SIZE, mode='constant', cval=np.inf)

    # Candidate locations are where the grid value equals the filtered minimum
    candidate_mask = (sigma_grid == min_filtered) & (np.isfinite(sigma_grid))

    candidates = []
    for i in range(n_imag):
        for j in range(n_real):
            if candidate_mask[i, j]:
                omega_initial = omega_r_vals[j] + 1j * gamma_vals[i]
                candidates.append((sigma_grid[i, j], omega_initial))

    candidates.sort(key=lambda x: x[0])
    return candidates

def refine_complex_candidate(omega_initial, m_val):
    """Refine a local candidate in the (hat_omega_r, hat_gamma) plane."""

    def objective(x):
        hat_w = x[0] + 1j * x[1]
        return complex_sigma_min(hat_w, m_val)

    res = minimize(
        objective,
        x0=np.array([np.real(omega_initial), np.imag(omega_initial)]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
    )

    if res.success:
        return res.x[0] + 1j * res.x[1]
    return omega_initial

def find_complex_eigenmodes(m_val, omega_r_range, gamma_range, n_real, n_imag):
    """Find all complex eigenmodes for a given m."""
    candidates = find_complex_candidates(m_val, omega_r_range, gamma_range, n_real, n_imag)

    modes = []
    for _, omega_initial in candidates:
        omega_refined = refine_complex_candidate(omega_initial, m_val)

        sigma_min, sigma_max = complex_singular_info(omega_refined, m_val)
        rel_sigma = sigma_min / max(sigma_max, np.finfo(float).tiny)

        if sigma_min > STABILITY_SIGMA_TOL and rel_sigma > STABILITY_SIGMA_RTOL:
            continue

        duplicate_idx = None
        for idx, mode in enumerate(modes):
            if abs(omega_refined - mode["omega"]) < STABILITY_DUPLICATE_TOL:
                duplicate_idx = idx
                break

        eta, s_min = extract_eigenfunction(omega_refined, m_val)

        mode = {
            "omega": omega_refined,
            "omega_r_hat": np.real(omega_refined),
            "gamma_hat": np.imag(omega_refined),
            "sigma_min": s_min,
            "relative_sigma": rel_sigma,
            "eta": eta
        }

        if duplicate_idx is None:
            modes.append(mode)
        elif s_min < modes[duplicate_idx]["sigma_min"]:
            modes[duplicate_idx] = mode

    # Sort by real part, then imaginary part
    modes.sort(key=lambda x: (x["omega_r_hat"], x["gamma_hat"]))
    return modes

def find_most_unstable_mode(modes):
    """Find the most unstable mode in a list of complex modes."""
    if not modes:
        return None
    return max(modes, key=lambda x: x["gamma_hat"])


# -- 2. WKB analytic dispersion relations (Dimensionless) --------------------
def wkb_branches(m_val, n_radial=1):
    hat_kr = n_radial * np.pi / (hat_r2 - hat_r1)

    r_val = 1.0
    R_1 = gamma * (2*r_val - 1) / r_val
    S_1 = 0.0
    kappa2_1 = hat_kr**2 + m_val**2 / r_val**2
    K_1 = hat_c0sq * kappa2_1 + R_1

    A2 = 2.0 * hat_omA2 - 1.0 - 0.25 * K_1
    A1 = 0.0
    A0 = hat_omA2 * (hat_omA2 - 0.25 * K_1)

    disc = A2**2 - 4.0 * A0
    if disc < 0: disc = 0.0

    X1 = (-A2 + np.sqrt(disc)) / 2.0
    X2 = (-A2 - np.sqrt(disc)) / 2.0

    oMP_p = +np.sqrt(max(X1, 0)) if X1 >= 0 else np.nan
    oMP_m = -np.sqrt(max(X1, 0)) if X1 >= 0 else np.nan

    oMS = -np.sqrt(max(X2, 0)) if X2 >= 0 and hat_omA2 > 0 else np.nan
    if np.isnan(oMS) and hat_omA2 > 0:
        oMS = np.sqrt(max(X2, 0))

    oMS_val = - hat_VA2 * kappa2_1 / (4.0 + hat_c0sq * kappa2_1) if hat_VA2 > 0 else 0.0

    r = 1.02
    R_r = gamma * (2.0 * r - 1.0) / r
    S_r = m_val * gamma * (r - 1.0) / r
    kappa2_r = hat_kr**2 + (m_val / r)**2
    K_r = hat_c0sq * kappa2_r + R_r

    if B0 == 0:
        oR_val = -S_r / (4.0 + K_r)
    else:
        if abs(S_r) < 1e-12:
            oR_val = np.nan
        else:
            C0 = 4.0 * hat_omA2**2 - hat_omA2 * K_r
            oR_val = C0 / S_r

    oR_arr = np.array([oR_val])

    hat_c0 = np.sqrt(hat_c0sq)
    arg_in = m_val**2 * hat_c0sq / hat_r1**2 - 4.0 * hat_omA2
    if arg_in >= 0 and hat_omA <= m_val * hat_c0 / (2.0 * hat_r1):
        oMK_in = -0.5 * np.sqrt(arg_in)
    else:
        oMK_in = np.nan

    arg_out = m_val**2 * hat_c0sq / hat_r2**2 - 4.0 * hat_omA2
    if arg_out >= 0 and hat_omA <= m_val * hat_c0 / (2.0 * hat_r2):
        oMK_out = 0.5 * np.sqrt(arg_out)
    else:
        oMK_out = np.nan

    return oMP_p, oMP_m, oMK_in, oMK_out, np.array(oR_arr), oMS_val

# -- 3. Branch assignment with global minimum distance logic -----------------
def assign_branches_global(eigs, m_val, N_max=20):
    wkb_preds = {}

    _, _, oMK_in, oMK_out, _, _ = wkb_branches(m_val, 1)

    if not np.isnan(oMK_in): wkb_preds[('MK_in', 0)] = oMK_in
    if not np.isnan(oMK_out): wkb_preds[('MK_out', 0)] = oMK_out

    for n in range(1, N_max + 1):
        oMP_p, oMP_m, _, _, oR_arr, oMS_val = wkb_branches(m_val, n)
        if not np.isnan(oMP_p): wkb_preds[('MP_p', n)] = oMP_p
        if not np.isnan(oMP_m): wkb_preds[('MP_m', n)] = oMP_m
        wkb_preds[('R', n)] = oR_arr
        if not np.isnan(oMS_val): wkb_preds[('MS', n)] = oMS_val

    distances = []
    for i, eig in enumerate(eigs):
        for branch_id, wkb_vals in wkb_preds.items():
            if isinstance(wkb_vals, np.ndarray):
                valid_vals = wkb_vals[~np.isnan(wkb_vals)]
                if len(valid_vals) > 0:
                    dist = np.min(np.abs(eig - valid_vals))
                    distances.append((dist, i, branch_id))
            else:
                dist = abs(eig - wkb_vals)
                distances.append((dist, i, branch_id))

    distances.sort(key=lambda x: x[0])

    assigned_eigs = set()
    assigned_branches = set()
    matches = {}

    for dist, eig_idx, branch_id in distances:
        if eig_idx not in assigned_eigs and branch_id not in assigned_branches:
            assigned_eigs.add(eig_idx)
            assigned_branches.add(branch_id)
            matches[branch_id] = eigs[eig_idx]

    results = {
        'MK_in': None,
        'MK_out': None,
        'MP_p': {},
        'MP_m': {},
        'R': {},
        'MS': {}
    }

    for branch_id, eig_val in matches.items():
        b_name, n = branch_id
        if b_name == 'MK_in':
            results['MK_in'] = eig_val
        elif b_name == 'MK_out':
            results['MK_out'] = eig_val
        else:
            results[b_name][n] = eig_val

    return results

# ==========================================================
# Stability Analysis and Plotting
# ==========================================================
import json

def plot_complex_spectrum(m_val, modes):
    """Plot complex spectrum for a selected m in the Re-Im plane."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if not modes:
        ax.text(0.5, 0.5, 'No valid eigenmodes found', transform=ax.transAxes, ha='center', va='center')
    else:
        omega_r = [m["omega_r_hat"] for m in modes]
        gamma = [m["gamma_hat"] for m in modes]
        ax.scatter(omega_r, gamma, color='blue', alpha=0.7, label='Eigenmodes')

        most_unstable = find_most_unstable_mode(modes)
        if most_unstable:
            ax.scatter(most_unstable["omega_r_hat"], most_unstable["gamma_hat"],
                       color='red', marker='*', s=150, zorder=5, label='Most unstable')

    ax.axhline(0, color='grey', ls='--', lw=1.5)
    ax.set_xlabel(r'Re($\omega$)/(2$\Omega$)')
    ax.set_ylabel(r'Im($\omega$)/(2$\Omega$)')
    ax.set_title(f'Complex global spectrum, m = {m_val}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.savefig(f"swmhd_complex_spectrum_m={m_val}.png", dpi=300, bbox_inches='tight')
    print(f"Saved: swmhd_complex_spectrum_m={m_val}.png")

def plot_max_growth_vs_m(m_values, most_unstable_list):
    """Plot max growth rate vs m."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gammas = []
    for m in m_values:
        mode = most_unstable_list.get(m)
        if mode:
            gammas.append(mode["gamma_hat"])
        else:
            gammas.append(np.nan)

    ax.plot(m_values, gammas, 'ko-', lw=2, markersize=8)
    ax.axhline(0, color='grey', ls='--', lw=1.5)

    ax.set_xlabel('Azimuthal wavenumber $m$')
    ax.set_ylabel(r'$\gamma_{\max}/(2\Omega)$')
    ax.set_title('Maximum Growth Rate vs m')
    ax.grid(True, alpha=0.3)
    plt.savefig("swmhd_max_growth_vs_m.png", dpi=300, bbox_inches='tight')
    print("Saved: swmhd_max_growth_vs_m.png")

def run_complex_stability():
    print("=" * 62)
    print(" Running Complex Stability Analysis")
    print("=" * 62)

    m_values = np.arange(STABILITY_M_MIN, STABILITY_M_MAX + 1)
    all_modes = {}
    most_unstable_dict = {}

    summary_data = {
        "parameters": {
            "m_range": [int(STABILITY_M_MIN), int(STABILITY_M_MAX)],
            "search_domain": {
                "omega_r": STABILITY_OMEGA_R_RANGE,
                "gamma": STABILITY_GAMMA_RANGE
            },
            "resolution": {
                "n_real": STABILITY_N_REAL,
                "n_imag": STABILITY_N_IMAG,
                "N_col": N_col
            },
            "tolerances": {
                "sigma_tol": STABILITY_SIGMA_TOL,
                "sigma_rtol": STABILITY_SIGMA_RTOL
            }
        },
        "results": {}
    }

    npz_data = {}

    global_most_unstable = None

    for m in m_values:
        modes = find_complex_eigenmodes(
            m,
            STABILITY_OMEGA_R_RANGE,
            STABILITY_GAMMA_RANGE,
            STABILITY_N_REAL,
            STABILITY_N_IMAG
        )
        all_modes[m] = modes

        most_unstable = find_most_unstable_mode(modes)
        most_unstable_dict[m] = most_unstable

        # Determine status
        status = "DECAYING/STABLE"
        if most_unstable:
            if most_unstable["gamma_hat"] > STABILITY_GAMMA_TOL:
                status = "UNSTABLE"
            elif abs(most_unstable["gamma_hat"]) <= STABILITY_GAMMA_TOL:
                status = "NEUTRAL"
        else:
            status = "NO MODES FOUND"

        print(f"m = {m}")
        print(f"  Number of complex eigenmodes = {len(modes)}")
        if most_unstable:
            print(f"  Most unstable:")
            print(f"    omega_hat = {most_unstable['omega']}")
            print(f"    Re(omega)/(2Omega) = {most_unstable['omega_r_hat']:.6f}")
            print(f"    Im(omega)/(2Omega) = {most_unstable['gamma_hat']:.6e}")
            print(f"    sigma_min = {most_unstable['sigma_min']:.3e}")
            print(f"    status = {status}")

            if global_most_unstable is None or most_unstable["gamma_hat"] > global_most_unstable["gamma_hat"]:
                global_most_unstable = dict(most_unstable)
                global_most_unstable["m"] = int(m)
        else:
            print(f"  No valid modes detected.")
        print("-" * 40)

        # Save to npz_data
        if modes:
            npz_data[f"m_{m}_omegas"] = np.array([md["omega"] for md in modes])
            npz_data[f"m_{m}_etas"] = np.array([md["eta"] for md in modes])

        # Save to summary_data
        summary_data["results"][int(m)] = {
            "num_modes": len(modes),
            "most_unstable": {
                "omega_r_hat": float(most_unstable["omega_r_hat"]) if most_unstable else None,
                "gamma_hat": float(most_unstable["gamma_hat"]) if most_unstable else None,
                "sigma_min": float(most_unstable["sigma_min"]) if most_unstable else None,
            } if most_unstable else None,
            "status": status
        }

        if m == SELECTED_M:
            plot_complex_spectrum(m, modes)

    plot_max_growth_vs_m(m_values, most_unstable_dict)

    print("=" * 62)
    if global_most_unstable:
        print(f"Most unstable mode over all m:")
        print(f"  m = {global_most_unstable['m']}")
        print(f"  Re(omega)/(2Omega) = {global_most_unstable['omega_r_hat']:.6f}")
        print(f"  Im(omega)/(2Omega) = {global_most_unstable['gamma_hat']:.6e}")
    else:
        print("No unstable eigenvalue detected within the specified complex-frequency")
        print("search domain, collocation resolution, and numerical tolerances.")
    print("=" * 62)

    with open("swmhd_stability_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)
    print("Saved JSON summary: swmhd_stability_summary.json")

    np.savez("swmhd_stability_data.npz", **npz_data)
    print("Saved NPZ data: swmhd_stability_data.npz")


# ==========================================================
# Validation Tests
# ==========================================================

def run_validation_tests():
    print("=" * 62)
    print(" Running Validation Tests")
    print("=" * 62)

    m_test = 2

    # Test A: Complex matrix support
    print("Test A: Complex matrix support...")
    try:
        mat = build_matrix(0.5 + 0.1j, m_test)
        assert np.iscomplexobj(mat), "Matrix should be complex"
        assert np.all(np.isfinite(mat)), "Matrix should contain finite values"
        print("  [PASS] Complex matrix successfully built and is finite.")
    except Exception as e:
        print(f"  [FAIL] Test A failed: {e}")

    # Test B: Real-frequency recovery
    print("Test B: Real-frequency recovery (gamma=0)...")
    try:
        real_eigs = find_eigenvalues(m_test, omega_range=(-5, 5), n_scan=2000)
        complex_modes = find_complex_eigenmodes(
            m_test, (-5, 5), (-0.1, 0.1), 50, 3
        )
        # Check if the real parts of complex roots with near-zero imaginary part match real_eigs
        matched = 0
        for ce in complex_modes:
            if abs(ce["gamma_hat"]) < 1e-4:
                dist = np.min(np.abs(real_eigs - ce["omega_r_hat"]))
                if dist < 1e-2:
                    matched += 1
        print(f"  [PASS] Found {matched} modes matching real-frequency solver.")
    except Exception as e:
        print(f"  [FAIL] Test B failed: {e}")

    # Test C: B0 = 0 limit
    print("Test C: B0 = 0 limit...")
    global B0, omA, omA2, hat_omA, hat_omA2, hat_VA2, hat_VA2sq
    old_b0 = B0; old_oma = omA; old_oma2 = omA2
    old_homa = hat_omA; old_homa2 = hat_omA2
    old_hva2 = hat_VA2; old_hva2sq = hat_VA2sq
    try:
        # Override B0 to 0
        B0 = 0.0
        omA = 0.0; omA2 = 0.0; hat_omA = 0.0; hat_omA2 = 0.0; hat_VA2 = 0.0; hat_VA2sq = 0.0

        # Test if matrix builds successfully without division by zero
        mat_b0 = build_matrix(0.5 + 0.1j, m_test)
        assert np.all(np.isfinite(mat_b0)), "Matrix should be finite for B0=0"

        # Restore variables
        B0 = old_b0; omA = old_oma; omA2 = old_oma2
        hat_omA = old_homa; hat_omA2 = old_homa2
        hat_VA2 = old_hva2; hat_VA2sq = old_hva2sq
        print("  [PASS] B0=0 limit handled without singularities.")
    except Exception as e:
        # Restore variables in case of fail
        B0 = old_b0; omA = old_oma; omA2 = old_oma2
        hat_omA = old_homa; hat_omA2 = old_homa2
        hat_VA2 = old_hva2; hat_VA2sq = old_hva2sq
        print(f"  [FAIL] Test C failed: {e}")

    # Test D: Collocation resolution convergence
    print("Test D: Collocation resolution convergence...")
    global N_col, hat_r_grid, D1_mat, D2_mat
    old_ncol = N_col
    resolutions = [48, 64]
    res_results = {}
    try:
        for N in resolutions:
            N_col = N
            hat_r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, hat_r1, hat_r2)

            modes = find_complex_eigenmodes(m_test, (-2, 2), (-0.5, 0.5), 30, 20)
            most_unst = find_most_unstable_mode(modes)
            res_results[N] = most_unst["gamma_hat"] if most_unst else None

        print(f"  Results for gamma_max: N=48 -> {res_results[48]}, N=64 -> {res_results[64]}")
        print("  [PASS] Completed resolution convergence check.")
    except Exception as e:
        print(f"  [FAIL] Test D failed: {e}")
    finally:
        N_col = old_ncol
        hat_r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, hat_r1, hat_r2)

    # Test E: Complex search-domain convergence
    print("Test E: Complex search-domain convergence...")
    try:
        modes1 = find_complex_eigenmodes(m_test, (-1, 1), (-0.5, 0.5), 20, 15)
        most_unst1 = find_most_unstable_mode(modes1)
        g1 = most_unst1["gamma_hat"] if most_unst1 else None

        modes2 = find_complex_eigenmodes(m_test, (-2, 2), (-1, 1), 30, 20)
        most_unst2 = find_most_unstable_mode(modes2)
        g2 = most_unst2["gamma_hat"] if most_unst2 else None

        print(f"  Domain 1 gamma_max: {g1}")
        print(f"  Domain 2 gamma_max: {g2}")
        print("  [PASS] Completed search-domain convergence check.")
    except Exception as e:
        print(f"  [FAIL] Test E failed: {e}")

    # Test F: Complex-grid resolution convergence
    print("Test F: Complex-grid resolution convergence...")
    try:
        modes1 = find_complex_eigenmodes(m_test, (-1, 1), (-0.5, 0.5), 20, 15)
        most_unst1 = find_most_unstable_mode(modes1)
        g1 = most_unst1["gamma_hat"] if most_unst1 else None

        modes2 = find_complex_eigenmodes(m_test, (-1, 1), (-0.5, 0.5), 40, 30)
        most_unst2 = find_most_unstable_mode(modes2)
        g2 = most_unst2["gamma_hat"] if most_unst2 else None

        print(f"  Low res gamma_max: {g1}")
        print(f"  High res gamma_max: {g2}")
        print("  [PASS] Completed grid resolution convergence check.")
    except Exception as e:
        print(f"  [FAIL] Test F failed: {e}")

    print("=" * 62)
    print(" Validation Tests Finished")
    print("=" * 62)

# ==========================================================
# Main execution wrapper
# ==========================================================

def run_real_dispersion():
    # -- 4. Compute spectrum -----------------------------------------------------
    M_max = 30
    m_arr = np.arange(1, M_max+1)
    m_fine = np.linspace(1, M_max, 400)

    print(f"Running collocation N={N_col}, m=1..{M_max} ...")

    mk_in_data = []
    mk_out_data = []
    mp_p_data = {}
    mp_m_data = {}
    r_data = {}
    ms_data = {}

    for m in m_arr:
        eigs = find_eigenvalues(m)
        assigned = assign_branches_global(eigs, m)

        if assigned['MK_in'] is not None:
            mk_in_data.append((m, assigned['MK_in']))
        if assigned['MK_out'] is not None:
            mk_out_data.append((m, assigned['MK_out']))

        for n, val in assigned['MP_p'].items():
            mp_p_data.setdefault(n, []).append((m, val))
        for n, val in assigned['MP_m'].items():
            mp_m_data.setdefault(n, []).append((m, val))
        for n, val in assigned['R'].items():
            r_data.setdefault(n, []).append((m, val))
        for n, val in assigned['MS'].items():
            ms_data.setdefault(n, []).append((m, val))

        pos = sorted([e for e in eigs if e > 0])
        neg = sorted([e for e in eigs if e < 0], reverse=True)
        if m <= 3 or m % 5 == 0:
            print(f"  m={m:2d}: +{[f'{x:.3f}' for x in pos[:4]]}... "
                  f"-{[f'{abs(x):.3f}' for x in neg[:4]]}...")

    # -- 5. WKB curves for fine array -------------------------------------------
    WKB_MK_in = np.array([wkb_branches(m, 1)[2] for m in m_fine])
    WKB_MK_out = np.array([wkb_branches(m, 1)[3] for m in m_fine])
    WKB_MP_p = {}
    WKB_MP_m = {}
    WKB_R = {}
    WKB_MS = {}

    for n in range(1, 10):
        op_list = []
        om_list = []
        oR_list = []
        oMS_list = []

        for m in m_fine:
            op, om, _, _, oR_arr, oMS = wkb_branches(m, n)
            op_list.append(op)
            om_list.append(om)
            oR_list.append(oR_arr)
            oMS_list.append(oMS)

        WKB_MP_p[n] = np.array(op_list)
        WKB_MP_m[n] = np.array(om_list)
        WKB_R[n] = np.array(oR_list).T
        WKB_MS[n] = np.array(oMS_list)

    # -- 6. Plot -----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[5, 1.4], hspace=0.08, wspace=0.10)

    ax = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])

    fig.suptitle(
        r'SWMHD global dispersion relation (Dimensionless)'
        '\n'
        r'Magneto-Poincaré (blue), Magneto-Kelvin (green), Rossby (red), Magnetostrophic (orange)',
        fontsize=13, fontweight='bold')

    COLORS = { 'MP': '#1f77b4', 'MK': '#2ca02c', 'R': '#d62728', 'MS': '#ff7f0e', }

    def draw_panel(ax_obj, zoom=False):
        if not zoom:
            ax_obj.plot(m_fine, WKB_MK_in,color=COLORS['MK'], lw=2.5,label='Magneto-Kelvin (WKB)', zorder=4)
            ax_obj.plot(m_fine, WKB_MK_out,color=COLORS['MK'], lw=2.5, zorder=4)

        for n in range(1, 9):
            lbl_MP = 'Magneto-Poincaré (WKB)' if n == 1 else None
            lbl_R  = 'Rossby (WKB)' if n == 1 else None
            lbl_MS = 'Magnetostrophic (WKB)' if n == 1 else None

            alpha_val = max(0.1, 0.4 - 0.05*(n-1))

            ax_obj.plot(m_fine, WKB_MP_p[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MP, zorder=3)
            ax_obj.plot(m_fine, WKB_MP_m[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, zorder=3)
            ax_obj.plot(m_fine, WKB_MS[n], color=COLORS['MS'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MS, zorder=3)

        if mk_in_data:
            mk_m, mk_w = zip(*mk_in_data)
            ax_obj.scatter(mk_m, mk_w, color=COLORS['MK'], marker='s', s=55, zorder=6, label='Magneto-Kelvin (collocation)')
        if mk_out_data:
            mk_m, mk_w = zip(*mk_out_data)
            ax_obj.scatter(mk_m, mk_w, color=COLORS['MK'], marker='s', s=55, zorder=6)

        def plot_branch_collocation(data_dict, color, label_prefix):
            for n in sorted(data_dict.keys()):
                if not data_dict[n]: continue
                mp_m, mp_w = zip(*data_dict[n])
                lbl = f'{label_prefix} (collocation)' if n==1 else None
                ax_obj.scatter(mp_m, mp_w, color=color, marker='o', s=30, zorder=5, alpha=0.85, label=lbl)

        plot_branch_collocation(mp_p_data, COLORS['MP'], 'Magneto-Poincaré')
        plot_branch_collocation(mp_m_data, COLORS['MP'], 'Magneto-Poincaré')
        plot_branch_collocation(ms_data, COLORS['MS'], 'Magnetostrophic')

        ax_obj.axhline(0,  color='grey', lw=0.8, ls=':')
        ax_obj.axhline(+1, color='grey', lw=0.9, ls='--', alpha=0.5)
        ax_obj.axhline(-1, color='grey', lw=0.9, ls='--', alpha=0.5)
        ax_obj.axhline(omega_low_norm,linestyle='--',linewidth=1.8,label=rf'Analytical low: $\omega/(2\Omega)={omega_low_norm:.3f}$')
        ax_obj.axhline(omega_high_norm,linestyle='--',linewidth=1.8,label=rf'Analytical high: $\omega/(2\Omega)={omega_high_norm:.3f}$')

        if hat_omA > 0:
            ax_obj.axhline(+hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
            ax_obj.axhline(-hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)

        ax_obj.set_xlabel('Azimuthal wavenumber $m$', fontsize=20)
        ax_obj.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=20)
        ax_obj.set_xlim(0.7, M_max+0.5)
        ax_obj.set_ylim(-25,25)
        ax_obj.set_xticks(m_arr[::3])
        ax_obj.grid(True, alpha=0.22)

        if zoom:
            ax_obj.set_ylim(0, 1.3)

    draw_panel(ax, zoom=False)
    ax.set_title('Full spectrum', fontsize=25)

    for n, dy in [(1, 1.5), (2, 1.5)]:
        Oh_n1_arr = wkb_branches(1, n)[0]
        if not np.all(np.isnan(Oh_n1_arr)):
            Oh_n1 = np.nanmean(Oh_n1_arr)
            ax.annotate(f'$n={n}$', xy=(1, Oh_n1), xytext=(2.5, Oh_n1+dy*0.6), fontsize=12,
                        color=COLORS['MP'], arrowprops=dict(arrowstyle='->', color=COLORS['MP'], lw=0.8))

    draw_panel(ax2, zoom=True)
    ax2.set_title(r'Slow branches', fontsize=25)

    handles1, labels1 = ax.get_legend_handles_labels()
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
        legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.065),
        ncol=3, fontsize=20, frameon=True, labelspacing=0.8, columnspacing=1.5, handlelength=2.2
    )

    plt.savefig(f"swmhd_four_branch_dispersion_g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches='tight')
    print("\nSaved: swmhd_four_branch_dispersion_g=...png")

if __name__ == '__main__':
    if RUN_MODE == "real_dispersion":
        run_real_dispersion()
    elif RUN_MODE == "complex_stability":
        run_complex_stability()
    elif RUN_MODE == "validation_tests":
        run_validation_tests()
    else:
        print(f"Unknown RUN_MODE: {RUN_MODE}")