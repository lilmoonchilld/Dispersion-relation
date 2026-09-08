import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings("ignore")

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
MU0   = 4 * np.pi * 1e-7
Omega = 0.5e-4        # rad/s
H0    =500        # m
g     = 9.81          # m/s^2
B0    = 8.5e-4       # T
rho0  = 1000.0        # kg/m^3
C     = 9.375e8     # m^3/s^2

r1    = 0.5e6         # inner radius [m]
r2    = 1e6           # outer radius [m]
r0    = 0.5 * (r1 + r2)

VA2      = B0**2 / (MU0 * rho0)
omA2     = VA2 / H0**2
omA      = np.sqrt(omA2)
beta_g   = 2.0 * C / r0**3
beta_eff = beta_g
c0sq     = g * H0
c0       = np.sqrt(c0sq)
f        = 2.0 * Omega          # Coriolis parameter

# -- Dimensionless Parameters ------------------------------------------------
f_scale = 2.0 * Omega
hat_r1 = r1 / r0
hat_r2 = r2 / r0
hat_c0sq = c0sq / (Omega**2 * r0**2)
hat_VA2 = VA2 / (Omega**2 * r0**2)
hat_VA2sq=np.sqrt(hat_VA2)
gamma = 2.0 * C / (Omega**2 * r0**3)
hat_omA = np.sqrt(VA2) / (f_scale * H0)
hat_omA2 = hat_omA**2

# Two positive-frequency magnetic-rotational modes
omega_low  = Omega - np.sqrt(Omega**2 - omA**2)
omega_high = Omega + np.sqrt(Omega**2 - omA**2)

# Normalised frequencies
omega_low_norm  = omega_low / (2 * Omega)
omega_high_norm = omega_high / (2 * Omega)

print("=" * 62)
print("  System Parameters (Dimensionless)")
print("=" * 62)
print(f"  Three Governing Numbers:")
print(f"    1. Magnetic-Coriolis ratio (hat_VA2sq) : {hat_VA2sq:.4f}")
print(f"    2. Inverse Of Froude Number Square          (hat_c0^2)    : {hat_c0sq:.4f}")
print(f"    3. Radial-gravity ratio    (gamma)       : {gamma:.4f}")
print("-" * 62)
print(f"  Domain: hat_r in [{hat_r1:.3f}, {hat_r2:.3f}]")
print(f"V_A2       = {omA:.6e} m/s")
print(f"omega_low = {omega_low:.6e} s^-1")
print(f"omega_high= {omega_high:.6e} s^-1")
print(f"low/(2Ω)  = {omega_low_norm:.6f}")
print(f"high/(2Ω) = {omega_high_norm:.6f}")
print()


# -- 1. Collocation machinery ------------------------------------------------
N_col = 64

def chebyshev_lobatto(N, r_min, r_max):
    j = np.arange(N); xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N); c[0] = 2; c[-1] = 2
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
    Pv = 1.0 / hat_r_grid - ( gamma) * (hat_r_grid - 1.0) / hat_c0sq

    # Q_coeff
    term1 = 4.0 * hat_omega * (ws_hat**2 - 1.0) / (ws_hat * hat_c0sq)
    term2 = -m_val**2 / hat_r_grid**2
    term3 = -(gamma) * (2.0 * hat_r_grid - 1.0) / (hat_c0sq * hat_r_grid)
    term4 = -m_val * (gamma) * (hat_r_grid - 1.0) / (ws_hat * hat_c0sq * hat_r_grid)
    Qv = term1 + term2 + term3 + term4

    L = D2_mat + np.diag(Pv) @ D1_mat + np.diag(Qv)
    for idx in [0, N_col - 1]:
        rb = hat_r_grid[idx]
        bc_eta_coeff = -(ws_hat * (gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
        L[idx,:] = ws_hat * hat_c0sq * D1_mat[idx,:] + bc_eta_coeff * np.eye(N_col)[idx]
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
    if singular_values is None:
        return np.inf
    return float(singular_values[-1])

def singular_value_info(hat_omega, m_val):
    """Return (sigma_min, sigma_max) for L(hat_omega, m_val)."""
    singular_values = _singular_values(hat_omega, m_val)
    if singular_values is None:
        return np.inf, np.inf
    return float(singular_values[-1]), float(singular_values[0])

def extract_eigenfunction(hat_omega, m_val):
    """
    Extract the normalized right singular vector associated with sigma_min.

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
    if valid.size == 0:
        return []

    if valid.size == 1:
        return [valid]

    base_step = abs(scan_full[1] - scan_full[0]) if scan_full.size > 1 else np.inf
    jumps = np.where(np.diff(valid) > 1.5 * base_step)[0] + 1
    return [segment for segment in np.split(valid, jumps) if segment.size > 0]

def find_eigenmodes(
    m_val,
    omega_range=(-45, 45),
    n_scan=10000,
    sigma_tol=1e-12,
    sigma_rtol=1e-12,
    duplicate_tol=5e-4,
    min_abs_omega=0.01,
    xatol=1e-11,
    maxiter=120,
):
    """
    Return sorted nonlinear eigenmodes for azimuthal mode m_val.

    This replaces the determinant search with a singular-value nonlinear
    eigenvalue solve. Candidate frequencies are detected as local minima of
    sigma_min(L), then refined by bounded scalar minimization in the local
    scan bracket. A candidate is accepted when the absolute smallest singular
    value is below sigma_tol, or when the relative singular value
    sigma_min/sigma_max is below sigma_rtol.

    Each returned mode is a dictionary with keys:
        omega, eta, sigma_min, relative_sigma
    """
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
            if not np.isfinite(sigma_scan[k]):
                continue
            if sigma_scan[k] <= sigma_scan[k - 1] and sigma_scan[k] <= sigma_scan[k + 1]:
                candidates.append((sigma_scan[k], scan[k - 1], scan[k], scan[k + 1]))

    candidates.sort(key=lambda item: item[0])
    modes = []

    for _, left, center, right in candidates:
        if not (np.isfinite(left) and np.isfinite(center) and np.isfinite(right)):
            continue
        if left >= right or left < 0 < right:
            continue

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

        if sigma_min > sigma_tol and relative_sigma > sigma_rtol:
            continue

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

# -- 2. WKB analytic dispersion relations (Dimensionless) --------------------
def wkb_branches(m_val, n_radial=1):
    hat_kr = n_radial * np.pi / (hat_r2 - hat_r1)

    # 1. MP+, MP-, and MS evaluated strictly at r=1
    r_val = 1.0
    R_1 = gamma * (2*r_val - 1) / r_val
    S_1 = 0.0  # S(1) = 0 exactly
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

    # MS comes from the slow mode root at r=1. Rossby is exactly 0 here.
    oMS = -np.sqrt(max(X2, 0)) if X2 >= 0 and hat_omA2 > 0 else np.nan
    if np.isnan(oMS) and hat_omA2 > 0:
        oMS = np.sqrt(max(X2, 0)) # Might be positive

    # Recalculate accurately for sign handling: MS approx at r=1
    oMS_val = - hat_VA2 * kappa2_1 / (4.0 + hat_c0sq * kappa2_1) if hat_VA2 > 0 else 0.0

    # 2. Rossby wave evaluated at r_hat = 1.02
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
    # Kelvin waves
    # inner
    hat_c0 = np.sqrt(hat_c0sq)

    arg_in = m_val**2 * hat_c0sq / hat_r1**2 - 4.0 * hat_omA2
    if arg_in >= 0 and hat_omA <= m_val * hat_c0 / (2.0 * hat_r1):
        oMK_in = -0.5 * np.sqrt(arg_in)
    else:
        oMK_in = np.nan

    # outer
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

    if not np.isnan(oMK_in):
        wkb_preds[('MK_in', 0)] = oMK_in
    if not np.isnan(oMK_out):
        wkb_preds[('MK_out', 0)] = oMK_out

    for n in range(1, N_max + 1):
        oMP_p, oMP_m, _, _, oR_arr, oMS_val = wkb_branches(m_val, n)
        if not np.isnan(oMP_p): wkb_preds[('MP_p', n)] = oMP_p
        if not np.isnan(oMP_m): wkb_preds[('MP_m', n)] = oMP_m
        wkb_preds[('R', n)] = oR_arr
        if not np.isnan(oMS_val): wkb_preds[('MS', n)] = oMS_val

    distances = []
    for i, eig in enumerate(eigs):
        for branch_id, wkb_vals in wkb_preds.items():
            # If wkb_vals is an array (MP, MS, R), find minimum distance to the band
            if isinstance(wkb_vals, np.ndarray):
                valid_vals = wkb_vals[~np.isnan(wkb_vals)]
                if len(valid_vals) > 0:
                    dist = np.min(np.abs(eig - valid_vals))
                    distances.append((dist, i, branch_id))
            else:
                # For MK which is a scalar
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

# WKB_R[n] will be a 2D array: shape (10 points, len(m_fine)) for Rossby.
# The others are evaluated strictly at r=1 returning scalars for each m.
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

ax  = fig.add_subplot(gs[0, 0:3])
ax2 = fig.add_subplot(gs[0, 3:6])
# fig.suptitle(
#     r'SWMHD global dispersion relation (Dimensionless)'
#     '\n'
#     '\n'
#     r'Magneto-Poincaré (blue), Magneto-Kelvin (green), Rossby (red), Magnetostrophic (orange)',
#     fontsize=13, fontweight='bold')

COLORS = {
    'MP':  '#1f77b4',
    'MK':  '#2ca02c',
    'R':   '#d62728',
    'MS':  '#ff7f0e',
}

def draw_panel(ax_obj, zoom=False):
    # WKB curves
    if not zoom:
        ax_obj.plot(m_fine, WKB_MK_in,color=COLORS['MK'], lw=2.5,label='Magneto-Kelvin (WKB)', zorder=4)
        ax_obj.plot(m_fine, WKB_MK_out,color=COLORS['MK'], lw=2.5, zorder=4)

    for n in range(1, 9):
        lbl_MP = 'Magneto-Poincaré (WKB)' if n == 1 else None
        lbl_R  = 'Rossby (WKB)' if n == 1 else None
        lbl_MS = 'Magnetostrophic (WKB)' if n == 1 else None

        alpha_val = max(0.1, 0.4 - 0.05*(n-1))

        # Plot single scalar lines for MP and MS evaluated at r=1
        ax_obj.plot(m_fine, WKB_MP_p[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MP, zorder=3)
        ax_obj.plot(m_fine, WKB_MP_m[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, zorder=3)
        ax_obj.plot(m_fine, WKB_MS[n], color=COLORS['MS'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MS, zorder=3)

        # Plot the 10 lines only for the Rossby branch to form a continuum band
        #ax_obj.plot(m_fine, WKB_R[n][0],color=COLORS['R'],lw=1.5,ls='-',alpha=alpha_val,label=lbl_R,zorder=3)

    # Collocation points
    if mk_in_data:
        mk_m, mk_w = zip(*mk_in_data)
        ax_obj.scatter(mk_m, mk_w, color=COLORS['MK'], marker='s', s=55, zorder=6,
                       label='Magneto-Kelvin (collocation)')
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
    #plot_branch_collocation(r_data, COLORS['R'], 'Rossby')
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

    # Custom limit for right zoom panel
    if zoom:
        ax_obj.set_ylim(0, 1.3)

draw_panel(ax, zoom=False)

ax.set_title('Full spectrum', fontsize=25)

# Annotate n=1,2 on left panel at m=1 for MP
for n, dy in [(1, 1.5), (2, 1.5)]:
    Oh_n1_arr = wkb_branches(1, n)[0] # MP_p array
    if not np.all(np.isnan(Oh_n1_arr)):
        Oh_n1 = np.nanmean(Oh_n1_arr)
        ax.annotate(f'$n={n}$', xy=(1, Oh_n1), xytext=(2.5, Oh_n1+dy*n*0.6),
                    fontsize=12, color=COLORS['MP'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['MP'], lw=0.8))

draw_panel(ax2, zoom=True)
ax2.set_title(r'Slow branches', fontsize=25)

# ==========================================================
# Create one legend for the entire figure
# ==========================================================

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
    legend_handles, legend_labels,
    loc='center',
    bbox_to_anchor=(0.5, 0.065),
    ncol=3,
    fontsize=20,
    frameon=True,
    labelspacing=0.8,
    columnspacing=1.5,
    handlelength=2.2
)
# Parameter box
# pbox = (
#     rf"$\hat{{\omega}}_A = {hat_omA:.4f}$"","
#     rf"$\hat{{c}}_0^2 = {hat_c0sq:.4f}$"","
#     rf"$\gamma = {gamma:.4f}$"","
#     rf"$\Omega = {Omega:.1e}$ rad/s,  $H_0 = {H0:.0f}$ m"","
#     rf"$r_1 = {r1:.1e}$ m,  $r_2 = {r2:.1e}$ m"","
#     rf"$N = {N_col}$""."
# )



#plt.savefig(f"swmhd_four_branch_dispersion_g={g}_C={C}_B={B0}.pdf", dpi=180, bbox_inches='tight')
plt.savefig(f"swmhd_four_branch_dispersion._g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches='tight')
print("\nSaved: swmhd_four_branch_dispersion.pdf / .png")


# ==========================================================
# OPTIONAL: Save legend separately
# Place this after creating legend_handles and legend_labels.
# ==========================================================

fig_leg = plt.figure(figsize=(10, 2))
fig_leg.legend(legend_handles,legend_labels,loc="center",ncol=4,fontsize=13,frameon=True,)

#fig_leg.savefig(f"Legend_g={g}_C={C}_B={B0}.pdf", dpi=300, bbox_inches="tight")
#fig_leg.savefig(f"Legend_g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches="tight")
plt.close(fig_leg)

# ==========================================================
# OPTIONAL: Save each panel separately
# ==========================================================
# Left panel:
fig_left, ax_left = plt.subplots(figsize=(8,6))
draw_panel(ax_left, zoom=False)
fig_left.tight_layout()
ax_left.set_ylim(-25,25)
#fig_left.savefig(f"FullSpectrum_g={g}_C={C}_B={B0}.pdf", dpi=300, bbox_inches="tight")
#fig_left.savefig(f"FullSpectrum_g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches="tight")

# Right panel:
fig_zoom, ax_zoom = plt.subplots(figsize=(6,6))
draw_panel(ax_zoom, zoom=True)
ax_zoom.set_ylim(-1,1)
fig_zoom.tight_layout()
#fig_zoom.savefig(f"ZoomSpectrum_g={g}_C={C}_B={B0}.pdf", dpi=300, bbox_inches="tight")
#fig_zoom.savefig(f"ZoomSpectrum_g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches="tight")



# ==========================================================
# 7. Purely Imaginary Modes Solver (Pure Growth/Decay)
# ==========================================================
# This section searches strictly along Re(omega) = 0 for modes
# with omega = i * gamma.
import csv

def pure_imaginary_omega(hat_gamma):
    return 1j * hat_gamma

def pure_gamma_sigma_min(hat_gamma, m_val, min_abs_gamma=1e-4):
    if abs(hat_gamma) <= min_abs_gamma:
        return np.inf, np.inf

    hat_omega = pure_imaginary_omega(hat_gamma)
    try:
        L = build_matrix(hat_omega, m_val)
        s = svdvals(L, overwrite_a=False, check_finite=False)
        if len(s) == 0:
            return np.inf, np.inf
        return float(np.real(s[-1])), float(np.real(s[0]))
    except Exception:
        return np.inf, np.inf

def pure_imaginary_objective(hat_gamma, m_val, min_abs_gamma=1e-4):
    sigma_min, sigma_max = pure_gamma_sigma_min(hat_gamma, m_val, min_abs_gamma)
    if sigma_max == np.inf or sigma_max == 0:
        return np.inf
    return sigma_min / sigma_max

def extract_complex_eigenfunction(hat_omega, m_val):
    try:
        L = build_matrix(hat_omega, m_val)
        _, s, Vh = svd(L, full_matrices=False, overwrite_a=False, check_finite=False)
        eta = Vh[-1, :].copy()
        norm = np.max(np.abs(eta))
        if norm > 0:
            eta /= norm

        # compute residual
        residual = np.linalg.norm(L @ eta) / np.linalg.norm(eta)

        return eta, float(np.real(s[-1])), residual
    except Exception:
        return None, np.inf, np.inf

def find_pure_imaginary_modes(m_val, gamma_range=(-5.0, 5.0), n_scan=2000,
                              min_abs_gamma=1e-4, sigma_rtol=1e-8, residual_tol=1e-6,
                              gamma_zero_tol=1e-8, duplicate_tol=1e-4):
    scan_points = np.linspace(gamma_range[0], gamma_range[1], n_scan)
    valid_points = scan_points[np.abs(scan_points) > min_abs_gamma]

    sigma_scan = np.array([pure_imaginary_objective(g, m_val, min_abs_gamma) for g in valid_points])

    candidates = []
    for k in range(1, len(valid_points) - 1):
        if not np.isfinite(sigma_scan[k]):
            continue
        if sigma_scan[k] <= sigma_scan[k - 1] and sigma_scan[k] <= sigma_scan[k + 1]:
            candidates.append((valid_points[k - 1], valid_points[k], valid_points[k + 1]))

    modes = []
    for left, center, right in candidates:
        try:
            res = minimize_scalar(
                pure_imaginary_objective,
                args=(m_val, min_abs_gamma),
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 200}
            )
            if res.success and np.isfinite(res.x):
                gamma_refined = float(res.x)
            else:
                gamma_refined = float(center)
        except Exception:
            gamma_refined = float(center)

        rel_sigma = pure_imaginary_objective(gamma_refined, m_val, min_abs_gamma)
        if rel_sigma > sigma_rtol:
            continue

        eta, sigma_min, residual = extract_complex_eigenfunction(pure_imaginary_omega(gamma_refined), m_val)

        if residual > residual_tol:
            continue

        # check duplicate
        is_duplicate = False
        for mode in modes:
            if abs(gamma_refined - mode['gamma_hat']) < duplicate_tol:
                is_duplicate = True
                break

        if not is_duplicate:
            if abs(gamma_refined) < gamma_zero_tol:
                stability = 'neutral'
            elif gamma_refined > 0:
                stability = 'growing'
            else:
                stability = 'decaying'

            modes.append({
                'm': m_val,
                'gamma_hat': gamma_refined,
                'gamma_dimensional': 2.0 * Omega * gamma_refined,
                'sigma_min': sigma_min,
                'relative_sigma': rel_sigma,
                'residual': residual,
                'stability': stability,
                'eta': eta
            })

    modes.sort(key=lambda x: x['gamma_hat'])
    return modes

print("\n" + "=" * 62)
print("  Pure-imaginary stability spectrum")
print("  (Searching specifically for purely growing/decaying modes")
print("   with Re(omega) = 0)")
print("=" * 62)

all_pure_modes = []
most_unstable_per_m = {}

gamma_range_val = (-5.0, 5.0)

for m in m_arr:
    modes = find_pure_imaginary_modes(m, gamma_range=gamma_range_val)
    all_pure_modes.extend(modes)

    print(f"m = {m:2d}:")
    if len(modes) == 0:
        print("    no accepted pure-imaginary eigenmode found")
        most_unstable_per_m[m] = None
    else:
        most_unstable = max(modes, key=lambda x: x['gamma_hat'])
        most_unstable_per_m[m] = most_unstable
        print(f"    most unstable gamma_hat = {most_unstable['gamma_hat']:+.6e}")
        print(f"    gamma = {most_unstable['gamma_dimensional']:+.6e} s^-1")
        print(f"    omega = 0 + {most_unstable['gamma_hat']:+.6e} i")
        print(f"    sigma_min = {most_unstable['sigma_min']:.3e}, residual = {most_unstable['residual']:.3e}")
        print(f"    classification = {most_unstable['stability']}")

print("=" * 62)
print("  Summary Table")
print("  m       gamma_hat       gamma [s^-1]       classification")
print("-" * 62)
for m in m_arr:
    mu = most_unstable_per_m.get(m)
    if mu is None:
        print(f" {m:2d}       none found")
    else:
        print(f" {m:2d}       {mu['gamma_hat']:+.5e}     {mu['gamma_dimensional']:+.5e}      {mu['stability']}")
print("=" * 62)

# Global most unstable
global_most_unstable = None
for m, mu in most_unstable_per_m.items():
    if mu is not None:
        if global_most_unstable is None or mu['gamma_hat'] > global_most_unstable['gamma_hat']:
            global_most_unstable = mu

if global_most_unstable is not None:
    print("GLOBAL MOST UNSTABLE PURELY-IMAGINARY MODE")
    print("=" * 62)
    print(f"m = {global_most_unstable['m']}")
    print(f"gamma_hat = {global_most_unstable['gamma_hat']:+.6e}")
    print(f"gamma = {global_most_unstable['gamma_dimensional']:+.6e} s^-1")
    print(f"omega_hat = i({global_most_unstable['gamma_hat']:+.6e})")
    print(f"relative_sigma = {global_most_unstable['relative_sigma']:.3e}")
    print(f"residual = {global_most_unstable['residual']:.3e}")
    if global_most_unstable['gamma_hat'] > 0:
        print("-> The scan contains a purely growing mode.")
    else:
        print("-> No purely growing mode with Re(omega)=0 was found in the specified m and gamma search ranges.")
else:
    print("No pure-imaginary eigenmodes found across all m.")
print("=" * 62)

# Save to CSV
csv_filename = 'swmhd_pure_imaginary_modes.csv'
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['m', 'mode_index', 'omega_r_hat', 'gamma_hat', 'gamma_dimensional', 'sigma_min', 'relative_sigma', 'residual', 'stability'])

    # Sort all modes by m, then by gamma_hat
    all_pure_modes.sort(key=lambda x: (x['m'], x['gamma_hat']))

    idx_map = {}
    for mode in all_pure_modes:
        m = mode['m']
        idx_map[m] = idx_map.get(m, 0) + 1
        writer.writerow([
            m,
            idx_map[m],
            0.0,
            mode['gamma_hat'],
            mode['gamma_dimensional'],
            mode['sigma_min'],
            mode['relative_sigma'],
            mode['residual'],
            mode['stability']
        ])
print(f"Saved: {csv_filename}")

summary_csv_filename = 'swmhd_pure_imaginary_summary.csv'
with open(summary_csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['m', 'most_unstable_gamma_hat', 'most_unstable_gamma', 'stability'])
    for m in m_arr:
        mu = most_unstable_per_m.get(m)
        if mu is not None:
            writer.writerow([m, mu['gamma_hat'], mu['gamma_dimensional'], mu['stability']])
        else:
            writer.writerow([m, 'NaN', 'NaN', 'none found'])
print(f"Saved: {summary_csv_filename}")


# Plot purely imaginary dispersion relation
fig_stab, ax_stab = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax_stab.axhline(0, color='grey', lw=1.0, ls='--')

x_m = []
y_gamma = []
colors = []

for m in m_arr:
    mu = most_unstable_per_m.get(m)
    if mu is not None:
        x_m.append(m)
        y_gamma.append(mu['gamma_hat'])
        if mu['gamma_hat'] > 0:
            colors.append('r')
        elif mu['gamma_hat'] < 0:
            colors.append('b')
        else:
            colors.append('k')

ax_stab.scatter(x_m, y_gamma, c=colors, marker='o', s=60, zorder=5)
ax_stab.plot(x_m, y_gamma, color='k', lw=1.5, zorder=4, alpha=0.6)

ax_stab.set_xlabel('Azimuthal wavenumber $m$', fontsize=20)
ax_stab.set_ylabel(r"Growth/decay rate $\gamma/(2\Omega)$", fontsize=20)
ax_stab.set_xlim(0.7, M_max+0.5)
ax_stab.set_xticks(m_arr[::3])
ax_stab.grid(True, alpha=0.22)
ax_stab.set_title(r'Most unstable purely-imaginary mode', fontsize=22)

fig_stab.savefig("swmhd_pure_imaginary_dispersion.png", dpi=300, bbox_inches='tight')
print("Saved: swmhd_pure_imaginary_dispersion.png")

print("\nNote: This calculation searches only for purely imaginary eigenfrequencies")
print("with Re(omega)=0. No purely growing mode with Re(omega)=0 was found")
print("in the specified m and gamma search ranges (if applicable).")
