import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# Publication style settings (Preserved from Reference Code)
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
# 0. Physical parameters (Dimensional - Reference Code Values)
# ==========================================================
MU0 = 4 * np.pi * 1e-7
Omega = 0.5e-4      # rad/s
H0 = 500.0          # m
g = 9.81            # m/s^2
B0 = 5e-4           # T
rho0 = 1000.0       # kg/m^3
C = 0.0             # m^3/s^2

r1 = 0.5e6          # inner radius [m]
r2 = 1.0e6          # outer radius [m]
r0 = 0.5 * (r1 + r2)# mean radius [m]

VA2 = B0**2 / (MU0 * rho0)
omA2 = VA2 / H0**2
omA = np.sqrt(omA2)
beta_eff = 2.0 * C / r0**3
c0sq = g * H0
c0 = np.sqrt(c0sq)

# Plotting normalization scale for zero rotation limit
PLOT_FREQ_SCALE = 1e-4  # rad/s

def print_system_parameters():
    print("=" * 62)
    print(" System Parameters (Dimensional SWMHD Solver)")
    print("=" * 62)
    print(f" 1. Coriolis rotation rate (Omega)   : {Omega:.4e} rad/s")
    print(f" 2. Base layer thickness (H0)        : {H0:.1f} m")
    print(f" 3. Vertical magnetic field (B0)    : {B0:.4e} T")
    print(f" 4. Alfven frequency (omega_A)       : {omA:.4e} rad/s")
    print(f" 5. Effective beta (beta_eff)        : {beta_eff:.4e} 1/(s^2 m)")
    print("-" * 62)
    print(f" Domain: r in [{r1/1e6:.3f}, {r2/1e6:.3f}] 10^6 m (r0 = {r0/1e6:.3f} 10^6 m)")
    print("=" * 62)

# ==========================================================
# 1. Collocation machinery (Dimensional Physical Coordinates)
# ==========================================================
N_col = 64

def chebyshev_lobatto(N, r_min, r_max):
    """
    Returns Chebyshev Lobatto grid rp (in metres) and differentiation matrices
    D1 (1/m) and D2 (1/m^2) mapped onto [r_min, r_max].
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
                D[i, k] = (c[i] / c[k]) * (-1)**(i + k) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    sc = 2.0 / (r_max - r_min)
    D1 = sc * D
    D2 = D1 @ D1
    rp = 0.5 * (r_min + r_max) + 0.5 * (r_max - r_min) * xi
    rp = rp[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]
    return rp, D1, D2

r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, r1, r2)

def build_dimensional_matrices(m_val, N=N_col, r_g=r_grid, D1=D1_mat):
    """
    Constructs the 5-field dimensional pencil A q = lambda B q
    q = [eta, u_r, u_theta, b_r, b_theta]^T
    where lambda = -i * omega (dimensional frequency in rad/s).
    """
    A = np.zeros((5*N, 5*N), dtype=complex)
    B = np.zeros((5*N, 5*N), dtype=complex)

    # 1. Continuity: lambda * eta + H0 (d u_r/dr + u_r/r + i m u_theta/r) = 0
    for i in range(N):
        B[i, i] = 1.0
        A[i, N:2*N] = - H0 * D1[i, :]
        A[i, N+i] -= H0 / r_g[i]
        A[i, 2*N+i] = - 1j * m_val * H0 / r_g[i]

    # 2. Radial momentum:
    # lambda * H0 * u_r + 2 Omega H0 u_theta + g H0 d_eta/dr - beta_eff (r-r0) eta + B0/(MU0*rho0) b_r = 0
    for i in range(1, N-1):
        B[N+i, N+i] = H0
        A[N+i, 0:N] = - g * H0 * D1[i, :]
        A[N+i, i] += beta_eff * (r_g[i] - r0)
        A[N+i, 2*N+i] = - 2.0 * Omega * H0
        A[N+i, 3*N+i] = - B0 / (MU0 * rho0)

    # Impermeable wall boundary conditions u_r(r1) = 0 and u_r(r2) = 0
    A[N, N] = 1.0
    B[N, :] = 0.0

    A[2*N-1, 2*N-1] = 1.0
    B[2*N-1, :] = 0.0

    # 3. Azimuthal momentum:
    # lambda * H0 * u_theta - 2 Omega H0 u_r + i m g H0 / r eta + B0/(MU0*rho0) b_theta = 0
    for i in range(N):
        B[2*N+i, 2*N+i] = H0
        A[2*N+i, i] = - 1j * m_val * g * H0 / r_g[i]
        A[2*N+i, N+i] = 2.0 * Omega * H0
        A[2*N+i, 4*N+i] = - B0 / (MU0 * rho0)

    # 4. Radial induction: lambda * H0 * b_r + B0 * u_r = 0
    for i in range(N):
        B[3*N+i, 3*N+i] = H0
        A[3*N+i, N+i] = - B0

    # 5. Azimuthal induction: lambda * H0 * b_theta + B0 * u_theta = 0
    for i in range(N):
        B[4*N+i, 4*N+i] = H0
        A[4*N+i, 2*N+i] = - B0

    return A, B

def solve_dimensional_eigenproblem(m_val, N=N_col, r_g=r_grid, D1=D1_mat):
    """
    Solves A q = lambda B q directly for dimensional frequencies omega = i * lambda.
    Filters infinite, NaN, and non-physical complex eigenvalues.
    Returns sorted real dimensional frequencies in rad/s.
    """
    A, B = build_dimensional_matrices(m_val, N, r_g, D1)
    evals, _ = sla.eig(A, B)

    # Discard infinite or NaN eigenvalues
    finite_mask = np.isfinite(evals)
    evals = evals[finite_mask]

    # Physical frequencies omega = i * lambda = 1j * evals
    omega = 1j * evals

    # Retain physical oscillatory modes with negligible imaginary numerical residual
    real_mask = np.abs(omega.imag) < 1e-3 * (np.abs(omega.real) + 1e-10)
    real_omegas = omega.real[real_mask]

    # Remove duplicates
    unique_roots = []
    for w in sorted(real_omegas):
        if not any(abs(w - ur) < 1e-8 for ur in unique_roots):
            unique_roots.append(w)

    return np.array(sorted(unique_roots))

# ==========================================================
# 2. WKB analytic dispersion relations (Dimensional)
# ==========================================================
def wkb_branches(m_val, n_radial=1):
    """
    WKB branch estimates derived consistently in dimensional units [rad/s].
    """
    kr = n_radial * np.pi / (r2 - r1)
    r_val = r0

    kappa2_1 = kr**2 + m_val**2 / r_val**2
    R_1 = beta_eff * (2.0 * r_val - r0) / r_val
    K_1 = c0sq * kappa2_1 + R_1

    # Dimensional quartic at r = r0:
    # omega^4 + A2*omega^2 + A1*omega + A0 = 0
    # At r = r0, S_1 = 0 => A1 = 0
    A2 = 2.0 * omA2 - 4.0 * Omega**2 - K_1
    A0 = omA2 * (omA2 - K_1)

    disc = A2**2 - 4.0 * A0
    disc = max(disc, 0.0)

    X_plus = (-A2 + np.sqrt(disc)) / 2.0
    X_minus = (-A2 - np.sqrt(disc)) / 2.0

    # Magneto-Poincare branches
    if X_plus >= 0.0:
        oMP_p = +np.sqrt(X_plus)
        oMP_m = -np.sqrt(X_plus)
    else:
        oMP_p = np.nan
        oMP_m = np.nan

    # Magnetostrophic / slow branch
    if X_minus >= 0.0:
        oMS_p = +np.sqrt(X_minus)
        oMS_m = -np.sqrt(X_minus)
    else:
        oMS_p = np.nan
        oMS_m = np.nan

    oMS_val = oMS_m

    # Rossby branch
    r_grid_wkb = np.linspace(r1, r2, 10)
    oR_arr = []
    for r in r_grid_wkb:
        R_r = beta_eff * (2.0 * r - r0) / r
        S_r = 2.0 * Omega * m_val * beta_eff * (r - r0) / r
        kappa2_r = kr**2 + (m_val / r)**2
        K_r = c0sq * kappa2_r + R_r

        if B0 == 0:
            if abs(4.0 * Omega**2 + K_r) > 1e-14:
                oR_val = -S_r / (4.0 * Omega**2 + K_r)
            else:
                oR_val = np.nan
        else:
            if abs(S_r) > 1e-12:
                a = omA2
                oR_val = (4.0 * a**2 - a * K_r) / S_r
            else:
                oR_val = np.nan
        oR_arr.append(oR_val)

    oR_arr = np.asarray(oR_arr)

    # Magneto-Kelvin branches
    arg_in = m_val**2 * c0sq / r1**2 - 4.0 * omA2
    if arg_in >= 0.0 and omA <= m_val * c0 / (2.0 * r1):
        oMK_in = -0.5 * np.sqrt(arg_in)
    else:
        oMK_in = np.nan

    arg_out = m_val**2 * c0sq / r2**2 - 4.0 * omA2
    if arg_out >= 0.0 and omA <= m_val * c0 / (2.0 * r2):
        oMK_out = +0.5 * np.sqrt(arg_out)
    else:
        oMK_out = np.nan

    return oMP_p, oMP_m, oMK_in, oMK_out, oR_arr, oMS_val

# ==========================================================
# 3. Branch assignment with global minimum distance logic
# ==========================================================
def assign_branches_global(eigs, m_val, N_max=20):
    """
    Global minimum-distance assignment comparing dimensional collocation frequencies
    with dimensional WKB predictions [rad/s].
    """
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
# Helper: Plot frequency conversion function
# ==========================================================
def norm_freq(omega_val):
    """
    Converts dimensional frequency [rad/s] to plot representation:
    omega / (2 * Omega) when Omega != 0, or omega / PLOT_FREQ_SCALE when Omega == 0.
    """
    if Omega != 0.0:
        return omega_val / (2.0 * Omega)
    else:
        return omega_val / PLOT_FREQ_SCALE

# ==========================================================
# 4. Legacy solver comparison function (Regression check)
# ==========================================================
def find_eigenvalues_legacy(m_val, omega_range=(-45, 45), n_scan=4000):
    """
    Original dimensionless determinant-based solver retained purely for regression testing.
    Returns dimensional frequencies = 2 * Omega * hat_omega.
    """
    if Omega == 0.0:
        return np.array([])

    f_scale = 2.0 * Omega
    hat_r1 = r1 / r0
    hat_r2 = r2 / r0
    hat_c0sq = c0sq / (Omega**2 * r0**2)
    gamma = 2.0 * C / (Omega**2 * r0**3)
    hat_omA = np.sqrt(VA2) / (f_scale * H0)
    hat_omA2 = hat_omA**2

    hat_r_grid, D1_m, D2_m = chebyshev_lobatto(N_col, hat_r1, hat_r2)

    def omega_star_hat(ho):
        return ho + hat_omA2 / ho

    def build_matrix(ho):
        ws_hat = omega_star_hat(ho)
        Pv = 1.0 / hat_r_grid - (gamma) * (hat_r_grid - 1.0) / hat_c0sq
        term1 = 4.0 * ho * (ws_hat**2 - 1.0) / (ws_hat * hat_c0sq)
        term2 = -m_val**2 / hat_r_grid**2
        term3 = -(gamma) * (2.0 * hat_r_grid - 1.0) / (hat_c0sq * hat_r_grid)
        term4 = -m_val * (gamma) * (hat_r_grid - 1.0) / (ws_hat * hat_c0sq * hat_r_grid)
        Qv = term1 + term2 + term3 + term4
        L = D2_m + np.diag(Pv) @ D1_m + np.diag(Qv)
        for idx in [0, N_col - 1]:
            rb = hat_r_grid[idx]
            bc_eta_coeff = -(ws_hat * (gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
            L[idx, :] = ws_hat * hat_c0sq * D1_m[idx, :] + bc_eta_coeff * np.eye(N_col)[idx]
        return L

    scan = np.linspace(*omega_range, n_scan)
    scan = scan[np.abs(scan) > 0.01]
    ld = []
    for w in scan:
        try:
            d = np.log(np.abs(sla.det(build_matrix(w))) + 1e-300)
        except Exception:
            d = np.nan
        ld.append(d)

    roots = []
    for k in range(1, len(ld) - 1):
        if np.isfinite(ld[k]) and ld[k] < ld[k-1] and ld[k] < ld[k+1]:
            w = complex(scan[k])
            for _ in range(80):
                dw = 1e-6*abs(w) + 1e-10
                try:
                    f0 = sla.det(build_matrix(w))
                    fp = ((sla.det(build_matrix(w+dw)) - sla.det(build_matrix(w-dw))) / (2*dw))
                    if abs(fp) < 1e-300: break
                    step = -f0 / fp; w += step
                    if abs(step) < 1e-10 * (abs(w) + 1): break
                except Exception:
                    break
            try:
                residual = abs(sla.det(build_matrix(w)))
                valid_ld = [x for x in ld if np.isfinite(x)]
                ld_max = max(valid_ld) if len(valid_ld) > 0 else 0
                if residual < 1e-3 * np.exp(ld_max):
                    dup = any(abs(w - wr) < 5e-3 for wr in roots)
                    if not dup:
                        roots.append(w)
            except Exception:
                pass
    hat_roots = np.array(sorted([wr.real for wr in roots]))
    return hat_roots * (2.0 * Omega)

# ==========================================================
# 5. Main Solver Execution Function
# ==========================================================
def main():
    print_system_parameters()

    M_max = 30
    m_arr = np.arange(1, M_max+1)
    m_fine = np.linspace(1, M_max, 1000)

    print(f"Running dimensional 5-field collocation solver N={N_col}, m=1..{M_max} ...")

    mk_in_data = []
    mk_out_data = []
    mp_p_data = {}
    mp_m_data = {}
    r_data = {}
    ms_data = {}

    for m in m_arr:
        eigs = solve_dimensional_eigenproblem(m)
        assigned = assign_branches_global(eigs, m)

        if assigned['MK_in'] is not None:
            mk_in_data.append((m, norm_freq(assigned['MK_in'])))
        if assigned['MK_out'] is not None:
            mk_out_data.append((m, norm_freq(assigned['MK_out'])))

        for n, val in assigned['MP_p'].items():
            mp_p_data.setdefault(n, []).append((m, norm_freq(val)))
        for n, val in assigned['MP_m'].items():
            mp_m_data.setdefault(n, []).append((m, norm_freq(val)))
        for n, val in assigned['R'].items():
            r_data.setdefault(n, []).append((m, norm_freq(val)))
        for n, val in assigned['MS'].items():
            ms_data.setdefault(n, []).append((m, norm_freq(val)))

        pos = sorted([norm_freq(e) for e in eigs if norm_freq(e) > 0])
        neg = sorted([norm_freq(e) for e in eigs if norm_freq(e) < 0], reverse=True)
        if m <= 3 or m % 5 == 0:
            print(f"  m={m:2d}: +{[f'{x:.3f}' for x in pos[:4]]}... "
                  f"-{[f'{abs(x):.3f}' for x in neg[:4]]}...")

    # Compute fine WKB curves
    WKB_MK_in = np.array([norm_freq(wkb_branches(m, 1)[2]) for m in m_fine])
    WKB_MK_out = np.array([norm_freq(wkb_branches(m, 1)[3]) for m in m_fine])
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
            op_list.append(norm_freq(op))
            om_list.append(norm_freq(om))
            oR_list.append(norm_freq(oR_arr))
            oMS_list.append(norm_freq(oMS))

        WKB_MP_p[n] = np.array(op_list)
        WKB_MP_m[n] = np.array(om_list)
        WKB_R[n] = np.array(oR_list).T
        WKB_MS[n] = np.array(oMS_list)

    # ==========================================================
    # 6. Plotting (Reference Code Architecture & Style)
    # ==========================================================
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios':[1.6,1]})
    fig.suptitle(
        r'SWMHD global dispersion relation (Dimensional)' '\n\n'
        r'Magneto-Poincaré (blue), Magneto-Kelvin (green), Rossby (red), Magnetostrophic (orange)',
        fontsize=13, fontweight='bold'
    )

    COLORS = {
        'MP': '#1f77b4',
        'MK': '#2ca02c',
        'R':  '#d62728',
        'MS': '#ff7f0e',
    }

    def draw_panel(ax_obj, zoom=False):
        # WKB curves
        ax_obj.plot(m_fine, WKB_MK_in, color=COLORS['MK'], lw=2.5, label='Magneto-Kelvin (WKB)', zorder=4)
        ax_obj.plot(m_fine, WKB_MK_out, color=COLORS['MK'], lw=2.5, zorder=4)

        for n in range(1, 9):
            lbl_MP = f'Magneto-Poincaré $n={n}$ (WKB)' if n<=2 and not zoom else ('Magneto-Poincaré (WKB)' if n==1 else None)
            lbl_R = f'Rossby $n={n}$ (WKB)' if n<=2 and not zoom else ('Rossby (WKB)' if n==1 else None)
            lbl_MS = f'Magnetostrophic $n={n}$ (WKB)' if n<=2 and not zoom else ('Magnetostrophic (WKB)' if n==1 else None)

            alpha_val = max(0.1, 0.4 - 0.05*(n-1))

            ax_obj.plot(m_fine, WKB_MP_p[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MP, zorder=3)
            ax_obj.plot(m_fine, WKB_MP_m[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, zorder=3)
            ax_obj.plot(m_fine, WKB_MS[n], color=COLORS['MS'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MS, zorder=3)

            for i in range(1):
                lR = lbl_R if i == 0 else None
                ax_obj.plot(m_fine, WKB_R[n][i], color=COLORS['R'], lw=1.0, ls='-', alpha=alpha_val, label=lR, zorder=3)

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
        plot_branch_collocation(r_data, COLORS['R'], 'Rossby')
        plot_branch_collocation(ms_data, COLORS['MS'], 'Magnetostrophic')

        ax_obj.axhline(0,  color='grey', lw=0.8, ls=':')
        ax_obj.axhline(+1, color='grey', lw=0.9, ls='--', alpha=0.5)
        ax_obj.axhline(-1, color='grey', lw=0.9, ls='--', alpha=0.5)

        hat_omA_val = norm_freq(omA)
        if hat_omA_val > 0:
            ax_obj.axhline(+hat_omA_val, color='purple', lw=0.8, ls=':', alpha=0.6)
            ax_obj.axhline(-hat_omA_val, color='purple', lw=0.8, ls=':', alpha=0.6)

        ax_obj.set_xlabel('Azimuthal wavenumber $m$', fontsize=13)
        if Omega != 0.0:
            ax_obj.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=13)
        else:
            ax_obj.set_ylabel(r'Normalised frequency $\omega/\omega_{\rm scale}$', fontsize=13)

        ax_obj.set_xlim(0.7, M_max+0.5)
        ax_obj.set_ylim(-50, 50)
        ax_obj.set_xticks(m_arr[::2])
        ax_obj.grid(True, alpha=0.22)

        if zoom:
            ax_obj.set_ylim(0, 1.5)

    draw_panel(ax, zoom=False)
    ax.set_title('Full spectrum', fontsize=11)

    # Annotate n=1,2 on left panel at m=1 for MP
    for n_val, dy in [(1, 1.5), (2, 1.5)]:
        Oh_n1 = norm_freq(wkb_branches(1, n_val)[0])
        if not np.isnan(Oh_n1):
            ax.annotate(f'$n={n_val}$', xy=(1, Oh_n1), xytext=(2.5, Oh_n1 + 0.6),
                        fontsize=9, color=COLORS['MP'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['MP'], lw=0.8))

    draw_panel(ax2, zoom=True)
    ax2.set_title(r'Slow branches zoom (Rossby & MS)', fontsize=11)

    # Combined figure legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    seen = set()
    legend_handles = []
    legend_labels = []
    for h, l in zip(all_handles, all_labels):
        if l and l not in seen:
            seen.add(l)
            legend_handles.append(h)
            legend_labels.append(l)

    fig.legend(
        legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=13, frameon=True, borderaxespad=0.5, labelspacing=0.8, handlelength=2.2
    )

    # Parameter box (Dimensional Parameters)
    pbox = (
        rf"$\Omega = {Omega:.1e}$ rad/s, $H_0 = {H0:.1f}$ m\n"
        rf"$B_0 = {B0:.1e}$ T, $\rho_0 = {rho0:.1f}$ kg/m$^3$\n"
        rf"$C = {C:.1e}$ m$^3$/s$^2$, $\beta_{{\mathrm{{eff}}}} = {beta_eff:.1e}$ s$^{{-2}}$m$^{{-1}}$\n"
        rf"$\omega_A = {omA:.4e}$ rad/s\n"
        rf"$r_1 = {r1:.1e}$ m, $r_2 = {r2:.1e}$ m\n"
        rf"$N = {N_col}$"
    )
    fig.text(0.52, 0.015, pbox, fontsize=10, va='bottom', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))

    fig.subplots_adjust(right=0.78)
    plt.tight_layout(rect=[0, 0.12, 0.78, 1])

    out_base = f"swmhd_four_branch_dispersion_g={g}_C={C}_B={B0}"
    fig.savefig(f"{out_base}.png", dpi=180, bbox_inches='tight')
    # fig.savefig(f"{out_base}.pdf", dpi=180, bbox_inches='tight')
    print(f"\nSaved combined figure: {out_base}.png")
    plt.close(fig)

    # Save legend separately
    fig_leg = plt.figure(figsize=(10, 2))
    fig_leg.legend(
        legend_handles, legend_labels, loc="center", ncol=4, fontsize=13, frameon=True
    )
    fig_leg.savefig(f"Legend_g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches="tight")
    # fig_leg.savefig(f"Legend_g={g}_C={C}_B={B0}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_leg)

    # Save FullSpectrum separately
    fig_left, ax_left = plt.subplots(figsize=(8, 6))
    draw_panel(ax_left, zoom=False)
    ax_left.set_ylim(-40, 40)
    fig_left.tight_layout()
    fig_left.savefig(f"FullSpectrum_g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches="tight")
    # fig_left.savefig(f"FullSpectrum_g={g}_C={C}_B={B0}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_left)

    # Save ZoomSpectrum separately
    fig_zoom, ax_zoom = plt.subplots(figsize=(6, 6))
    draw_panel(ax_zoom, zoom=True)
    ax_zoom.set_ylim(-1.5, 1.5)
    fig_zoom.tight_layout()
    fig_zoom.savefig(f"ZoomSpectrum_g={g}_C={C}_B={B0}.png", dpi=300, bbox_inches="tight")
    # fig_zoom.savefig(f"ZoomSpectrum_g={g}_C={C}_B={B0}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_zoom)

    print("Saved individual panel figures: FullSpectrum, ZoomSpectrum, and Legend PNGs.")

if __name__ == "__main__":
    main()
