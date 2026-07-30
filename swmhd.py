import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det
import warnings
warnings.filterwarnings("ignore")

# ── 0.  Physical parameters ─────────────────────────────────────────────────
MU0   = 4 * np.pi * 1e-7
Omega = 0.5e-4        # rad/s
H0    = 500.0         # m
g     = 9.81          # m/s^2
B0    = 0.0           # T
rho0  = 1000.0        # kg/m^3
C     = 0.0           # m^3/s^2

r1    = 0.5e7         # inner radius [m]
r2    = 1e7           # outer radius [m]
r0    = 0.5 * (r1 + r2)

VA2      = B0**2 / (MU0 * rho0)
omA2     = VA2 / H0**2
omA      = np.sqrt(omA2)
beta_g   = 2.0 * C / r0**3
beta_eff = beta_g
c0sq     = g * H0
c0       = np.sqrt(c0sq)
f        = 2.0 * Omega          # Coriolis parameter

# ── Dimensionless Parameters ────────────────────────────────────────────────
f_scale = 2.0 * Omega
hat_r1 = r1 / r0
hat_r2 = r2 / r0
hat_c0sq = c0sq / (Omega**2 * r0**2)
hat_VA2 = VA2 / (Omega**2 * r0**2)
gamma = 2.0 * C / (Omega**2 * r0**3)
hat_omA = np.sqrt(VA2) / (f_scale * H0)
hat_omA2 = hat_omA**2

print("=" * 62)
print("  System Parameters (Dimensionless)")
print("=" * 62)
print(f"  Three Governing Numbers:")
print(f"    1. Magnetic-Coriolis ratio (hat_omega_A) : {hat_omA:.4f}")
print(f"    2. Burger number           (hat_c0^2)    : {hat_c0sq:.4f}")
print(f"    3. Radial-gravity ratio    (gamma)       : {gamma:.4f}")
print("-" * 62)
print(f"  Domain: hat_r in [{hat_r1:.3f}, {hat_r2:.3f}]")
print()

# ── 1.  Collocation machinery ───────────────────────────────────────────────
N_col = 32

def chebyshev_lobatto(N, r_min, r_max):
    j = np.arange(N); xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N); c[0] = 2; c[-1] = 2
    X = np.tile(xi, (N, 1)); dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
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
    Pv = 1.0 / hat_r_grid - (1.0 + gamma) * (hat_r_grid - 1.0) / hat_c0sq

    # Q_coeff
    term1 = 4.0 * hat_omega * (ws_hat**2 - 1.0) / (ws_hat * hat_c0sq)
    term2 = -m_val**2 / hat_r_grid**2
    term3 = -(1.0 + gamma) * (2.0 * hat_r_grid - 1.0) / (hat_c0sq * hat_r_grid)
    term4 = -m_val * (1.0 + gamma) * (hat_r_grid - 1.0) / (ws_hat * hat_c0sq * hat_r_grid)
    Qv = term1 + term2 + term3 + term4

    L = D2_mat + np.diag(Pv) @ D1_mat + np.diag(Qv)
    for idx in [0, N_col - 1]:
        rb = hat_r_grid[idx]
        bc_eta_coeff = -(ws_hat * (1.0 + gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
        L[idx,:] = ws_hat * hat_c0sq * D1_mat[idx,:] + bc_eta_coeff * np.eye(N_col)[idx]
    return L

def find_eigenvalues(m_val, omega_range=(-45, 45), n_scan=3000):
    """Return sorted real eigenfrequencies (dimensionless) for azimuthal mode m_val."""
    scan = np.linspace(*omega_range, n_scan)
    # Avoid zero singularity
    scan = scan[np.abs(scan) > 0.01]
    ld = []
    for w in scan:
        try:
            d = np.log(np.abs(det(build_matrix(w, m_val))) + 1e-300)
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
                    f0 = det(build_matrix(w, m_val))
                    fp = ((det(build_matrix(w+dw, m_val))
                           - det(build_matrix(w-dw, m_val)))
                          / (2*dw))
                    if abs(fp) < 1e-300: break
                    step = -f0 / fp; w += step
                    if abs(step) < 1e-10 * (abs(w) + 1): break
                except Exception:
                    break
            try:
                residual = abs(det(build_matrix(w, m_val)))
                valid_ld = [x for x in ld if np.isfinite(x)]
                ld_max = max(valid_ld) if len(valid_ld) > 0 else 0
                if residual < 1e-3 * np.exp(ld_max):
                    dup = any(abs(w - wr) < 5e-3 for wr in roots)
                    if not dup:
                        roots.append(w)
            except Exception:
                pass
    return np.array(sorted([wr.real for wr in roots]))

# ── 2.  WKB analytic dispersion relations (Dimensionless) ────────────────────
def wkb_branches(m_val, n_radial=1):
    hat_kr = n_radial * np.pi / (hat_r2 - hat_r1)

    r_grid = np.linspace(hat_r1, hat_r2, 10)

    oMP_m_arr = []
    oMP_p_arr = []
    oR_hat_arr = []
    oMS_hat_arr = []

    for r in r_grid:
        hat_kth = m_val / r
        hat_k2 = hat_kr**2 + hat_kth**2

        R = (1+gamma)*(2*r - 1)/r
        S = m_val*(1+gamma)*(r - 1)/r

        A2 = 2.0*hat_omA2 - 1.0 - 0.25*(hat_c0sq*hat_k2 + R)
        A1 = -0.25 * S
        A0 = hat_omA2 * (hat_omA2 - 0.25*(hat_c0sq*hat_k2 + R))

        coeffs = [1.0, 0.0, A2, A1, A0]
        rts = np.roots(coeffs)
        # Filter for real roots
        rts = [x.real for x in rts if abs(x.imag) < 1e-6]

        # Identify roots if 4 are present
        if len(rts) == 4:
            rts = sorted(rts)
            oMP_m_val = rts[0]
            mid1 = rts[1]
            mid2 = rts[2]
            oMP_p_val = rts[3]

            # Tie breaker for R vs MS
            oR_approx = - (1+gamma) * m_val / (2.0 * (hat_k2 + 4.0 / hat_c0sq))
            if abs(mid1 - oR_approx) < abs(mid2 - oR_approx):
                oR_val = mid1
                oMS_val = mid2
            else:
                oR_val = mid2
                oMS_val = mid1
        elif len(rts) == 2:
            rts = sorted(rts)
            oMP_m_val = rts[0]
            oMP_p_val = rts[1]
            oR_val = np.nan
            oMS_val = np.nan
        else:
            oMP_m_val = np.nan
            oMP_p_val = np.nan
            oR_val = np.nan
            oMS_val = np.nan

        oMP_m_arr.append(oMP_m_val)
        oMP_p_arr.append(oMP_p_val)
        oR_hat_arr.append(oR_val)
        oMS_hat_arr.append(oMS_val)

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

    return np.array(oMP_p_arr), np.array(oMP_m_arr), oMK_in, oMK_out, np.array(oR_hat_arr), np.array(oMS_hat_arr)

# ── 3. Branch assignment with global minimum distance logic ─────────────────
def assign_branches_global(eigs, m_val, N_max=20):
    wkb_preds = {}

    _, _, oMK_in, oMK_out, _, _ = wkb_branches(m_val, 1)

    if not np.isnan(oMK_in):
        wkb_preds[('MK_in', 0)] = oMK_in
    if not np.isnan(oMK_out):
        wkb_preds[('MK_out', 0)] = oMK_out

    for n in range(1, N_max + 1):
        oMP_p_arr, oMP_m_arr, _, _, oR_arr, oMS_arr = wkb_branches(m_val, n)
        wkb_preds[('MP_p', n)] = oMP_p_arr
        wkb_preds[('MP_m', n)] = oMP_m_arr
        wkb_preds[('R', n)] = oR_arr
        wkb_preds[('MS', n)] = oMS_arr

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

# ── 4.  Compute spectrum ──────────────────────────────────────────────────────
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

# ── 5.  WKB curves for fine array ────────────────────────────────────────────
WKB_MK_in = np.array([wkb_branches(m, 1)[2] for m in m_fine])
WKB_MK_out = np.array([wkb_branches(m, 1)[3] for m in m_fine])
WKB_MP_p = {}
WKB_MP_m = {}
WKB_R = {}
WKB_MS = {}

# We evaluated wkb_branches at 10 discrete points for each m.
# WKB_MP_p[n] will be a 2D array: shape (10 points, len(m_fine)).
for n in range(1, 10):
    op_list = []
    om_list = []
    oR_list = []
    oMS_list = []

    for m in m_fine:
        op_arr, om_arr, _, _, oR_arr, oMS_arr = wkb_branches(m, n)
        op_list.append(op_arr)
        om_list.append(om_arr)
        oR_list.append(oR_arr)
        oMS_list.append(oMS_arr)

    WKB_MP_p[n] = np.array(op_list).T
    WKB_MP_m[n] = np.array(om_list).T
    WKB_R[n] = np.array(oR_list).T
    WKB_MS[n] = np.array(oMS_list).T

# ── 6.  Plot ──────────────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                               gridspec_kw={'width_ratios':[1.6,1]})
fig.suptitle(
    r'SWMHD global dispersion relation (Dimensionless)'
    '\n'
    r'Magneto-Poincaré (blue), Magneto-Kelvin (green), Rossby (red), Magnetostrophic (orange)',
    fontsize=13, fontweight='bold')

COLORS = {
    'MP':  '#1f77b4',
    'MK':  '#2ca02c',
    'R':   '#d62728',
    'MS':  '#ff7f0e',
}

def draw_panel(ax_obj, zoom=False):
    # WKB curves
    ax_obj.plot(m_fine, WKB_MK_in, color=COLORS['MK'], lw=2.5,
                label='Magneto-Kelvin (WKB)', zorder=4)
    ax_obj.plot(m_fine, WKB_MK_out, color=COLORS['MK'], lw=2.5, zorder=4)

    for n in range(1, 9):
        lbl_MP = f'Magneto-Poincaré $n={n}$ (WKB)' if n<=2 and not zoom else ('Magneto-Poincaré (WKB)' if n==1 else None)
        lbl_R = f'Rossby $n={n}$ (WKB)' if n<=2 and not zoom else ('Rossby (WKB)' if n==1 else None)
        lbl_MS = f'Magnetostrophic $n={n}$ (WKB)' if n<=2 and not zoom else ('Magnetostrophic (WKB)' if n==1 else None)

        alpha_val = max(0.1, 0.4 - 0.05*(n-1))

        # Plot the 10 lines for each branch to form a continuum band
        for i in range(10):
            lMP = lbl_MP if i == 0 else None
            lR = lbl_R if i == 0 else None
            lMS = lbl_MS if i == 0 else None

            ax_obj.plot(m_fine, WKB_MP_p[n][i], color=COLORS['MP'], lw=1.0, ls='-', alpha=alpha_val, label=lMP, zorder=3)
            ax_obj.plot(m_fine, WKB_MP_m[n][i], color=COLORS['MP'], lw=1.0, ls='-', alpha=alpha_val, zorder=3)
            ax_obj.plot(m_fine, WKB_R[n][i], color=COLORS['R'], lw=1.0, ls='-', alpha=alpha_val, label=lR, zorder=3)
            ax_obj.plot(m_fine, WKB_MS[n][i], color=COLORS['MS'], lw=1.0, ls='-', alpha=alpha_val, label=lMS, zorder=3)

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

    if hat_omA > 0:
        ax_obj.axhline(+hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
        ax_obj.axhline(-hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)

    ax_obj.set_xlabel('Azimuthal wavenumber $m$', fontsize=13)
    ax_obj.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=13)
    ax_obj.set_xlim(0.7, M_max+0.5)
    ax_obj.set_xticks(m_arr[::2])
    ax_obj.grid(True, alpha=0.22)

    # Custom limit for right zoom panel
    if zoom:
        ax_obj.set_ylim(-0.4, 0.4)

draw_panel(ax, zoom=False)

# deduplicate legend for left panel
handles, labels = ax.get_legend_handles_labels()
seen = set()
unique_handles = []
unique_labels = []
for h, lbl in zip(handles, labels):
    if lbl not in seen:
        seen.add(lbl)
        unique_handles.append(h)
        unique_labels.append(lbl)
ax.legend(unique_handles, unique_labels, fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
ax.set_title('Full spectrum', fontsize=11)

# Annotate n=1,2 on left panel at m=1 for MP
for n, dy in [(1, 1.5), (2, 1.5)]:
    Oh_n1_arr = wkb_branches(1, n)[0] # MP_p array
    if not np.all(np.isnan(Oh_n1_arr)):
        Oh_n1 = np.nanmean(Oh_n1_arr)
        ax.annotate(f'$n={n}$', xy=(1, Oh_n1), xytext=(2.5, Oh_n1+dy*n*0.6),
                    fontsize=9, color=COLORS['MP'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['MP'], lw=0.8))

draw_panel(ax2, zoom=True)
ax2.set_title(r'Slow branches zoom (Rossby & MS)', fontsize=11)

handles, labels = ax2.get_legend_handles_labels()
seen = set()
unique_handles = []
unique_labels = []
for h, lbl in zip(handles, labels):
    if lbl not in seen and ("Rossby" in lbl or "Magnetostrophic" in lbl) and "WKB" in lbl:
        seen.add(lbl)
        unique_handles.append(h)
        unique_labels.append(lbl)
ax2.legend(unique_handles, unique_labels, fontsize=8, loc='lower left', framealpha=0.9)

# Parameter box
pbox = (
    f"$\\hat{{\\omega}}_A = {hat_omA:.4f}$\n"
    f"$\\hat c_0^2 = {hat_c0sq:.4f}$\n"
    f"$\\gamma = {gamma:.4f}$\n"
    f"$\\Omega = {Omega:.1e}$ rad/s,  $H_0 = {H0}$ m\n"
    f"$r_1 = {r1:.1e}$ m,  $r_2 = {r2:.1e}$ m\n"
    f"$N = {N_col}$"
)
fig.text(0.52, 0.015, pbox, fontsize=10, va='bottom', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig('swmhd_four_branch_dispersion.pdf', dpi=180, bbox_inches='tight')
plt.savefig('swmhd_four_branch_dispersion.png', dpi=180, bbox_inches='tight')
print("\nSaved: swmhd_four_branch_dispersion.pdf / .png")
