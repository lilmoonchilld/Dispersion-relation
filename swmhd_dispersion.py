import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det
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


# ── 0.  Physical parameters ─────────────────────────────────────────────────
MU0   = 4 * np.pi * 1e-7
Omega = 0.5e-4        # rad/s
H0    = 500.0         # m
g     = 9.81          # m/s^2
B0    = 0         # T
rho0  = 1000.0        # kg/m^3
C     = 1.5e9         # m^3/s^2

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

# ── Dimensionless Parameters ────────────────────────────────────────────────
f_scale = 2.0 * Omega
hat_r1 = r1 / r0
hat_r2 = r2 / r0
hat_c0sq = c0sq / (Omega**2 * r0**2)
hat_VA2 = VA2 / (Omega**2 * r0**2)
hat_VA2sq=np.sqrt(hat_VA2)
gamma = 2.0 * C / (Omega**2 * r0**3)
hat_omA = np.sqrt(VA2) / (f_scale * H0)
hat_omA2 = hat_omA**2

print("=" * 62)
print("  System Parameters (Dimensionless)")
print("=" * 62)
print(f"  Three Governing Numbers:")
print(f"    1. Magnetic-Coriolis ratio (hat_VA2sq) : {hat_VA2sq:.4f}")
print(f"    2. Inverse Of Froude Number Square          (hat_c0^2)    : {hat_c0sq:.4f}")
print(f"    3. Radial-gravity ratio    (gamma)       : {gamma:.4f}")
print("-" * 62)
print(f"  Domain: hat_r in [{hat_r1:.3f}, {hat_r2:.3f}]")
print()

# ── 1.  Collocation machinery ───────────────────────────────────────────────
N_col = 64

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
    Pv = 1.0 / hat_r_grid - ( 1+gamma) * (hat_r_grid - 1.0) / hat_c0sq

    # Q_coeff
    term1 = 4.0 * hat_omega * (ws_hat**2 - 1.0) / (ws_hat * hat_c0sq)
    term2 = -m_val**2 / hat_r_grid**2
    term3 = -(1+ gamma) * (2.0 * hat_r_grid - 1.0) / (hat_c0sq * hat_r_grid)
    term4 = -m_val * ( 1+gamma) * (hat_r_grid - 1.0) / (ws_hat * hat_c0sq * hat_r_grid)
    Qv = term1 + term2 + term3 + term4

    L = D2_mat + np.diag(Pv) @ D1_mat + np.diag(Qv)
    for idx in [0, N_col - 1]:
        rb = hat_r_grid[idx]
        bc_eta_coeff = -(ws_hat * (1+ gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
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

    r_grid = np.linspace(hat_r1, hat_r2, 100)

    local_roots = []
    for r in r_grid:
        R_r = (1 + gamma) * (2*r - 1) / r
        S_r = m_val * (1 + gamma) * (r - 1) / r
        kappa2_r = hat_kr**2 + (m_val**2) / (r**2)

        K_r = hat_c0sq * kappa2_r + R_r

        # 4*omega^4 + (8*hat_omA2 - 4 - K_r)*omega^2 - S_r * omega + (4*hat_omA2^2 - hat_omA2 * K_r) = 0
        C4 = 4.0
        C3 = 0.0
        C2 = 8.0 * hat_omA2 - 4.0 - K_r
        C1 = -S_r
        C0 = 4.0 * hat_omA2**2 - hat_omA2 * K_r

        roots = np.roots([C4, C3, C2, C1, C0])
        # Only keep real parts if the root is relatively real, but polynomial could have complex roots if mode goes unstable
        # However for global matching we only want strictly real physical modes
        # We'll sort by real part.
        roots_sorted = np.sort(np.real(roots))
        local_roots.append(roots_sorted)

    local_roots = np.array(local_roots)

    # Distance based tracking across the r_grid to maintain branch identity
    num_r = len(r_grid)
    tracked_roots = np.zeros_like(local_roots)
    tracked_roots[0] = local_roots[0]
    for i in range(1, num_r):
        prev = tracked_roots[i-1]
        curr = local_roots[i]

        dist = np.abs(curr[:, None] - prev[None, :])
        assigned_curr = np.zeros(4)
        for j in range(4):
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            assigned_curr[min_idx[1]] = curr[min_idx[0]]
            dist[min_idx[0], :] = np.inf
            dist[:, min_idx[1]] = np.inf

        tracked_roots[i] = assigned_curr

    avg_roots = np.nanmean(tracked_roots, axis=0)

    # Sort them to assign: MP_m, MS/R, R/MS, MP_p
    # avg_roots is already roughly sorted because we sorted the first element by real part, and then tracked.
    # Let's ensure strict order for branch assignment
    avg_roots = np.sort(avg_roots)

    if len(avg_roots) < 4:
        # Fallback
        oMP_m, mid1, mid2, oMP_p = np.nan, np.nan, np.nan, np.nan
    else:
        oMP_m = avg_roots[0]
        mid1 = avg_roots[1]
        mid2 = avg_roots[2]
        oMP_p = avg_roots[3]

    # Identify R and MS from the middle roots.
    # To differentiate Rossby and MS, we can use the old analytical approximation
    # at r=1 to see which one it's closer to.
    hat_k2_avg = hat_kr**2 + (m_val/1.0)**2
    oR_approx = - (1+gamma) * m_val / (2.0 * (hat_k2_avg + 4.0 / hat_c0sq))

    if abs(mid1 - oR_approx) < abs(mid2 - oR_approx):
        oR_hat = mid1
        oMS_hat = mid2
    else:
        oR_hat = mid2
        oMS_hat = mid1

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

    return oMP_p, oMP_m, oMK_in, oMK_out, oR_hat, oMS_hat

# ── 3. Branch assignment with global minimum distance logic ─────────────────
def assign_branches_global(eigs, m_val, N_max=20):
    wkb_preds = {}

    _, _, oMK_in, oMK_out, _, _ = wkb_branches(m_val, 1)

    if not np.isnan(oMK_in):
        wkb_preds[('MK_in', 0)] = oMK_in
    if not np.isnan(oMK_out):
        wkb_preds[('MK_out', 0)] = oMK_out

    for n in range(1, N_max + 1):
        oMP_p, oMP_m, _, _, oR, oMS = wkb_branches(m_val, n)
        if not np.isnan(oMP_p): wkb_preds[('MP_p', n)] = oMP_p
        if not np.isnan(oMP_m): wkb_preds[('MP_m', n)] = oMP_m
        if not np.isnan(oR): wkb_preds[('R', n)] = oR
        if not np.isnan(oMS): wkb_preds[('MS', n)] = oMS

    distances = []
    for i, eig in enumerate(eigs):
        for branch_id, wkb_val in wkb_preds.items():
            dist = abs(eig - wkb_val)
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

for n in range(1, 10):
    WKB_MP_p[n] = np.zeros(len(m_fine))
    WKB_MP_m[n] = np.zeros(len(m_fine))
    WKB_R[n] = np.zeros(len(m_fine))
    WKB_MS[n] = np.zeros(len(m_fine))

    for i, m in enumerate(m_fine):
        op, om, _, _, oR, oMS = wkb_branches(m, n)
        WKB_MP_p[n][i] = op
        WKB_MP_m[n][i] = om
        WKB_R[n][i] = oR
        WKB_MS[n][i] = oMS

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

        alpha_val = max(0.2, 0.7 - 0.08*(n-1))

        ax_obj.plot(m_fine, WKB_MP_p[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MP, zorder=3)
        ax_obj.plot(m_fine, WKB_MP_m[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, zorder=3)
        ax_obj.plot(m_fine, WKB_R[n], color=COLORS['R'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_R, zorder=3)
        ax_obj.plot(m_fine, WKB_MS[n], color=COLORS['MS'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MS, zorder=3)

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
    ax_obj.set_ylim(-50,50)
    ax_obj.set_xticks(m_arr[::2])
    ax_obj.grid(True, alpha=0.22)

    # Custom limit for right zoom panel
    if zoom:
        ax_obj.set_ylim(-0.01, 0.01)

draw_panel(ax, zoom=False)

ax.set_title('Full spectrum', fontsize=11)

# Annotate n=1,2 on left panel at m=1 for MP
for n, dy in [(1, 1.5), (2, 1.5)]:
    Oh_n1 = wkb_branches(1, n)[0] # MP_p
    if not np.isnan(Oh_n1):
        ax.annotate(f'$n={n}$', xy=(1, Oh_n1), xytext=(2.5, Oh_n1+dy*n*0.6),
                    fontsize=9, color=COLORS['MP'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['MP'], lw=0.8))

draw_panel(ax2, zoom=True)
ax2.set_title(r'Slow branches zoom (Rossby & MS)', fontsize=11)

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
    legend_handles,
    legend_labels,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),   # outside right
    fontsize=13,                  # same as axis-title font
    frameon=True,
    borderaxespad=0.5,
    labelspacing=0.8,
    handlelength=2.2
)

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

fig.subplots_adjust(right=0.78)

plt.tight_layout(rect=[0, 0.12, 0.78, 1])
plt.savefig('swmhd_four_branch_dispersion.pdf', dpi=180, bbox_inches='tight')
plt.savefig('swmhd_four_branch_dispersion.png', dpi=180, bbox_inches='tight')
print("\nSaved: swmhd_four_branch_dispersion.pdf / .png")


# ==========================================================
# OPTIONAL: Save legend separately
# Place this after creating legend_handles and legend_labels.
# ==========================================================

fig_leg = plt.figure(figsize=(10, 2))
fig_leg.legend(
    legend_handles,
    legend_labels,
    loc="center",
    ncol=4,
    fontsize=13,
    frameon=True,
)

fig_leg.savefig("Legend.pdf", dpi=300, bbox_inches="tight")
fig_leg.savefig("Legend.png", dpi=300, bbox_inches="tight")
plt.close(fig_leg)

# ==========================================================
# OPTIONAL: Save each panel separately
# ==========================================================
# Left panel:
fig_left, ax_left = plt.subplots(figsize=(8,6))
draw_panel(ax_left, zoom=False)
fig_left.tight_layout()
ax_left.set_ylim(-50,50)
fig_left.savefig("FullSpectrum.pdf", dpi=300, bbox_inches="tight")
fig_left.savefig("FullSpectrum.png", dpi=300, bbox_inches="tight")

# Right panel:
fig_zoom, ax_zoom = plt.subplots(figsize=(6,6))
draw_panel(ax_zoom, zoom=True)
ax_zoom.set_ylim(-0.01,0.01)
fig_zoom.tight_layout()
fig_zoom.savefig("ZoomSpectrum.pdf", dpi=300, bbox_inches="tight")
fig_zoom.savefig("ZoomSpectrum.png", dpi=300, bbox_inches="tight")
