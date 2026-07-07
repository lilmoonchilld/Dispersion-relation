import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det
import warnings
warnings.filterwarnings("ignore")

# ── 0.  Physical parameters ─────────────────────────────────────────────────
MU0   = 4 * np.pi * 1e-7
Omega = 0.5e-4        # rad/s
H0    = 500       # m
g     = 9.81       # m/s^2
B0    = 0      # T
rho0  = 1000.0     # kg/m^3
C     = 0    # m^3/s^2

r1    = 0.5e6        # inner radius [m]
r2    = 1e6        # outer radius [m]
r0    = 0.5 * (r1 + r2)

VA2      = B0**2 / (MU0 * rho0)
omA2     = VA2 / H0**2
omA      = np.sqrt(omA2)
beta_g   = 2.0 * C / r0**3
beta_eff =  beta_g
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
N_col = 24

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

def build_matrix(hat_omega, m_val, N=N_col, r_grid=hat_r_grid, D1=D1_mat, D2=D2_mat):
    ws_hat = omega_star_hat(hat_omega)

    # P_coeff: 1/r - (1+gamma)(r-1)/c0^2
    Pv = 1.0 / r_grid - (1.0 + gamma) * (r_grid - 1.0) / hat_c0sq

    # Q_coeff
    term1 = 4.0 * hat_omega * (ws_hat**2 - 1.0) / (ws_hat * hat_c0sq)
    term2 = -m_val**2 / r_grid**2
    term3 = -(1.0 + gamma) * (2.0 * r_grid - 1.0) / (hat_c0sq * r_grid)
    term4 = -m_val * (1.0 + gamma) * (r_grid - 1.0) / (ws_hat * hat_c0sq * r_grid)
    Qv = term1 + term2 + term3 + term4

    L = D2 + np.diag(Pv) @ D1 + np.diag(Qv)
    for idx in [0, N - 1]:
        rb = r_grid[idx]
        bc_eta_coeff = -(ws_hat * (1.0 + gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
        L[idx,:] = ws_hat * hat_c0sq * D1[idx,:] + bc_eta_coeff * np.eye(N)[idx]
    return L

def find_eigenvalues_raw(m_val, N, omega_range=(-40, 40), n_scan=3000):
    scan = np.linspace(*omega_range, n_scan)
    scan = scan[np.abs(scan) > 0.01]
    r_grid, D1, D2 = chebyshev_lobatto(N, hat_r1, hat_r2)
    ld = []
    for w in scan:
        try:
            d = np.log(np.abs(det(build_matrix(w, m_val, N, r_grid, D1, D2))) + 1e-300)
        except Exception:
            d = np.nan
        ld.append(d)

    roots = []
    for k in range(1, len(ld) - 1):
        if ld[k] < ld[k-1] and ld[k] < ld[k+1]:
            w = complex(scan[k])
            for _ in range(80):
                dw = 1e-6*abs(w) + 1e-10
                try:
                    f0 = det(build_matrix(w, m_val, N, r_grid, D1, D2))
                    fp = ((det(build_matrix(w+dw, m_val, N, r_grid, D1, D2))
                           - det(build_matrix(w-dw, m_val, N, r_grid, D1, D2)))
                          / (2*dw))
                    if abs(fp) < 1e-300: break
                    step = -f0 / fp; w += step
                    if abs(step) < 1e-10 * (abs(w) + 1): break
                except Exception:
                    break
            try:
                residual = abs(det(build_matrix(w, m_val, N, r_grid, D1, D2)))
                valid_ld = [x for x in ld if np.isfinite(x)]
                ld_max = max(valid_ld) if len(valid_ld) > 0 else 0
                if residual < 1e-3 * np.exp(ld_max):
                    dup = any(abs(w - wr) < 5e-3 for wr in roots)
                    if not dup:
                        roots.append(w)
            except Exception:
                pass
    return np.array(sorted([wr.real for wr in roots]))

def find_eigenvalues(m_val, omega_range=(-40, 40), n_scan=3000):
    """Return sorted real eigenfrequencies iteratively filtered."""
    roots_N = find_eigenvalues_raw(m_val, N_col, omega_range, n_scan)
    roots_N2 = find_eigenvalues_raw(m_val, N_col + 2, omega_range, n_scan)

    if len(roots_N2) == 0:
        return np.array([])

    tol = 1e-6
    filtered_roots = []

    while tol <= 0.1:
        filtered_roots = []
        for r in roots_N:
            diffs = np.abs(roots_N2 - r)
            min_diff = np.min(diffs)
            rel_diff = min_diff / max(abs(r), 1.0)
            if rel_diff < tol:
                filtered_roots.append(r)

        # We expect 4 physical branches (MP+, MP-, MK out, MK in) for small C, plus maybe a few Rossby.
        # But specifically, if m is small and hat_c0sq is moderate, there are at least 4 fast modes.
        # So we should only break if we have at least 4 roots.
        if len(filtered_roots) >= 4:
            break

        tol *= 10

    return np.array(filtered_roots)


# ── 2.  WKB analytic dispersion relations (Dimensionless) ────────────────────
def wkb_branches(m_val, n_radial=3):
    """
    Returns dimensionless frequencies (hat_omega) for:
    (MP_plus, MP_minus, MK_outer, MK_inner, R, MS)
    """
    hat_r = 1.0
    hat_kr = n_radial * np.pi / (hat_r2 - hat_r1)
    hat_kth = m_val / hat_r
    hat_k2 = hat_kr**2 + hat_kth**2

    hat_R = 1.0 + gamma
    hat_S = 0.0

    # Poincare Quartic Roots
    A2 = 2.0 * hat_omA2 - 1.0 - 0.25 * (hat_c0sq * hat_k2 + hat_R)
    A1 = -0.25 * hat_S
    A0 = hat_omA2 * (hat_omA2 - 0.25 * (hat_c0sq * hat_k2 + hat_R))

    disc = A2**2 - 4.0 * A0
    if disc < 0: disc = 0.0

    X1 = (-A2 + np.sqrt(disc)) / 2.0
    X2 = (-A2 - np.sqrt(disc)) / 2.0

    oMP_p = +np.sqrt(max(X1, 0))
    oMP_m = -np.sqrt(max(X1, 0))

    # Magneto-Kelvin Outer Boundary
    outer_arg = (m_val**2 * hat_c0sq) / hat_r2**2 - 4.0 * hat_omA2
    oMK_outer = 0.5 * np.sqrt(outer_arg) if outer_arg >= 0 else np.nan

    # Magneto-Kelvin Inner Boundary
    inner_arg = (m_val**2 * hat_c0sq) / hat_r1**2 - 4.0 * hat_omA2
    oMK_inner = -0.5 * np.sqrt(inner_arg) if inner_arg >= 0 else np.nan

    # Rossby
    oR_hat = - (1.0 + gamma) * m_val / (2.0 * (hat_k2 + 4.0 / hat_c0sq))

    # Magnetostrophic
    oMS_hat = - hat_VA2 * hat_k2 / (4.0 + hat_c0sq * hat_k2)

    return oMP_p, oMP_m, oMK_outer, oMK_inner, oR_hat, oMS_hat

# ── 3.  Compute everything ───────────────────────────────────────────────────
M_max  = 30
m_arr  = np.arange(1, M_max + 1)
m_fine = np.linspace(1, M_max, 200)

wkb = {key: [] for key in ['MP_p','MP_m','MK_out','MK_in','R','MS']}
for m_val in m_fine:
    oMP_p, oMP_m, oMK_out, oMK_in, oR, oMS = wkb_branches(m_val)
    wkb['MP_p'].append(oMP_p)
    wkb['MP_m'].append(oMP_m)
    wkb['MK_out'].append(oMK_out)
    wkb['MK_in'].append(oMK_in)
    wkb['R'].append(oR)
    wkb['MS'].append(oMS)

print("Running Chebyshev collocation for m = 1 …", M_max, "…")
col_eigs = {}
for m_val in m_arr:
    eigs = find_eigenvalues(m_val)
    col_eigs[m_val] = eigs
    pos = sorted([e for e in eigs if e > 0])
    neg = sorted([e for e in eigs if e < 0], reverse=True)
    if m_val <= 3 or m_val % 1 == 0:
        print(f"  m={m_val:2d}: +{[f'{x:.3f}' for x in pos]}  "
              f"-{[f'{abs(x):.3f}' for x in neg]}")

print()

# ── 4.  Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                         gridspec_kw={'width_ratios': [1.6, 1]})

COLORS = {
    'MP':  '#1f77b4',
    'MK':  '#2ca02c',
    'R':   '#d62728',
    'MS':  '#ff7f0e',
    'col': 'k',
}

ax = axes[0]

for m_val in m_arr:
    eigs_n = col_eigs[m_val]
    ax.scatter([m_val] * len(eigs_n), eigs_n,
               s=36, color=COLORS['col'], zorder=5,
               alpha=0.85, marker='o')

ax.axhline(0,   color='grey', lw=0.7, ls=':')
ax.axhline(+1,  color='grey', lw=0.6, ls='--', alpha=0.5)
ax.axhline(-1,  color='grey', lw=0.6, ls='--', alpha=0.5)
if hat_omA > 0:
    ax.axhline(+hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
    ax.axhline(-hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
    ax.text(M_max + 0.05, hat_omA + 0.1, r'$\hat\omega = \hat\omega_A$',
            va='bottom', ha='left', fontsize=8, color='purple')

ax.text(M_max + 0.05, 1.05,  r'$\hat\omega = 1$',
        va='bottom', ha='left', fontsize=8, color='grey')

handles, labels = ax.get_legend_handles_labels()
col_dot = plt.Line2D([0],[0], marker='o', color='w',
                     markerfacecolor='k', markersize=7,
                     label='Global eigenvalues (collocation)')
ax.legend(handles=handles + [col_dot],
          loc='upper left', fontsize=9, framealpha=0.9)

ax.set_xlabel('Azimuthal wavenumber  $m$', fontsize=13)
ax.set_ylabel(r'Normalised frequency  $\hat\omega$', fontsize=13)
ax.set_title('SWMHD global dispersion relation\n' r'($\hat\omega$ vs $m$, dimensionless)', fontsize=12)
ax.set_xlim(0.7, M_max + 0.4)
ax.set_xticks(m_arr)
ax.grid(True, alpha=0.25)

# ── Right panel: zoom on slow branches ───────────────────────────────────────
ax2 = axes[1]
for m_val in m_arr:
    slow = [e for e in col_eigs[m_val] if abs(e) < 1]
    if slow:
        ax2.scatter([m_val]*len(slow), slow, s=50,
                    color=COLORS['col'], zorder=5, marker='o')

ax2.axhline(0,  color='grey', lw=0.7, ls=':')
ax2.axhline(-1, color='grey', lw=0.6, ls='--', alpha=0.5,
            label=r'$\hat\omega = -1$')
ax2.set_xlabel('Azimuthal wavenumber  $m$', fontsize=13)
ax2.set_ylabel(r'$\hat\omega$', fontsize=13)
ax2.set_title('Slow branches (zoom)', fontsize=12)
ax2.set_xlim(0.7, M_max + 0.4)
ax2.set_ylim(-0.5, 0.5)
ax2.set_xticks(m_arr)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.25)

param_text = (
    "$\\hat{\\omega}_A = %.4f$" % hat_omA + "\n" +
    "$\\hat{c}_0^2 = %.4f$" % hat_c0sq + "\n" +
    "$\\gamma = %.4f$" % gamma + "\n" +
    "$\\hat{r}_1 = %.3f$, $\\hat{r}_2 = %.3f$" % (hat_r1, hat_r2)
)
fig.text(0.52, 0.015, param_text,
         fontsize=10, va='bottom', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig('swmhd_four_branch_dispersion.pdf', dpi=180, bbox_inches='tight')
plt.savefig('swmhd_four_branch_dispersion.png', dpi=180, bbox_inches='tight')
print("Saved: swmhd_four_branch_dispersion.pdf / .png")

# ── 5.  Print interpretation table ───────────────────────────────────────────
print()
print("=" * 68)
print("  WKB dimensionless eigenfrequencies (hat_omega)")
print("=" * 68)
print(f"{'m':>3}  {'MP_p':>8}  {'MP_m':>8}  {'MK_out':>8}  {'MK_in':>8}  "
      f"{'Rossby':>10}  {'MS':>10}")
print("-" * 68)
for m_val in m_arr:
    oMP_p, oMP_m, oMK_out, oMK_in, oR, oMS = wkb_branches(m_val)
    out_val = f"{oMK_out:>8.4f}" if not np.isnan(oMK_out) else f"{'NaN':>8}"
    in_val  = f"{oMK_in:>8.4f}" if not np.isnan(oMK_in) else f"{'NaN':>8}"
    print(f"{m_val:>3}  {oMP_p:>8.4f}  {oMP_m:>8.4f}  {out_val}  {in_val}  "
          f"{oR:>10.5f}  {oMS:>10.5f}")
