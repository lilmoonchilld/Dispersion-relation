"""
SWMHD Global Dispersion Relation — Four-Branch Dispersion Plot
==============================================================
Y-axis: normalised frequency  omega_hat = omega / (2*Omega)  (= omega/f)
X-axis: azimuthal wavenumber  m  (integer, 1..M_max)

Four wave branches plotted:
  1. Magneto-Poincaré (MP)  — fast inertio-gravity modified by B
  2. Magneto-Kelvin   (MK)  — gravity-Alfvén hybrid, non-dispersive
  3. Rossby           (R)   — retrograde slow waves, driven by beta_eff PV gradient
  4. Magnetostrophic  (MS)  — magnetically modified Rossby, |omega| << f

Curves: WKB local dispersion (analytic)
Dots:   Chebyshev collocation eigenvalues (global, finite-annulus)

Physical parameters (SI) — identical to swmhd_global_dispersion.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import svd
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# ── 0.  Physical parameters ─────────────────────────────────────────────────
MU0   = 4 * np.pi * 1e-7
Omega = 5e-5          # rad/s  (since f = 2*Omega = 1e-4)
D0    = 500.0         # m (D0)
g     = 9.81          # m/s^2
B0    = 0.085         # T
rho0  = 1000.0        # kg/m^3
C     = 1e9           # m^3/s^2

r1    = 5e5           # inner radius [m] (500 km)
r2    = 1e6           # outer radius [m] (1000 km)
r0    = 0.5 * (r1 + r2) # 750 km

VA2      = B0**2 / (MU0 * rho0)
omA2     = VA2 / D0**2
omA      = np.sqrt(omA2)
beta_eff = 2.0 * C / (r0**3) # Do not add Omega^2
c0sq     = g * D0
c0       = np.sqrt(c0sq)
f        = 2.0 * Omega          # Coriolis parameter

# Normalisation frequency
f_norm   = f                    # omega_hat = omega / f

print("=" * 62)
print("  Parameters")
print("=" * 62)
print(f"  f = 2Ω = {f:.4e} rad/s   (normalisation frequency)")
print(f"  ω_A   = {omA:.4e} rad/s   ω_A/f = {omA/f:.2f}")
print(f"  c0    = {c0:.4f} m/s")
print(f"  β_eff = {beta_eff:.4e} s⁻²")
print(f"  r0={r0:.3f} m,  annulus width Δr={r2-r1:.2f} m")
print(f"  B0={B0:.4f} T")
print()

# ── 1.  Collocation machinery ──────────────────────────────────────────────
N_col = 32

def chebyshev_lobatto(N, r1, r2):
    j = np.arange(N); xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N); c[0] = 2; c[-1] = 2
    X = np.tile(xi, (N, 1)); dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    sc = 2.0 / (r2 - r1); D1 = sc * D; D2 = D1 @ D1
    rp = 0.5*(r1+r2) + 0.5*(r2-r1)*xi
    rp = rp[::-1]; D1 = D1[::-1,::-1]; D2 = D2[::-1,::-1]
    return rp, D1, D2

r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, r1, r2)

def build_matrix(omega, m_val):
    if abs(omega) < 1e-15: return np.eye(N_col) * 1e10 # Avoid division by zero
    ws = omega + omA2 / omega

    P_vec = 1.0 / r_grid - beta_eff * (r_grid - r0) / c0sq

    Q_vec = (-omega * (4 * Omega**2 - ws**2) / (ws * c0sq)
             - (m_val / r_grid)**2
             - beta_eff * (2 * r_grid - r0) / (c0sq * r_grid)
             - 2 * Omega * m_val * beta_eff * (r_grid - r0) / (ws * c0sq * r_grid))

    L = D2_mat + np.diag(P_vec) @ D1_mat + np.diag(Q_vec)

    # Boundary conditions: omega_* [ d_eta/dr - beta_eff*(r-r0)*eta ] - 2*Omega*(m/r)*eta = 0
    for idx in [0, N_col - 1]:
        rb = r_grid[idx]
        L[idx, :] = ws * (D1_mat[idx, :] - beta_eff * (rb - r0) * np.eye(N_col)[idx]) \
                    - 2 * Omega * (m_val / rb) * np.eye(N_col)[idx]

    return L

def svd_indicator(omega, m_val):
    L = build_matrix(omega, m_val)
    if np.any(np.isnan(L)) or np.any(np.isinf(L)):
        return 1.0
    try:
        s = svd(L, compute_uv=False)
        return s[-1] / s[0] if s[0] > 1e-15 else 1.0
    except Exception:
        return 1.0

def find_eigenvalues_svd(m_val, omega_range=(-60*f, 60*f), n_scan=4000):
    """SVD-based local minima search."""
    scan = np.linspace(omega_range[0], omega_range[1], n_scan)
    # Avoid scanning exactly near 0 where ws is singular
    scan = scan[np.abs(scan) > 1e-3 * f]

    S_vals = np.array([svd_indicator(w, m_val) for w in scan])

    roots = []
    for i in range(1, len(S_vals) - 1):
        if S_vals[i] < S_vals[i-1] and S_vals[i] < S_vals[i+1] and S_vals[i] < 1e-1:
            # Refine
            res = minimize_scalar(lambda w: svd_indicator(w, m_val),
                                  bounds=(scan[i-1], scan[i+1]), method='bounded')
            if res.success and res.fun < 1e-1:
                wr = res.x
                dup = any(abs(wr - r) < 5e-3*f for r in roots)
                if not dup:
                    roots.append(wr)

    return np.array(sorted(roots))

# ── 2.  Exact WKB Analytic Dispersion Relations ────────────────────────────

def get_A_coeffs(r_eval, omega, m_val, kr, B_val):
    VA2_eval = B_val**2 / (MU0 * rho0)
    omA2_eval = VA2_eval / D0**2
    kappa2 = kr**2 + (m_val/r_eval)**2

    A2 = 2*omA2_eval - 4*Omega**2 + c0sq*kappa2 + beta_eff*(2*r_eval - r0)/r_eval
    A1 = 2*Omega*m_val*beta_eff*(r_eval - r0)/r_eval
    A0 = omA2_eval*(omA2_eval + c0sq*kappa2 + beta_eff*(2*r_eval - r0)/r_eval)

    return A2, A1, A0

def wkb_roots_full(m_val, r_eval, kr, B_val):
    A2, A1, A0 = get_A_coeffs(r_eval, 0.0, m_val, kr, B_val)
    coeffs = [1.0, 0.0, A2, A1, A0]
    rts = np.roots(coeffs)
    # Filter only real roots
    real_rts = sorted([r.real for r in rts if abs(r.imag) < 1e-10])
    return real_rts

def wkb_hydro_rossby(m_val, r_eval, kr):
    # Pure hydrodynamic limit B=0
    A2_h, A1, A0 = get_A_coeffs(r_eval, 0.0, m_val, kr, 0.0)

    if abs(A2_h) < 1e-15:
        return np.nan
    return -A1 / A2_h


# ── 3.  Compute ────────────────────────────────────────────────────────────
M_max = 30
m_arr = np.arange(1, M_max + 1)
m_fine = np.linspace(1, M_max, 200)

kr_1 = 1 * np.pi / (r2 - r1)

wkb_r0 = {
    'm': m_fine,
    'roots': []
}
for m in m_fine:
    rts = wkb_roots_full(m, r0, kr_1, B0)
    wkb_r0['roots'].append(rts)

delta_r = 0.1 * (r2 - r1)
r_plus = r0 + delta_r
r_minus = r0 - delta_r

wkb_R_plus = []
wkb_R_minus = []
for m in m_fine:
    wkb_R_plus.append(wkb_hydro_rossby(m, r_plus, kr_1))
    wkb_R_minus.append(wkb_hydro_rossby(m, r_minus, kr_1))

col_eigs = {}
print(f"Running Chebyshev collocation for m = 1 … {M_max} …")
for m_val in m_arr:
    eigs = find_eigenvalues_svd(m_val)
    col_eigs[m_val] = eigs / f_norm
    pos = sorted([e/f_norm for e in eigs if e > 0])
    neg = sorted([e/f_norm for e in eigs if e < 0], reverse=True)
print("Done.")

# ── 4.  Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                         gridspec_kw={'width_ratios': [1.6, 1]})

# Colour / style palette
COLORS = {
    'WKB': '#1f77b4',   # blue for full WKB
    'R_plus':   '#d62728',   # red
    'R_minus':  '#2ca02c',   # green
    'col': 'k',         # black dots for collocation
}

ax = axes[0]

# ── WKB curves ────────────────────────────────────────────────────────────────
# Extract up to 4 roots
wkb_array = np.full((len(m_fine), 4), np.nan)
for i, rts in enumerate(wkb_r0['roots']):
    for j, r in enumerate(rts[:4]):
        wkb_array[i, j] = r / f_norm

for j in range(4):
    label = 'Full WKB quartic' if j == 0 else None
    ax.plot(m_fine, wkb_array[:, j], color=COLORS['WKB'], lw=2.0, ls='-', label=label)

# ── Collocation dots ─────────────────────────────────────────────────────────
for m_val in m_arr:
    eigs_n = col_eigs[m_val]
    ax.scatter([m_val] * len(eigs_n), eigs_n,
               s=36, color=COLORS['col'], zorder=5,
               alpha=0.85, marker='o')

# Reference lines
ax.axhline(0,   color='grey', lw=0.7, ls=':')
ax.axhline(+1,  color='grey', lw=0.6, ls='--', alpha=0.5)   # omega = f
ax.axhline(-1,  color='grey', lw=0.6, ls='--', alpha=0.5)
ax.axhline(+omA/f_norm, color='purple', lw=0.8, ls=':', alpha=0.6)
ax.axhline(-omA/f_norm, color='purple', lw=0.8, ls=':', alpha=0.6)

# Annotations for reference lines
ax.text(M_max + 0.05, 1.05,  r'$\hat\omega = 1$  $(f)$',
        va='bottom', ha='left', fontsize=8, color='grey')
ax.text(M_max + 0.05, omA/f_norm + 0.1, r'$\hat\omega = \omega_A/f$',
        va='bottom', ha='left', fontsize=8, color='purple')

# Legend
handles, labels = ax.get_legend_handles_labels()
col_dot = plt.Line2D([0],[0], marker='o', color='w',
                     markerfacecolor='k', markersize=7,
                     label='Global eigenvalues (collocation)')
ax.legend(handles=handles + [col_dot],
          loc='upper left', fontsize=9, framealpha=0.9)

ax.set_xlabel('Azimuthal wavenumber  $m$', fontsize=13)
ax.set_ylabel(r'Normalised frequency  $\hat\omega = \omega / f$', fontsize=13)
ax.set_title('SWMHD global dispersion relation\n'
             r'($\hat\omega$ vs $m$, exact WKB vs Global SVD)', fontsize=12)
ax.set_xlim(0.7, M_max + 0.4)
ax.set_xticks(m_arr[::2])
ax.grid(True, alpha=0.25)

# ── Right panel: zoom on slow branches ───────────────────────────────────────
ax2 = axes[1]
ax2.plot(m_fine, np.array(wkb_R_plus) / f_norm,  color=COLORS['R_plus'],  lw=2.0, ls='-',
         label=r'Rossby B=0 (r0 + $\delta r$)')
ax2.plot(m_fine, np.array(wkb_R_minus) / f_norm, color=COLORS['R_minus'], lw=2.0, ls='-',
         label=r'Rossby B=0 (r0 - $\delta r$)')

# Collocation slow modes (|omega_hat| < 1.2)
for m_val in m_arr:
    slow = [e for e in col_eigs[m_val] if abs(e) < 1.2]
    if slow:
        ax2.scatter([m_val]*len(slow), slow, s=50,
                    color=COLORS['col'], zorder=5, marker='o')

ax2.axhline(0,  color='grey', lw=0.7, ls=':')
ax2.axhline(-1, color='grey', lw=0.6, ls='--', alpha=0.5,
            label=r'$\hat\omega = -1$  $(-f)$')
ax2.set_xlabel('Azimuthal wavenumber  $m$', fontsize=13)
ax2.set_ylabel(r'$\hat\omega = \omega / f$', fontsize=13)
ax2.set_title('Slow branches (zoom) with B=0 Hydro Rossby', fontsize=12)
ax2.set_xlim(0.7, M_max + 0.4)
ax2.set_ylim(-0.2, 0.1)
ax2.set_xticks(m_arr[::2])
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.25)

# ── Parameter box ─────────────────────────────────────────────────────────────
param_text = (
    f"$f = 2\\Omega = {f:.1e}$ rad/s\n"
    f"$\\omega_A = {omA:.3e}$ rad/s  $= {omA/f:.2f}f$\n"
    f"$c_0 = {c0:.2f}$ m/s\n"
    f"$\\beta_{{\\rm eff}} = {beta_eff:.3e}$ s$^{{-2}}$\n"
    f"$r_0 = {r0:.0f}$ m,  "
    f"$\\Delta r = {r2-r1:.0f}$ m\n"
    f"$B_0 = {B0:.3f}$ T,  $D_0 = {D0:.0f}$ m"
)
fig.text(0.52, 0.015, param_text,
         fontsize=8.5, va='bottom', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig('swmhd_four_branch_dispersion.png', dpi=180, bbox_inches='tight')
print("Saved: swmhd_four_branch_dispersion.png")

# ── 5.  Print interpretation table ───────────────────────────────────────────
print()
print("=" * 80)
print("  WKB and Global eigenfrequencies (normalised by f)")
print("=" * 80)
print(f"{'m':>3}  {'WKB (r0)':>30}  {'R_hydro(r0+dr)':>15}  {'R_hydro(r0-dr)':>15}  {'Global (pos)':>20}  {'Global (neg)':>20}")
print("-" * 80)

for m_val in m_arr:
    rts = wkb_roots_full(m_val, r0, kr_1, B0)
    wkb_r0_str = " ".join([f"{r/f:>7.4f}" for r in rts])

    r_plus_val = wkb_hydro_rossby(m_val, r_plus, kr_1) / f
    r_minus_val = wkb_hydro_rossby(m_val, r_minus, kr_1) / f

    eigs = col_eigs[m_val]
    pos = [e for e in eigs if e > 0]
    neg = [e for e in eigs if e < 0]
    pos_str = " ".join([f"{e:>6.3f}" for e in pos])
    neg_str = " ".join([f"{e:>6.3f}" for e in neg])

    print(f"{m_val:>3}  {wkb_r0_str:>30}  {r_plus_val:>15.5f}  {r_minus_val:>15.5f}  {pos_str:>20}  {neg_str:>20}")
