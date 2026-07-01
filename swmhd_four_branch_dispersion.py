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

Curves: WKB local dispersion (analytic, dimensionless)
Dots:   Chebyshev collocation eigenvalues (global, finite-annulus)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det
import warnings
warnings.filterwarnings("ignore")

# ── 0.  Physical parameters ─────────────────────────────────────────────────
MU0   = 4 * np.pi * 1e-7
Omega = 1e-4        # rad/s
H0    = 50       # m
g     = 9.81       # m/s^2
B0    = 1e-5      # T
rho0  = 1000.0     # kg/m^3
C     = 3e16       # m^3/s^2

r1    = 9e7        # inner radius [m]
r2    = 1e8        # outer radius [m]
r0    = 0.5 * (r1 + r2)

VA2      = B0**2 / (MU0 * rho0)
omA2     = VA2 / H0**2
omA      = np.sqrt(omA2)
beta_g   = 2.0 * C / r0**3
beta_eff = Omega**2 + beta_g
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
N_base = 32
N_test = 34
drift_tolerance = 1e-6

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


def omega_star_hat(hat_omega):
    """Dimensionless modified frequency hat_omega_* = hat_omega + hat_omega_A^2 / hat_omega."""
    return hat_omega + hat_omA2 / hat_omega

def build_matrix(hat_omega, m_val, hat_r_grid, D1_mat, D2_mat, N):
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
    for idx in [0, N - 1]:
        rb = hat_r_grid[idx]
        bc_eta_coeff = -(ws_hat * (1.0 + gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
        L[idx,:] = ws_hat * hat_c0sq * D1_mat[idx,:] + bc_eta_coeff * np.eye(N)[idx]
    return L

def find_eigenvalues(m_val, N, omega_range=(-10, 10), n_scan=2000):
    """Return sorted real eigenfrequencies (dimensionless) for azimuthal mode m_val at resolution N."""
    hat_r_grid, D1_mat, D2_mat = chebyshev_lobatto(N, hat_r1, hat_r2)

    scan = np.linspace(*omega_range, n_scan)
    # Avoid zero singularity
    scan = scan[np.abs(scan) > 0.01]
    ld = []
    for w in scan:
        try:
            d = np.log(np.abs(det(build_matrix(w, m_val, hat_r_grid, D1_mat, D2_mat, N))) + 1e-300)
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
                    f0 = det(build_matrix(w, m_val, hat_r_grid, D1_mat, D2_mat, N))
                    fp = ((det(build_matrix(w+dw, m_val, hat_r_grid, D1_mat, D2_mat, N))
                           - det(build_matrix(w-dw, m_val, hat_r_grid, D1_mat, D2_mat, N)))
                          / (2*dw))
                    if abs(fp) < 1e-300: break
                    step = -f0 / fp; w += step
                    if abs(step) < 1e-10 * (abs(w) + 1): break
                except Exception:
                    break
            try:
                residual = abs(det(build_matrix(w, m_val, hat_r_grid, D1_mat, D2_mat, N)))
                valid_ld = [x for x in ld if np.isfinite(x)]
                ld_max = max(valid_ld) if len(valid_ld) > 0 else 0
                if residual < 1e-3 * np.exp(ld_max):
                    dup = any(abs(w - wr) < 5e-3 for wr in roots)
                    if not dup:
                        roots.append(w)
            except Exception:
                pass
    return np.array(sorted([wr.real for wr in roots]))


def filter_spurious_roots(roots_N, roots_N2, tolerance):
    """
    Compares roots found at resolution N against resolution N+2.
    Only keeps roots that appear in both sets within the specified tolerance.
    """
    if len(roots_N2) == 0:
        return np.array([])

    valid_roots = []
    for root in roots_N:
        # Minimum absolute distance to any root in N2
        min_dist = np.min(np.abs(roots_N2 - root))
        if min_dist <= tolerance:
            valid_roots.append(root)

    return np.array(valid_roots)

# ── 2.  WKB analytic dispersion relations (Dimensionless) ────────────────────
# Evaluated at hat_r = 1
def wkb_branches(m_val, n_radial=1):
    """
    Returns dimensionless frequencies (hat_omega) for:
    (MP_plus, MP_minus, MK, R, MS)
    Evaluated at hat_r = 1.
    """
    hat_r = 1.0
    # Total wavenumber squared
    # dimensional k_r = n*pi / (r2-r1) -> dimensionless hat_k_r = n*pi / (hat_r2 - hat_r1)
    hat_kr = n_radial * np.pi / (hat_r2 - hat_r1)
    hat_kth = m_val / hat_r
    hat_k2 = hat_kr**2 + hat_kth**2

    # Beta functions at hat_r = 1
    # hat_R(1) = 1 + gamma
    # hat_S(1) = 0
    hat_R = 1.0 + gamma
    hat_S = 0.0

    # Local VA2 proxy: hat_VA2 is V_A^2 / (Omega^2 r_0^2). We need the WKB eq.
    # We solve the quartic for hat_omega evaluated at hat_r=1.

    A2 = 2.0 * hat_omA2 - 1.0 - 0.25 * (hat_c0sq * hat_k2 + hat_R)
    A1 = -0.25 * hat_S  # this is zero at hat_r=1
    A0 = hat_omA2 * (hat_omA2 - 0.25 * (hat_c0sq * hat_k2 + hat_R))

    # For A1=0, it's a biquadratic: hat_omega^4 + A2*hat_omega^2 + A0 = 0
    disc = A2**2 - 4.0 * A0
    if disc < 0: disc = 0.0

    # roots for X = hat_omega^2
    X1 = (-A2 + np.sqrt(disc)) / 2.0   # Fast mode (Poincare)
    X2 = (-A2 - np.sqrt(disc)) / 2.0   # Slow mode (Alfvén/Rossby hybrid)

    oMP_p = +np.sqrt(max(X1, 0))
    oMP_m = -np.sqrt(max(X1, 0))

    # Magneto-Kelvin limit: ω ≈ ±√(c0²+VA²)·kθ -> hat_omega = ± sqrt(c0^2+VA^2)*kth / 2Omega
    # Wait, the prompt says "non-dispersive". Using the exact analytical limit:
    # omega_MK / f = sqrt(hat_c0sq + hat_VA2) * hat_kth / 2 (from dimensional formula)
    # let's map it directly from dimensional: oMK = np.sqrt(c0sq + VA2) * (m/r0)
    # oMK / f = np.sqrt(c0sq + VA2) * m / (2*Omega*r0)
    # c0sq/(Omega^2 r0^2) = hat_c0sq => c0sq = hat_c0sq * Omega^2 r0^2
    oMK_hat = np.sqrt(hat_c0sq + hat_VA2) * hat_kth / 2.0

    # Rossby (B=0 analytical limit): ω_R = -β_eff·kθ / (k² + f²/c0²)
    # beta_eff = Omega^2(1+gamma). So beta_eff / f = Omega^2(1+gamma)/(2Omega) = Omega(1+gamma)/2
    # This evaluates more cleanly:
    # oR_hat = - (1+gamma) * hat_kth / (2 * (hat_k2 + 4/hat_c0sq))
    oR_hat = - (1.0 + gamma) * hat_kth / (2.0 * hat_k2 + 8.0 / hat_c0sq)
    # wait, the exact dimensional is: oR = -beta_eff * (m/r0) / (k2 + f^2/c0sq)
    # oR / f = - (Omega^2(1+gamma) * (m/r0) / (k_r^2 + m^2/r0^2 + 4Omega^2/c0sq)) / (2Omega)
    # = - (1+gamma) * m / (2 * (hat_kr^2 + m^2 + 4/hat_c0sq))
    oR_hat = - (1.0 + gamma) * m_val / (2.0 * (hat_k2 + 4.0 / hat_c0sq))

    # Magnetostrophic: oMS = -2*Omega * VA2 * k2 / (f * (f**2 + c0sq*k2))
    # oMS / f = -2*Omega * VA2 * (hat_k2/r0^2) / (2*Omega * (4*Omega^2 + c0sq*hat_k2/r0^2))
    # = - hat_VA2 * hat_k2 / (4 + hat_c0sq * hat_k2)
    oMS_hat = - hat_VA2 * hat_k2 / (4.0 + hat_c0sq * hat_k2)

    return oMP_p, oMP_m, oMK_hat, oR_hat, oMS_hat

# ── 3.  Compute everything ───────────────────────────────────────────────────
M_max  = 30
m_arr  = np.arange(1, M_max + 1)
m_fine = np.linspace(1, M_max, 200)

wkb = {key: [] for key in ['MP_p','MP_m','MK','R','MS']}
for m_val in m_fine:
    oMP_p, oMP_m, oMK, oR, oMS = wkb_branches(m_val)
    wkb['MP_p'].append(oMP_p)
    wkb['MP_m'].append(oMP_m)
    wkb['MK'].append(oMK)
    wkb['R'].append(oR)
    wkb['MS'].append(oMS)

print("Running Chebyshev collocation (with N vs N+2 filtering) for m = 1 …", M_max, "…")
col_eigs = {}
for m_val in m_arr:
    roots_N = find_eigenvalues(m_val, N_base)
    roots_N2 = find_eigenvalues(m_val, N_test)

    eigs = filter_spurious_roots(roots_N, roots_N2, drift_tolerance)
    col_eigs[m_val] = eigs

    pos = sorted([e for e in eigs if e > 0])
    neg = sorted([e for e in eigs if e < 0], reverse=True)
    if m_val <= 3 or m_val % 5 == 0:
        print(f"  m={m_val:2d} ({len(roots_N)} -> {len(eigs)} roots): "
              f"+{[f'{x:.3f}' for x in pos]}  "
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

ax.plot(m_fine, wkb['MP_p'], color=COLORS['MP'], lw=2.0, ls='-',
        label='Magneto-Poincaré (WKB)')
ax.plot(m_fine, wkb['MP_m'], color=COLORS['MP'], lw=2.0, ls='--')
ax.plot(m_fine, wkb['MK'],   color=COLORS['MK'], lw=2.0, ls='-',
        label='Magneto-Kelvin (WKB)')
ax.plot(m_fine, wkb['R'],    color=COLORS['R'],  lw=2.0, ls='-',
        label='Rossby (WKB)')
ax.plot(m_fine, wkb['MS'],   color=COLORS['MS'], lw=2.0, ls='-',
        label='Magnetostrophic (WKB)')

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
ax.set_title(r'SWMHD global dispersion relation' '\n'
             r'($\hat\omega$ vs $m$, dimensionless)', fontsize=12)
ax.set_xlim(0.7, M_max + 0.4)
ax.set_xticks(m_arr)
ax.grid(True, alpha=0.25)

# ── Right panel: zoom on slow branches ───────────────────────────────────────
ax2 = axes[1]
ax2.plot(m_fine, wkb['R'],  color=COLORS['R'],  lw=2.0, ls='-',
         label='Rossby (WKB)')
ax2.plot(m_fine, wkb['MS'], color=COLORS['MS'], lw=2.0, ls='-',
         label='Magnetostrophic (WKB)')

for m_val in m_arr:
    slow = [e for e in col_eigs[m_val] if abs(e) < 1.2]
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
ax2.set_ylim(-0.2, 0.1)
ax2.set_xticks(m_arr)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.25)

param_text = (
    f"$\\hat{{\\omega}}_A = {hat_omA:.4f}$\n"
    f"$\\hat{{c}}_0^2 = {hat_c0sq:.4f}$\n"
    f"$\\gamma = {gamma:.4f}$\n"
    f"$\\hat{{r}}_1 = {hat_r1:.3f}$, $\\hat{{r}}_2 = {hat_r2:.3f}$"
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
print("=" * 62)
print("  WKB dimensionless eigenfrequencies (hat_omega)")
print("=" * 62)
print(f"{'m':>3}  {'MP_p':>8}  {'MP_m':>8}  {'MK':>8}  "
      f"{'Rossby':>10}  {'MS':>10}")
print("-" * 62)
for m_val in m_arr:
    oMP_p, oMP_m, oMK, oR, oMS = wkb_branches(m_val)
    print(f"{m_val:>3}  {oMP_p:>8.4f}  {oMP_m:>8.4f}  "
          f"{oMK:>8.4f}  {oR:>10.5f}  {oMS:>10.5f}")
