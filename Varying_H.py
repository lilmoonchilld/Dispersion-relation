import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det
import warnings
warnings.filterwarnings("ignore")

# ── 0.  Physical parameters ─────────────────────────────────────────────────
MU0   = 4 * np.pi * 1e-7
Omega = 1e-4        # rad/s
H0    = 100       # m
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

def find_eigenvalues(m_val, omega_range=(-10, 10), n_scan=2000):
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
        if ld[k] < ld[k-1] and ld[k] < ld[k+1]:
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

# ── 2.  Compute everything ───────────────────────────────────────────────────
M_max  = 30
m_arr  = np.arange(1, M_max + 1)

print("Running Chebyshev collocation for m = 1 …", M_max, "…")
col_eigs = {}
for m_val in m_arr:
    eigs = find_eigenvalues(m_val)
    col_eigs[m_val] = eigs

print("\nValid Eigenvalues:")
print(f"{'m':>3}  {'Roots (hat_omega)'}")
print("-" * 62)
for m_val in m_arr:
    eigs = col_eigs[m_val]
    pos = sorted([e for e in eigs if e > 0])
    neg = sorted([e for e in eigs if e < 0], reverse=True)
    out_str = ""
    if pos:
        out_str += "+[" + ", ".join([f"{x:.5f}" for x in pos]) + "] "
    if neg:
        out_str += "-[" + ", ".join([f"{x:.5f}" for x in neg]) + "]"
    if not out_str:
        out_str = "None"
    print(f"{m_val:>3}  {out_str}")


# ── 3.  Plot ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))

for m_val in m_arr:
    eigs_n = col_eigs[m_val]
    # filter out very large non-physical eigenvalues for plot clarity
    eigs_n = [e for e in eigs_n if abs(e) <= 1.5]
    if len(eigs_n) > 0:
        plt.scatter([m_val] * len(eigs_n), eigs_n,
                    s=36, color='k', zorder=5, alpha=0.85, marker='o')

plt.axhline(0, color='grey', lw=0.7, ls=':')
plt.axhline(+1, color='grey', lw=0.6, ls='--', alpha=0.5)
plt.axhline(-1, color='grey', lw=0.6, ls='--', alpha=0.5)
if hat_omA > 0:
    plt.axhline(+hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
    plt.axhline(-hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
    plt.text(M_max + 0.05, hat_omA + 0.02, r'$\hat\omega = \hat\omega_A$',
            va='bottom', ha='left', fontsize=8, color='purple')

plt.text(M_max + 0.05, 1.05,  r'$\hat\omega = 1$',
        va='bottom', ha='left', fontsize=8, color='grey')


plt.xlabel('Azimuthal wavenumber  $m$', fontsize=13)
plt.ylabel(r'Normalised frequency  $\hat{\omega}$', fontsize=13)
plt.title(r'SWMHD Global Dispersion Relation' '\n'
          r'($\hat{\omega}$ vs $m$, varying $H(r)$)', fontsize=12)
plt.xlim(0.7, M_max + 0.4)
plt.ylim(-1.5, 1.5)
plt.xticks(np.arange(1, M_max + 1, 2))
plt.grid(True, alpha=0.25)

param_text = (
    f"$\\hat{{\\omega}}_A = {hat_omA:.4f}$\n"
    f"$\\hat{{c}}_0^2 = {hat_c0sq:.4f}$\n"
    f"$\\gamma = {gamma:.4f}$\n"
    f"$\\hat{{r}}_1 = {hat_r1:.3f}$, $\\hat{{r}}_2 = {hat_r2:.3f}$"
)
plt.figtext(0.15, 0.15, param_text,
         fontsize=10, va='bottom', ha='left',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('Varying_H.png', dpi=180, bbox_inches='tight')
print("\nSaved plot to Varying_H.png")
