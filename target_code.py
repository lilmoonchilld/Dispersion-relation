import numpy as np
from scipy.linalg import det, eig
import matplotlib.pyplot as plt
from itertools import product as iproduct

# =============================================================================
# Physical parameters  (SI units throughout)
# =============================================================================
MU0   = 4 * np.pi * 1e-7   # permeability of free space [H/m]

# --- Adjustable parameters ---------------------------------------------------
Omega  = 1e-4       # rotation rate [rad/s]
H0     = 50.0       # equilibrium depth at r0 [m]
g      = 10.44      # vertical gravity [m/s^2]
B0     = 1e-5       # external vertical field [T]
rho0   = 1000.0     # fluid density [kg/m^3]
C      = 3e16       # radial gravity constant [m^3/s^2]  (g_r = -C/r^2)

r1     = 9e7        # inner radius [m]
r2     = 1e8        # outer radius [m]
r0     = 0.5 * (r1 + r2)   # reference radius [m]

m      = 1          # azimuthal wavenumber (integer)

N      = 64         # number of Chebyshev collocation points (including endpoints)
# =============================================================================

# =============================================================================
# Derived dimensional parameters
# =============================================================================
VA2      = B0**2 / (MU0 * rho0)            # Alfven speed squared [m^2/s^2]
omA2     = VA2 / H0**2                      # Alfven frequency squared [s^-2]
beta_g   = 2.0 * C / r0**3                  # radial gravity gradient [s^-2]
beta_eff = Omega**2 + beta_g               # effective beta [s^-2]
c0sq     = g * H0                           # gravity wave speed squared [m^2/s^2]
f        = 2.0 * Omega                      # Coriolis parameter [s^-1]

# =============================================================================
# Derived dimensionless parameters
# =============================================================================
f_scale = 2.0 * Omega
hat_r1 = r1 / r0
hat_r2 = r2 / r0
hat_c0sq = c0sq / (Omega**2 * r0**2)
hat_VA2 = VA2 / (Omega**2 * r0**2)
# User derivation specifies gamma = C / (g * H0 * r0).
# Let's align with the user's explicit formulas for dimensionless numbers:
# hat_F0^2 = Omega^2 * r0^2 / (g * H0)
# gamma = C / (g * H0 * r0)
hat_F0sq = (Omega**2 * r0**2) / (g * H0)
gamma = C / (g * H0 * r0)
beta = (r0 / H0)**2
hat_omA = np.sqrt(VA2) / (f_scale * H0)
hat_omA2 = hat_omA**2

print("=" * 60)
print("SWMHD Global Dispersion Solver (Non-dimensionalised)")
print("=" * 60)
print(f"  r1={r1:.3f} m,  r2={r2:.3f} m,  r0={r0:.3f} m")
print(f"  Omega={Omega:.3f} rad/s,  H0={H0:.4f} m,  g={g:.3f} m/s^2")
print(f"  B0={B0:.4f} T,  rho0={rho0:.1f} kg/m^3")
print(f"  C={C:.4f} m^3/s^2")
print("\nDimensionless Parameters:")
print(f"  hat_r1    = {hat_r1:.4f}")
print(f"  hat_r2    = {hat_r2:.4f}")
print(f"  hat_c0^2  = {hat_c0sq:.4f}")
print(f"  hat_F0^2  = {hat_F0sq:.4f}")
print(f"  hat_V_A^2 = {hat_VA2:.4f}")
print(f"  gamma     = {gamma:.4f}")
print(f"  beta      = {beta:.4f}")
print(f"  hat_omA   = {hat_omA:.4f}")
print(f"  Azimuthal wavenumber m = {m}")
print(f"  Chebyshev N = {N}")
print()

# =============================================================================
# Chebyshev differentiation matrices on [hat_r1, hat_r2]
# =============================================================================
def chebyshev_lobatto(N, r_min, r_max):
    j  = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))   # xi[0]=1 (right), xi[N-1]=-1 (left)

    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    X = np.tile(xi, (N, 1))
    dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))

    scale = 2.0 / (r_max - r_min)
    D1 = scale * D
    D2 = D1 @ D1

    r_phys = 0.5 * (r_min + r_max) + 0.5 * (r_max - r_min) * xi
    r_phys = r_phys[::-1]
    D1     = D1[::-1, ::-1]
    D2     = D2[::-1, ::-1]

    return r_phys, D1, D2

hat_r_grid, D1, D2 = chebyshev_lobatto(N, hat_r1, hat_r2)

# =============================================================================
# ODE coefficient functions (dimensionless)
# =============================================================================
def omega_star_hat(hat_omega):
    """Dimensionless modified frequency hat_omega_*"""
    return hat_omega + (hat_VA2 * beta) / (4.0 * hat_omega)

def build_matrix(hat_omega):
    """
    Assemble the (N x N) collocation matrix L(hat_omega) such that
    L(hat_omega) @ eta_vec = 0 at interior points, with BC rows at
    indices 0 and N-1.
    """
    ws_hat = omega_star_hat(hat_omega)

    # C1(r) = 1/r + (F_0^2 - gamma) + (F_0^2 + 2*gamma)*(r - 1)
    C1 = 1.0 / hat_r_grid + (hat_F0sq - gamma) + (hat_F0sq + 2.0 * gamma) * (hat_r_grid - 1.0)

    # K(r) = 4*F_0^2 * [omega * (omega_*^2 - 1)] / omega_* - m^2/r^2 - m/(omega_* * r) * [ (F_0^2 - gamma) + (F_0^2 + 2*gamma)*(r - 1) ]
    term1 = 4.0 * hat_F0sq * (hat_omega * (ws_hat**2 - 1.0)) / ws_hat
    term2 = -(m**2) / (hat_r_grid**2)
    term3 = -(m / (ws_hat * hat_r_grid)) * ((hat_F0sq - gamma) + (hat_F0sq + 2.0 * gamma) * (hat_r_grid - 1.0))
    K_r = term1 + term2 + term3

    L = D2 + np.diag(C1) @ D1 + np.diag(K_r)

    # Boundary Conditions (vr = 0)
    # => hat_omega_star * d(eta)/dr - m/r * eta = 0
    for idx in [0, N - 1]:
        r_bc = hat_r_grid[idx]
        bc_eta_coeff = - m / r_bc
        L[idx, :] = ws_hat * D1[idx, :]
        L[idx, idx] += bc_eta_coeff

    return L

# =============================================================================
# Determinant function for root-finding
# =============================================================================
def det_L(hat_omega):
    mat = build_matrix(hat_omega)
    norm = np.max(np.abs(mat))
    if norm == 0:
        return 0.0
    # To maintain same log-scale for roots, we return det, but actually scan_and_polish expects raw det
    # which overflows. We will modify scan_and_polish too.
    return det(mat)

def log_abs_det_L(hat_omega):
    mat = build_matrix(hat_omega)
    norm = np.max(np.abs(mat))
    if norm == 0:
        return np.nan
    return np.log(np.abs(det(mat / norm)) + 1e-300) + N * np.log(norm)

# =============================================================================
# Core Solver Logic (Unchanged)
# =============================================================================
def scan_and_polish(m_val, omega_real_range, n_scan=400, tol=1e-10, max_iter=50):
    global m
    m = m_val # Fix missing global parameter update for function scanning
    omega_min, omega_max = omega_real_range
    omega_scan = np.linspace(omega_min, omega_max, n_scan)
    omega_scan = omega_scan[np.abs(omega_scan) > 1e-6 * abs(omega_max)]

    log_det = []
    for w in omega_scan:
        try:
            d = log_abs_det_L(w)
        except Exception:
            d = np.nan
        log_det.append(d)
    log_det = np.array(log_det)

    roots = []
    for k in range(1, len(log_det) - 1):
        if (log_det[k] < log_det[k - 1]) and (log_det[k] < log_det[k + 1]):
            w0 = omega_scan[k]
            w = complex(w0)
            for _ in range(max_iter):
                dw = 1e-6 * abs(w) + 1e-10
                # Scale determinant calculation to avoid overflow
                norm_w = np.max(np.abs(build_matrix(w)))
                if norm_w == 0:
                    break
                f0 = det(build_matrix(w) / norm_w)
                fp = (det(build_matrix(w + dw) / norm_w) - det(build_matrix(w - dw) / norm_w)) / (2 * dw)
                if abs(fp) < 1e-300:
                    break
                step = -f0 / fp
                w += step
                if abs(step) < tol * (abs(w) + 1e-10):
                    break
            is_dup = any(abs(w - wr) < 1e-4 * abs(w) + 1e-8 for wr in roots)
            if not is_dup and abs(w.imag) < 1e-5:
                roots.append(w)

    return np.array(roots), omega_scan, log_det

# =============================================================================
# Run the scan
# =============================================================================
# Dimensionless reference scales
hat_omega_inertial = 1.0  # 1.0
hat_omega_gravity = np.sqrt(hat_c0sq) / 2.0  # (c0/r0) / (2*Omega)
hat_omega_alfven = hat_omA

hat_omega_max_scan = 50.0  # Increased for new spectrum
hat_omega_min_scan = -50.0

print(f"Scan range: [{hat_omega_min_scan:.3f}, {hat_omega_max_scan:.3f}] (dimensionless)")
print(f"Reference frequencies (dimensionless):")
print(f"  Inertial      = {hat_omega_inertial:.4f}")
print(f"  Gravity scale = {hat_omega_gravity:.4f}")
print(f"  Alfven scale  = {hat_omega_alfven:.4f}")
print()

roots, omega_scan, log_det = scan_and_polish(
    m, (hat_omega_min_scan, hat_omega_max_scan), n_scan=600
)

print(f"Found {len(roots)} mode(s) for m={m}:")
print("-" * 55)
for k, wr in enumerate(sorted(roots, key=lambda x: x.real)):
    ws_hat = omega_star_hat(wr)
    print(f"  Mode {k+1}: hat_omega = {wr.real:+.6f}  {wr.imag:+.6f}i")
    print(f"            hat_omega_* = {ws_hat.real:.6f}")
print()

# =============================================================================
# Eigenfunction recovery
# =============================================================================
def get_eigenfunction(hat_omega):
    L = build_matrix(hat_omega)
    from scipy.linalg import svd
    _, s, Vh = svd(L)
    eta_vec = Vh[-1, :].conj()
    eta_vec /= np.max(np.abs(eta_vec))
    return eta_vec

# =============================================================================
# Plots
# =============================================================================
fig_det, ax_det = plt.subplots(figsize=(10, 4))
valid = np.isfinite(log_det)
ax_det.plot(omega_scan[valid], log_det[valid], 'b-', lw=1.2,
            label=r'$\ln|\det \mathcal{L}(\hat{\omega})|$')
for wr in roots:
    ax_det.axvline(wr.real, color='r', ls='--', lw=0.8)
ax_det.set_xlabel(r'$\hat{\omega}$', fontsize=13)
ax_det.set_ylabel(r'$\ln|\det \mathcal{L}|$', fontsize=13)
ax_det.set_title(f'Global SWMHD determinant scan (dimensionless, m={m})', fontsize=13)
ax_det.legend()
ax_det.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('swmhd_det_scan.pdf', dpi=150)
print("Saved: swmhd_det_scan.pdf")

if len(roots) > 0:
    ncols = min(3, len(roots))
    nrows = (len(roots) + ncols - 1) // ncols
    fig_ef, axes = plt.subplots(nrows, ncols,
                                figsize=(5 * ncols, 4 * nrows),
                                squeeze=False)
    axes_flat = axes.flatten()

    for k, wr in enumerate(sorted(roots, key=lambda x: x.real)):
        eta_vec = get_eigenfunction(wr)
        ax = axes_flat[k]
        ax.plot(hat_r_grid, eta_vec.real, 'b-', label=r'Re($\tilde\eta$)', lw=1.5)
        ax.plot(hat_r_grid, eta_vec.imag, 'r--', label=r'Im($\tilde\eta$)', lw=1.5)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(1.0, color='grey', ls=':', lw=0.8, label=r'$\hat{r}=1$')
        ax.set_xlabel(r'$\hat{r}$', fontsize=11)
        ax.set_ylabel(r'$\tilde\eta(\hat{r})$ [normalised]', fontsize=11)
        title_str = (f'Mode {k+1}: ' r'$\hat{\omega}$' + f'={wr.real:+.4f}{wr.imag:+.4f}i')
        ax.set_title(title_str, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ws_hat = omega_star_hat(wr)
        for r_bc, label in [(hat_r1, 'r1'), (hat_r2, 'r2')]:
            idx_bc = 0 if r_bc == hat_r1 else -1
            eta_bc  = eta_vec[idx_bc]
            deta_bc = D1[idx_bc, :] @ eta_vec
            vr_bc   = ws_hat * deta_bc - (m / r_bc) * eta_bc
            print(f"  Mode {k+1}: |V_r BC residual at {label}| = {abs(vr_bc):.2e}")

    for ax in axes_flat[len(roots):]:
        ax.set_visible(False)

    plt.suptitle(f'SWMHD eigenfunctions (dimensionless, m={m})', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('swmhd_eigenfunctions.pdf', dpi=150, bbox_inches='tight')
    print("Saved: swmhd_eigenfunctions.pdf")
else:
    print("No roots found. Try widening scan range or increasing N.")

# =============================================================================
# Dispersion branches: sweep over m values
# =============================================================================
print()
print("Computing dispersion branches over m = 0..5 ...")
m_values = list(range(0, 10))
branch_data = {}

for m_val in m_values:
    m = m_val
    try:
        rts, _, _ = scan_and_polish(m_val, (hat_omega_min_scan, hat_omega_max_scan),
                                    n_scan=400)
        branch_data[m_val] = sorted([wr.real for wr in rts])
    except Exception as e:
        branch_data[m_val] = []
        print(f"  m={m_val}: failed ({e})")

m = 1   # restore

fig_disp, ax_disp = plt.subplots(figsize=(9, 6))
markers = ['o', 's', '^', 'D', 'v', 'P']
for idx, m_val in enumerate(m_values):
    omegas = branch_data[m_val]
    ax_disp.scatter([m_val] * len(omegas), omegas, marker=markers[idx % len(markers)],s=60, label=f'm={m_val}')
ax_disp.axhline(0, color='k', lw=0.5, ls=':')
ax_disp.set_ylim(-55, 55)
ax_disp.set_xlabel('Azimuthal wavenumber $m$', fontsize=13)
ax_disp.set_ylabel(r'$\hat{\omega}$', fontsize=13)
ax_disp.set_title(r'Global SWMHD dispersion relation' '\n' r'(dimensionless, $\hat{\omega}$ vs $m$)', fontsize=13)
ax_disp.legend(fontsize=10)
ax_disp.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('swmhd_dispersion_branches.pdf', dpi=150)
print("Saved: swmhd_dispersion_branches.pdf")

print("\nDone.")
