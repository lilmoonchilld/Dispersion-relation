"""
Global Dispersion Relation Solver for Shallow Water MHD
with Radial Gravity (KP2017 framework, rotating annular tank)

Physical setup
--------------
- Annular tank: inner radius r1, outer radius r2
- Rotation rate Omega, fluid depth H0 at reference radius r0
- External vertical field B0, density rho0
- Radial gravity: g_r(r) = -C/r^2  (C > 0)
- Azimuthal wavenumber m (integer)

Wave ODE (from your document, Section 6)
-----------------------------------------
eta'' + P(r)*eta' + Q(r)*eta = 0

where (defining D* = 4*Omega^2 - omega_*^2):

  P(r) = 1/r - beta_eff*(r - r0) / (g*H0)

  Q(r) = - omega*(4*Omega^2 - omega_*^2) / (omega_* * g * H0)
          - m^2/r^2
          - beta_eff*(2*r - r0) / (g*H0*r)
          - 2*Omega*m*beta_eff*(r - r0) / (omega_* * g*H0*r)

with  omega_* = omega + omega_A^2/omega
      omega_A^2 = B0^2 / (mu0*rho0*H0^2)
      beta_eff  = Omega^2 + 2*C/r0^3

Boundary conditions: V_r = 0 at r = r1 and r = r2.
From Cramer's rule (your eq. for v_r):

  omega_* * g*H0 * eta' - [omega_* * beta_eff*(r-r0) + 2*Omega*m*g*H0/r] * eta = 0

NOTE ON DIMENSIONAL ISSUE
--------------------------
In your r-momentum equation the term  -beta_eff*(r-r0)*eta  has
dimensions [T^-2 * L * L] = [L^2 T^-2]  only if beta_eff carries
[T^-2] and (r-r0)*eta carries [L^2].  That is self-consistent IF
the equation has been divided by H0 from the standard form.  The
code implements it as written in Section 6 of your document.
If a factor of H0 is missing, replace  beta_eff  ->  beta_eff/H0
in Q(r) and in the boundary-condition coefficients.

Method: Chebyshev collocation on N interior points
---------------------------------------------------
The ODE is discretised on the Chebyshev-Lobatto grid mapped to [r1,r2].
The boundary conditions replace the first and last rows of the system.
The resulting generalised eigenvalue problem is

        A(omega) x = 0,

where omega enters nonlinearly through omega_*. We therefore solve
a companion linearisation: treat omega_* as an intermediate quantity,
scan over real omega on a grid, and find zeros of det(A(omega)).
Alternatively, for fixed m we do a 2D scan (Re(omega), Im(omega)) --
that is expensive; instead we use a Newton-based contour argument on the
real axis first (valid for neutral modes), then allow Im(omega) != 0.

For practical use: the solver below

  1. Assembles the collocation matrix for a given omega (complex).
  2. Provides a determinant function det_A(omega) for root-finding.
  3. Scans a real-omega grid to bracket sign changes of |det_A|.
  4. Polishes each root with scipy.optimize.
  5. Plots the eigenfunction eta(r) for each found mode.

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.linalg import det, eig
import matplotlib.pyplot as plt
from itertools import product as iproduct

# =============================================================================
# Physical parameters  (SI units throughout)
# =============================================================================
MU0   = 4 * np.pi * 1e-7   # permeability of free space [H/m]

# --- Adjustable parameters ---------------------------------------------------
Omega  = 1.0        # rotation rate [rad/s]
H0     = 0.05       # equilibrium depth at r0 [m]
g      = 9.81       # vertical gravity [m/s^2]
B0     = 0.01       # external vertical field [T]
rho0   = 1000.0     # fluid density [kg/m^3]
C      = 0.5        # radial gravity constant [m^3/s^2]  (g_r = -C/r^2)

r1     = 0.3        # inner radius [m]
r2     = 0.8        # outer radius [m]
r0     = 0.5 * (r1 + r2)   # reference radius [m]

m      = 1          # azimuthal wavenumber (integer)

N      = 64         # number of Chebyshev collocation points (including endpoints)
# =============================================================================


# =============================================================================
# Derived parameters
# =============================================================================
VA2      = B0**2 / (MU0 * rho0)            # Alfven speed squared [m^2/s^2]
omA2     = VA2 / H0**2                      # Alfven frequency squared [s^-2]
beta_g   = 2.0 * C / r0**3                  # radial gravity gradient [s^-2]
beta_eff = Omega**2 + beta_g               # effective beta [s^-2]
c0sq     = g * H0                           # gravity wave speed squared [m^2/s^2]
f        = 2.0 * Omega                      # Coriolis parameter [s^-1]

print("=" * 60)
print("SWMHD Global Dispersion Solver")
print("=" * 60)
print(f"  r1={r1:.3f} m,  r2={r2:.3f} m,  r0={r0:.3f} m")
print(f"  Omega={Omega:.3f} rad/s,  H0={H0:.4f} m,  g={g:.3f} m/s^2")
print(f"  B0={B0:.4f} T,  rho0={rho0:.1f} kg/m^3")
print(f"  C={C:.4f} m^3/s^2  =>  g_r(r0)={-C/r0**2:.4f} m/s^2")
print(f"  beta_eff = {beta_eff:.4f} s^-2")
print(f"  omega_A  = {np.sqrt(omA2):.4f} rad/s")
print(f"  c0       = {np.sqrt(c0sq):.4f} m/s")
print(f"  Azimuthal wavenumber m = {m}")
print(f"  Chebyshev N = {N}")
print()


# =============================================================================
# Chebyshev differentiation matrices on [r1, r2]
# =============================================================================
def chebyshev_lobatto(N, r1, r2):
    """
    Return Chebyshev-Lobatto nodes and first/second derivative matrices
    mapped from [-1, 1] to [r1, r2].

    Returns
    -------
    r  : (N,) array of collocation points, r[0]=r1, r[N-1]=r2
    D1 : (N, N) first derivative matrix d/dr
    D2 : (N, N) second derivative matrix d^2/dr^2
    """
    # Nodes on [-1,1]: xi_j = cos(j*pi/(N-1)), j=0..N-1
    j  = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))   # xi[0]=1 (right), xi[N-1]=-1 (left)

    # Build D1 on [-1,1] using the standard Chebyshev formula
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
    D -= np.diag(D.sum(axis=1))   # diagonal: sum condition

    # Map from [-1,1] to [r1,r2]
    # r = (r1+r2)/2 + (r2-r1)/2 * xi  =>  d/dr = (2/(r2-r1)) * d/dxi
    scale = 2.0 / (r2 - r1)
    D1 = scale * D
    D2 = D1 @ D1

    # Physical nodes (ascending: r[0]=r1, r[N-1]=r2 after flip)
    r_phys = 0.5 * (r1 + r2) + 0.5 * (r2 - r1) * xi
    # xi[0]=1 => r_phys[0] = r2;  xi[N-1]=-1 => r_phys[N-1] = r1
    # Flip so r is ascending
    r_phys = r_phys[::-1]
    D1     = D1[::-1, ::-1]
    D2     = D2[::-1, ::-1]

    return r_phys, D1, D2


r_grid, D1, D2 = chebyshev_lobatto(N, r1, r2)
# r_grid[0] = r1 (inner wall),  r_grid[-1] = r2 (outer wall)


# =============================================================================
# ODE coefficient functions
# =============================================================================
def omega_star(omega):
    """Modified frequency omega_* = omega + omega_A^2 / omega."""
    return omega + omA2 / omega


def P_coeff(r):
    """Coefficient of eta' in the wave ODE."""
    return 1.0 / r - beta_eff * (r - r0) / c0sq


def Q_coeff(r, omega):
    """Coefficient of eta in the wave ODE."""
    ws = omega_star(omega)
    Ds = 4.0 * Omega**2 - ws**2
    term1 = -omega * Ds / (ws * c0sq)
    term2 = -m**2 / r**2
    term3 = -beta_eff * (2.0 * r - r0) / (c0sq * r)
    term4 = -2.0 * Omega * m * beta_eff * (r - r0) / (ws * c0sq * r)
    return term1 + term2 + term3 + term4


# =============================================================================
# Build the collocation matrix for a given omega
# =============================================================================
def build_matrix(omega):
    """
    Assemble the (N x N) collocation matrix L(omega) such that
    L(omega) @ eta_vec = 0 at interior points, with BC rows at
    indices 0 and N-1.

    The ODE is:   eta'' + P(r)*eta' + Q(r)*eta = 0
    => [D2 + diag(P)*D1 + diag(Q)] @ eta = 0   (interior rows)

    Boundary condition (V_r = 0):
      omega_* * c0sq * eta'
      - [omega_* * beta_eff*(r-r0) + 2*Omega*m*c0sq/r] * eta = 0
    => (omega_* * c0sq) * D1_row + diag_coeff * I_row   applied at r1, r2
    """
    ws = omega_star(omega)
    P_vec = P_coeff(r_grid)
    Q_vec = Q_coeff(r_grid, omega)

    # Interior ODE rows
    L = D2 + np.diag(P_vec) @ D1 + np.diag(Q_vec)

    # Boundary condition at r = r1 (row 0) and r = r2 (row N-1)
    for idx in [0, N - 1]:
        r_bc = r_grid[idx]
        bc_eta_coeff = -(ws * beta_eff * (r_bc - r0)
                         + 2.0 * Omega * m * c0sq / r_bc)
        L[idx, :] = ws * c0sq * D1[idx, :] + bc_eta_coeff * np.eye(N)[idx]

    return L


# =============================================================================
# Determinant function for root-finding
# =============================================================================
def det_L(omega):
    """Return det of collocation matrix (complex scalar)."""
    return det(build_matrix(omega))


# =============================================================================
# Eigenvalue approach: treat as generalised eigenvalue problem
# by linearising in omega_* (valid when omA2 / omega^2 << 1 OR
# when we want all branches simultaneously)
#
# Rewrite ODE in first-order form:
#   [eta', eta]^T  =>  companion system  A*u = omega*B*u
# This is nonlinear in omega due to omega_*; we use the scan-then-polish method.
# =============================================================================

def scan_and_polish(m_val, omega_real_range, n_scan=400,
                    tol=1e-10, max_iter=50):
    """
    Scan real omega axis for sign changes of |det L|, then polish
    each root with Newton's method (allowing complex correction).

    Parameters
    ----------
    omega_real_range : (omega_min, omega_max)
    n_scan           : number of scan points
    """
    omega_min, omega_max = omega_real_range
    omega_scan = np.linspace(omega_min, omega_max, n_scan)
    # Avoid omega=0 (singularity in omega_*)
    omega_scan = omega_scan[np.abs(omega_scan) > 1e-6 * abs(omega_max)]

    log_det = []
    for w in omega_scan:
        try:
            d = np.log(np.abs(det_L(w)) + 1e-300)
        except Exception:
            d = np.nan
        log_det.append(d)
    log_det = np.array(log_det)

    # Find local minima of |log det| as candidate roots
    roots = []
    for k in range(1, len(log_det) - 1):
        if (log_det[k] < log_det[k - 1]) and (log_det[k] < log_det[k + 1]):
            w0 = omega_scan[k]
            # Newton polish
            w = complex(w0)
            for _ in range(max_iter):
                dw = 1e-6 * abs(w) + 1e-10
                f0 = det_L(w)
                fp = (det_L(w + dw) - det_L(w - dw)) / (2 * dw)
                if abs(fp) < 1e-300:
                    break
                step = -f0 / fp
                w += step
                if abs(step) < tol * (abs(w) + 1e-10):
                    break
            if abs(det_L(w)) < 1e-6 * np.exp(np.nanmax(log_det[np.isfinite(log_det)])):
                # Check not a duplicate
                is_dup = any(abs(w - wr) < 1e-4 * abs(w) + 1e-8 for wr in roots)
                if not is_dup:
                    roots.append(w)

    return np.array(roots), omega_scan, log_det


# =============================================================================
# Run the scan
# =============================================================================
# Estimate natural scales for the scan range
omega_inertial = f                   # 2*Omega
omega_gravity  = np.sqrt(c0sq) / r0  # c0 / r0
omega_alfven   = np.sqrt(omA2)

omega_max_scan = 3.0 * max(omega_inertial, omega_gravity, omega_alfven)
omega_min_scan = -omega_max_scan

print(f"Scan range: [{omega_min_scan:.3f}, {omega_max_scan:.3f}] rad/s")
print(f"Reference frequencies:")
print(f"  2*Omega (inertial) = {omega_inertial:.4f} rad/s")
print(f"  c0/r0  (gravity)   = {omega_gravity:.4f} rad/s")
print(f"  omega_A (Alfven)   = {omega_alfven:.4f} rad/s")
print()

roots, omega_scan, log_det = scan_and_polish(
    m, (omega_min_scan, omega_max_scan), n_scan=600
)

print(f"Found {len(roots)} mode(s) for m={m}:")
print("-" * 55)
for k, wr in enumerate(sorted(roots, key=lambda x: x.real)):
    ws = omega_star(wr)
    print(f"  Mode {k+1}: omega = {wr.real:+.6f}  {wr.imag:+.6f}i  rad/s")
    print(f"            omega_* = {ws.real:.6f}")
    print(f"            Period  = {2*np.pi/abs(wr.real):.4f} s  "
          f"(if Re(omega) != 0)")
print()


# =============================================================================
# Eigenfunction recovery
# =============================================================================
def get_eigenfunction(omega):
    """Return normalised eta(r) for a given eigenfrequency."""
    L = build_matrix(omega)
    # SVD: smallest singular value -> eigenvector
    from scipy.linalg import svd
    _, s, Vh = svd(L)
    eta_vec = Vh[-1, :].conj()  # right singular vector for smallest s
    # Normalise: max |eta| = 1
    eta_vec /= np.max(np.abs(eta_vec))
    return eta_vec


# =============================================================================
# Plots
# =============================================================================
fig_det, ax_det = plt.subplots(figsize=(10, 4))
valid = np.isfinite(log_det)
ax_det.plot(omega_scan[valid], log_det[valid], 'b-', lw=1.2,
            label=r'$\ln|\det \mathcal{L}(\omega)|$')
for wr in roots:
    ax_det.axvline(wr.real, color='r', ls='--', lw=0.8)
ax_det.set_xlabel(r'$\omega$ [rad/s]', fontsize=13)
ax_det.set_ylabel(r'$\ln|\det \mathcal{L}|$', fontsize=13)
ax_det.set_title(f'Global SWMHD determinant scan  (m={m})', fontsize=13)
ax_det.legend()
ax_det.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/swmhd_det_scan.pdf', dpi=150)
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
        ax.plot(r_grid, eta_vec.real, 'b-', label=r'Re($\tilde\eta$)', lw=1.5)
        ax.plot(r_grid, eta_vec.imag, 'r--', label=r'Im($\tilde\eta$)', lw=1.5)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(r0, color='grey', ls=':', lw=0.8, label=r'$r_0$')
        ax.set_xlabel('r [m]', fontsize=11)
        ax.set_ylabel(r'$\tilde\eta(r)$ [normalised]', fontsize=11)
        title_str = (f'Mode {k+1}: '
                     r'$\omega$' + f'={wr.real:+.4f}{wr.imag:+.4f}i rad/s')
        ax.set_title(title_str, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # Verify BC: V_r should vanish at walls
        ws = omega_star(wr)
        for r_bc, label in [(r1, 'r1'), (r2, 'r2')]:
            idx_bc = 0 if r_bc == r1 else -1
            eta_bc  = eta_vec[idx_bc]
            deta_bc = D1[idx_bc, :] @ eta_vec
            vr_bc   = (ws * c0sq * deta_bc
                       - (ws * beta_eff * (r_bc - r0)
                          + 2 * Omega * m * c0sq / r_bc) * eta_bc)
            print(f"  Mode {k+1}: |V_r BC residual at {label}| = {abs(vr_bc):.2e}")

    # Hide unused axes
    for ax in axes_flat[len(roots):]:
        ax.set_visible(False)

    plt.suptitle(f'SWMHD eigenfunctions  (m={m})', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('outputs/swmhd_eigenfunctions.pdf',
                dpi=150, bbox_inches='tight')
    print("Saved: swmhd_eigenfunctions.pdf")
else:
    print("No roots found. Try widening scan range or increasing N.")

# =============================================================================
# Dispersion branches: sweep over m values
# =============================================================================
print()
print("Computing dispersion branches over m = 0..5 ...")
m_values = list(range(0, 6))
branch_data = {}   # m -> list of real(omega)

for m_val in m_values:
    m = m_val   # update global used in Q_coeff etc.
    try:
        rts, _, _ = scan_and_polish(m_val, (omega_min_scan, omega_max_scan),
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
    ax_disp.scatter([m_val] * len(omegas), omegas,
                    marker=markers[idx % len(markers)],
                    s=60, label=f'm={m_val}')
ax_disp.axhline(0, color='k', lw=0.5, ls=':')
ax_disp.set_xlabel('Azimuthal wavenumber m', fontsize=13)
ax_disp.set_ylabel(r'$\omega$ [rad/s]', fontsize=13)
ax_disp.set_title('Global SWMHD dispersion relation\n'
                  r'(eigenfrequencies vs $m$)', fontsize=13)
ax_disp.legend(fontsize=10)
ax_disp.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/swmhd_dispersion_branches.pdf',
            dpi=150)
print("Saved: swmhd_dispersion_branches.pdf")

# plt.show()
print("\nDone.")
