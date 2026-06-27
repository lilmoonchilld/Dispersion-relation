import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# =============================================================================
# Physical parameters (SI units)
# =============================================================================
MU0   = 4 * np.pi * 1e-7
Omega = 1.0
H0    = 0.05
g     = 9.81
B0    = 0.01
rho0  = 1000.0
C     = 0.5

r1    = 0.3
r2    = 0.8
r0    = 0.5 * (r1 + r2)
m     = 1

N     = 64  # Collocation points

# =============================================================================
# Derived parameters
# =============================================================================
beta_g   = 2.0 * C / r0**3
beta_eff = Omega**2 + beta_g
B_factor = B0 / (MU0 * rho0)

# =============================================================================
# Chebyshev differentiation
# =============================================================================
def chebyshev_lobatto(N, r1, r2):
    j  = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N)
    c[0] = 2.0; c[-1] = 2.0
    X = np.tile(xi, (N, 1))
    dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    scale = 2.0 / (r2 - r1)
    D1 = scale * D
    r_phys = 0.5 * (r1 + r2) + 0.5 * (r2 - r1) * xi
    # Flip to make ascending
    return r_phys[::-1], D1[::-1, ::-1]

# =============================================================================
# Build the 5N x 5N GEVP: A x = omega B x
# x = [eta, v_r, v_theta, B_r, B_theta]^T
#
# Original Equations (with d/dt -> -i omega):
# 1) -i w eta + H0 [ d v_r/dr + v_r/r + i m / r v_theta ] = 0
#    => w eta = -i H0 (D1 + R_inv) v_r + m H0 R_inv v_theta
#
# 2) -i w H0 v_r + g H0 d eta / dr - 2 Omega H0 v_theta - beta_eff(r-r0) eta + B_factor B_r = 0
#    => w H0 v_r = -i g H0 D1 eta + i 2 Omega H0 v_theta + i beta_eff diag(r-r0) eta - i B_factor B_r
#
# 3) -i w H0 v_theta + i m g H0 / r eta + 2 Omega H0 v_r + B_factor B_theta = 0
#    => w H0 v_theta = m g H0 R_inv eta - i 2 Omega H0 v_r - i B_factor B_theta
#
# 4) -i w H0 B_r + B0 v_r = 0
#    => w H0 B_r = -i B0 v_r
#
# 5) -i w H0 B_theta + B0 v_theta = 0
#    => w H0 B_theta = -i B0 v_theta
#
# Therefore B is diagonal with blocks [I, H0*I, H0*I, H0*I, H0*I]
# =============================================================================
def build_matrices(N_pts):
    r_grid, D1 = chebyshev_lobatto(N_pts, r1, r2)
    R_inv = np.diag(1.0 / r_grid)

    A = np.zeros((5*N_pts, 5*N_pts), dtype=complex)
    B = np.zeros((5*N_pts, 5*N_pts), dtype=complex)
    I = np.eye(N_pts)

    idx_eta = slice(0, N_pts)
    idx_vr  = slice(N_pts, 2*N_pts)
    idx_vth = slice(2*N_pts, 3*N_pts)
    idx_Br  = slice(3*N_pts, 4*N_pts)
    idx_Bth = slice(4*N_pts, 5*N_pts)

    # Eq 1: Continuity
    A[idx_eta, idx_vr]  = -1j * H0 * (D1 + R_inv)
    A[idx_eta, idx_vth] = m * H0 * R_inv
    B[idx_eta, idx_eta] = I

    # Eq 2: r-momentum
    A[idx_vr, idx_eta] = -1j * g * H0 * D1 + 1j * beta_eff * np.diag(r_grid - r0)
    A[idx_vr, idx_vth] = 1j * 2 * Omega * H0 * I
    A[idx_vr, idx_Br]  = -1j * B_factor * I
    B[idx_vr, idx_vr]  = H0 * I

    # Eq 3: theta-momentum
    A[idx_vth, idx_eta] = m * g * H0 * R_inv
    A[idx_vth, idx_vr]  = -1j * 2 * Omega * H0 * I
    A[idx_vth, idx_Bth] = -1j * B_factor * I
    B[idx_vth, idx_vth] = H0 * I

    # Eq 4: r-induction
    A[idx_Br, idx_vr] = -1j * B0 * I
    B[idx_Br, idx_Br] = H0 * I

    # Eq 5: theta-induction
    A[idx_Bth, idx_vth] = -1j * B0 * I
    B[idx_Bth, idx_Bth] = H0 * I

    # Boundary Conditions: v_r(r1) = 0 and v_r(r2) = 0
    # Overwrite the respective rows for the boundary conditions in the momentum block.
    # Note: Setting B[row,:] = 0 enforces 0 = A[row,:] @ X.
    A[N_pts, :] = 0
    A[N_pts, N_pts] = 1
    B[N_pts, :] = 0

    A[2*N_pts-1, :] = 0
    A[2*N_pts-1, 2*N_pts-1] = 1
    B[2*N_pts-1, :] = 0

    return A, B

if __name__ == '__main__':
    # =============================================================================
    # Solve GEVP with spectral drift filtering
    # =============================================================================
    print("Solving massive 5N x 5N GEVP with drift filtering...")
    A1, B1 = build_matrices(N)
    vals1, vecs1 = eig(A1, B1)
    vals1 = vals1[np.isfinite(vals1)]

    A2, B2 = build_matrices(N + 2)
    vals2, vecs2 = eig(A2, B2)
    vals2 = vals2[np.isfinite(vals2)]

    tol = 1e-6
    matched_vals = []
    for v1 in vals1:
        # Ignore highly singular modes (w=0 null space)
        if np.abs(v1) < 1e-4:
            continue
        diffs = np.abs(vals2 - v1)
        if len(vals2) > 0 and np.min(diffs) / max(np.abs(v1), 1e-10) < tol:
            matched_vals.append(v1)

    matched_vals = np.array(matched_vals)

    # Sort by real part
    sort_idx = np.argsort(matched_vals.real)
    matched_vals = matched_vals[sort_idx]

    print(f"Found {len(matched_vals)} valid, physically converged eigenvalues.")

    # Print the slow spectrum (e.g. |Re(omega)| < 20)
    slow_vals = matched_vals[np.abs(matched_vals.real) < 20.0]
    print("\nSample slow modes (|Re(omega)| < 20):")
    for v in slow_vals:
        if np.abs(v.imag) < 1e-5:
            print(f"  {v.real:+.6f} rad/s")

    # Plot full spectrum
    plt.figure(figsize=(8, 6))
    plt.scatter(matched_vals.real, matched_vals.imag, s=15, alpha=0.6, color='blue', edgecolors='k')
    plt.axhline(0, color='red', ls='--', lw=1)
    plt.axvline(0, color='red', ls='--', lw=1)
    plt.xlabel(r'Re($\omega$) [rad/s]', fontsize=13)
    plt.ylabel(r'Im($\omega$) [rad/s]', fontsize=13)
    plt.title(f'Global SWMHD Spectrum via 5N x 5N GEVP (m={m}, C={C})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(-25, 25)
    plt.ylim(-5, 5)
    plt.tight_layout()
    plt.savefig('outputs/spectrum_gevp.pdf')
    print("\nSaved spectrum plot to outputs/spectrum_gevp.pdf")
