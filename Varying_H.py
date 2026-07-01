import numpy as np
import matplotlib.pyplot as plt

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

# ── Dimensionless Parameters ────────────────────────────────────────────────
hat_r1 = r1 / r0
hat_r2 = r2 / r0
hat_c0sq = g * H0 / (Omega**2 * r0**2)
hat_VA2 = VA2 / (Omega**2 * r0**2)
gamma = C / (g * H0 * r0)
beta = (r0 / H0)**2

print("=" * 62)
print("  System Parameters (Dimensionless)")
print("=" * 62)
print(f"  hat_c0^2 : {hat_c0sq:.4f}")
print(f"  hat_VA^2 : {hat_VA2:.4e}")
print(f"  gamma    : {gamma:.4f}")
print(f"  beta     : {beta:.4e}")
print("-" * 62)
print(f"  Domain: hat_r in [{hat_r1:.3f}, {hat_r2:.3f}]")
print()

def chebyshev_lobatto(N, r_min, r_max):
    j = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N)
    c[0] = 2; c[-1] = 2
    X = np.tile(xi, (N, 1))
    dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    sc = 2.0 / (r_max - r_min)
    D1 = sc * D
    rp = 0.5 * (r_min + r_max) + 0.5 * (r_max - r_min) * xi
    # flip so that r is increasing
    rp = rp[::-1]
    D1 = D1[::-1, ::-1]
    return rp, D1

def solve_gevp(N, m_val):
    rp, D1 = chebyshev_lobatto(N, hat_r1, hat_r2)

    H = 1.0 + (1.0/hat_c0sq - gamma) * (rp - 1.0)
    dHdr = (1.0/hat_c0sq - gamma) * np.ones_like(rp)

    A = np.zeros((5*N, 5*N), dtype=complex)

    I = np.eye(N)
    Z = np.zeros((N, N))

    idx_eta = slice(0, N)
    idx_vr  = slice(N, 2*N)
    idx_vt  = slice(2*N, 3*N)
    idx_Br  = slice(3*N, 4*N)
    idx_Bt  = slice(4*N, 5*N)

    # 1. Continuity:
    A[idx_eta, idx_eta] = Z
    A[idx_eta, idx_vr]  = -1j * np.diag(H) @ D1 - 1j * np.diag(H / rp + dHdr)
    A[idx_eta, idx_vt]  = np.diag(m_val * H / rp)
    A[idx_eta, idx_Br]  = Z
    A[idx_eta, idx_Bt]  = Z

    # 2. Radial Momentum:
    A[idx_vr, idx_eta] = -1j * hat_c0sq * D1
    A[idx_vr, idx_vr]  = Z
    A[idx_vr, idx_vt]  = 2j * I
    A[idx_vr, idx_Br]  = 1j * np.diag(hat_VA2 * beta / H)
    A[idx_vr, idx_Bt]  = Z

    # 3. Azimuthal Momentum:
    A[idx_vt, idx_eta] = np.diag(m_val * hat_c0sq / rp)
    A[idx_vt, idx_vr]  = -2j * I
    A[idx_vt, idx_vt]  = Z
    A[idx_vt, idx_Br]  = Z
    A[idx_vt, idx_Bt]  = 1j * np.diag(hat_VA2 * beta / H)

    # 4. Radial Induction:
    A[idx_Br, idx_eta] = Z
    A[idx_Br, idx_vr]  = 1j * np.diag(np.sqrt(beta) / H)
    A[idx_Br, idx_vt]  = Z
    A[idx_Br, idx_Br]  = Z
    A[idx_Br, idx_Bt]  = Z

    # 5. Azimuthal Induction:
    A[idx_Bt, idx_eta] = Z
    A[idx_Bt, idx_vr]  = Z
    A[idx_Bt, idx_vt]  = 1j * np.diag(np.sqrt(beta) / H)
    A[idx_Bt, idx_Br]  = Z
    A[idx_Bt, idx_Bt]  = Z

    # Boundary Conditions: vr = 0 at r1 and r2
    A[N, :] = 0
    A[N, N] = -1000j
    A[2*N-1, :] = 0
    A[2*N-1, 2*N-1] = -1000j

    lam = np.linalg.eigvals(A)
    omega_hat = lam / 2.0
    return omega_hat

def get_valid_eigenvalues(m_val, N_base=64, tol=1e-5):
    om_N = solve_gevp(N_base, m_val)
    om_N2 = solve_gevp(N_base + 2, m_val)

    # Keep only purely real eigenvalues
    om_N = om_N[np.abs(om_N.imag) < 1e-4]
    om_N2 = om_N2[np.abs(om_N2.imag) < 1e-4]

    valid_om = []
    for w1 in om_N:
        if len(om_N2) == 0:
            continue
        diff = np.abs(om_N2 - w1)
        if np.min(diff) < tol:
            valid_om.append(w1.real)

    if len(valid_om) > 0:
        valid_om = np.unique(np.round(valid_om, 5))

    return np.array(valid_om)

M_max = 30
m_arr = np.arange(1, M_max + 1)

print("Running Block GEVP spectral collocation for m = 1 …", M_max, "…")
col_eigs = {}
for m_val in m_arr:
    eigs = get_valid_eigenvalues(m_val)
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

print("\nGenerating plot...")

plt.figure(figsize=(10, 6))

for m_val in m_arr:
    eigs_n = col_eigs[m_val]
    # filter out very large non-physical eigenvalues outside the [-1.5, 1.5] range for plot clarity
    eigs_n = eigs_n[np.abs(eigs_n) <= 1.5]
    if len(eigs_n) > 0:
        plt.scatter([m_val] * len(eigs_n), eigs_n,
                    s=20, color='k', zorder=5, alpha=0.7)

plt.axhline(0, color='grey', lw=0.7, ls=':')
plt.axhline(+1, color='grey', lw=0.6, ls='--', alpha=0.5)
plt.axhline(-1, color='grey', lw=0.6, ls='--', alpha=0.5)

plt.xlabel('Azimuthal wavenumber  $m$', fontsize=13)
plt.ylabel(r'Normalised frequency  $\hat{\omega}$', fontsize=13)
plt.title(r'SWMHD Global Eigenvalues' '\n'
          r'($\hat{\omega}$ vs $m$, varying $H(r)$)', fontsize=12)
plt.xlim(0.5, M_max + 0.5)
plt.ylim(-1.5, 1.5)
plt.xticks(np.arange(0, M_max + 1, 2))
plt.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('Varying_H.png', dpi=180, bbox_inches='tight')
print("Saved plot to Varying_H.png")
