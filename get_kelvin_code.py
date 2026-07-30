print('''import numpy as np
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
rho0  = 1000.0        # kg/m^3

# Parameters modified to ensure visible magnetic cutoff and valid C
B0    = 0.0005        # T
C     = 1.5e9         # m^3/s^2

f        = 2.0 * Omega
c0sq     = g * H0
c0       = np.sqrt(c0sq)
Rd       = c0 / f

VA2      = B0**2 / (MU0 * rho0)
omA2     = VA2 / H0**2
omA      = np.sqrt(omA2)

hat_omA = omA / f
hat_omA2 = hat_omA**2

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

def omega_star_hat(hat_omega):
    return hat_omega + hat_omA2 / hat_omega

def build_matrix(hat_omega, m_val, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq):
    ws_hat = omega_star_hat(hat_omega)
    Pv = 1.0 / hat_r_grid - ( 1+gamma) * (hat_r_grid - 1.0) / hat_c0sq
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

def find_eigenvalues(m_val, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq, omega_range=(-3.5, 3.5), n_scan=800):
    scan = np.linspace(*omega_range, n_scan)
    scan = scan[np.abs(scan) > 0.01]
    ld = []
    for w in scan:
        try:
            d = np.log(np.abs(det(build_matrix(w, m_val, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq))) + 1e-300)
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
                    f0 = det(build_matrix(w, m_val, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq))
                    fp = ((det(build_matrix(w+dw, m_val, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq))
                           - det(build_matrix(w-dw, m_val, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq)))
                          / (2*dw))
                    if abs(fp) < 1e-300: break
                    step = -f0 / fp; w += step
                    if abs(step) < 1e-10 * (abs(w) + 1): break
                except Exception:
                    break
            try:
                residual = abs(det(build_matrix(w, m_val, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq)))
                valid_ld = [x for x in ld if np.isfinite(x)]
                ld_max = max(valid_ld) if len(valid_ld) > 0 else 0
                if residual < 1e-3 * np.exp(ld_max):
                    dup = any(abs(w - wr) < 5e-3 for wr in roots)
                    if not dup:
                        roots.append(w.real)
            except Exception:
                pass
    return np.array(sorted(roots))

x_vals = np.linspace(0.6, 5.0, 30)
k_modes = [1, 3, 5]

num_in_data = {1: [], 3: [], 5: []}
num_out_data = {1: [], 3: [], 5: []}

for x in x_vals:
    # Physical un-scaling for a given x = r0 / Rd
    r0 = x * Rd
    r1 = (2/3) * r0
    r2 = (4/3) * r0

    hat_r1 = r1 / r0
    hat_r2 = r2 / r0
    hat_c0sq = c0sq / (Omega**2 * r0**2)
    gamma = 2.0 * C / (Omega**2 * r0**3)

    hat_r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, hat_r1, hat_r2)

    for k in k_modes:
        eigs = find_eigenvalues(k, hat_r_grid, D1_mat, D2_mat, gamma, hat_c0sq)

        # WKB predictions removing the 0.5 factor from the document definitions:
        arg_in = k**2 * hat_c0sq / hat_r1**2 - hat_omA2
        oMK_in = -np.sqrt(arg_in) if arg_in >= 0 else np.nan

        arg_out = k**2 * hat_c0sq / hat_r2**2 - hat_omA2
        oMK_out = np.sqrt(arg_out) if arg_out >= 0 else np.nan

        # Match numerical root
        if not np.isnan(oMK_in):
            dists = [abs(e - oMK_in) for e in eigs]
            if dists: num_in_data[k].append(eigs[np.argmin(dists)])
            else: num_in_data[k].append(np.nan)
        else:
            num_in_data[k].append(np.nan)

        if not np.isnan(oMK_out):
            dists = [abs(e - oMK_out) for e in eigs]
            if dists: num_out_data[k].append(eigs[np.argmin(dists)])
            else: num_out_data[k].append(np.nan)
        else:
            num_out_data[k].append(np.nan)

# Plotting
fig, ax = plt.subplots(figsize=(7, 5))
for y_val in [-1, 0, 1]:
    ax.axhline(y_val, color='gray', lw=1.0, alpha=0.5)

line_styles = {1: '-', 3: '--', 5: ':'}
color = '#0055ff'

x_fine = np.linspace(0.6, 5.0, 500)
for k in [1, 3, 5]:
    ls = line_styles[k]

    # WKB Theory lines natively using the derived un-halved formulas
    arg_in_th = (k * 1.5 / x_fine)**2 - hat_omA**2
    valid_in = arg_in_th >= 0
    sigma_in = np.full_like(x_fine, np.nan)
    sigma_in[valid_in] = -np.sqrt(arg_in_th[valid_in])

    arg_out_th = (k * 0.75 / x_fine)**2 - hat_omA**2
    valid_out = arg_out_th >= 0
    sigma_out = np.full_like(x_fine, np.nan)
    sigma_out[valid_out] = np.sqrt(arg_out_th[valid_out])

    ax.plot(x_fine, sigma_out, color=color, ls=ls, lw=2.0)
    ax.plot(x_fine, sigma_in, color=color, ls=ls, lw=2.0)

    # Collocation dots
    ax.scatter(x_vals, num_out_data[k], facecolors='none', edgecolors='r', marker='o', s=35, zorder=5)
    ax.scatter(x_vals, num_in_data[k], facecolors='none', edgecolors='r', marker='o', s=35, zorder=5)

ax.text(1.2, 0.45, r'$k = 1$', fontsize=18, color='k', ha='center', va='center')
ax.text(2.6, 0.8, r'$k = 3$', fontsize=18, color='k', ha='center', va='center')
ax.text(4.2, 1.15, r'$k = 5$', fontsize=18, color='k', ha='center', va='center')

ax.set_xlim(0.5, 5.0)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

# The ylabel string matching image
ax.set_ylabel(r'$\frac{\sigma_k^{K\pm}}{f_e}$', rotation=0, labelpad=25, va='center', fontsize=24)
ax.set_xlabel(r'$r_0/R_d$', fontsize=22)

# Format ticks exactly as in the image
ax.set_yticklabels(['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5'])
ax.set_xticklabels(['1', '2', '3', '4', '5'])

plt.tight_layout()
plt.savefig('kelvin_modes_variation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('kelvin_modes_variation.png', dpi=300, bbox_inches='tight')
print("Saved Kelvin Model overlay plot.")''')
