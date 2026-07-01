"""
SWMHD Parameter Sweep (Astrophysical/Planetary Scales)
======================================================
Generates standard m-dispersion plots for discrete "Low", "Medium",
and "High" parameter combinations.
Satisfies the constraint: (hat_r2 - hat_r1) * |1 - gamma/2| / hat_c0^2 <= 0.05
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det
import itertools
import warnings
warnings.filterwarnings("ignore")

# ── Fixed Geometry (from astrophysical scale prompt) ────────────────────────
r1 = 9e7
r2 = 1e8
r0 = 0.5 * (r1 + r2)

hat_r1 = r1 / r0
hat_r2 = r2 / r0
delta_hat_r = hat_r2 - hat_r1

# Create output directory
out_dir = 'sweep_astro'
os.makedirs(out_dir, exist_ok=True)

# ── 1. Collocation machinery ────────────────────────────────────────────────
N_col = 32 # increased slightly for robustness but kept tractable

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

def build_matrix(hat_omega, m_val, hat_omA2, hat_c0sq, gamma):
    ws_hat = hat_omega + hat_omA2 / hat_omega

    Pv = 1.0 / hat_r_grid - (1.0 + gamma) * (hat_r_grid - 1.0) / hat_c0sq

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

def find_eigenvalues(m_val, hat_omA2, hat_c0sq, gamma, omega_range=(-5, 5), n_scan=400):
    scan = np.linspace(*omega_range, n_scan)
    scan = scan[np.abs(scan) > 0.01]
    ld = []
    for w in scan:
        try:
            d = np.log(np.abs(det(build_matrix(w, m_val, hat_omA2, hat_c0sq, gamma))) + 1e-300)
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
                    f0 = det(build_matrix(w, m_val, hat_omA2, hat_c0sq, gamma))
                    fp = ((det(build_matrix(w+dw, m_val, hat_omA2, hat_c0sq, gamma))
                           - det(build_matrix(w-dw, m_val, hat_omA2, hat_c0sq, gamma))) / (2*dw))
                    if abs(fp) < 1e-300: break
                    step = -f0 / fp; w += step
                    if abs(step) < 1e-10 * (abs(w) + 1): break
                except Exception:
                    break
            try:
                residual = abs(det(build_matrix(w, m_val, hat_omA2, hat_c0sq, gamma)))
                valid_ld = [x for x in ld if np.isfinite(x)]
                ld_max = max(valid_ld) if len(valid_ld) > 0 else 0
                if residual < 1e-3 * np.exp(ld_max):
                    dup = any(abs(w - wr) < 5e-3 for wr in roots)
                    if not dup:
                        roots.append(w)
            except Exception:
                pass
    return np.array(sorted([wr.real for wr in roots]))


# ── 2. WKB analytic dispersion relations ────────────────────────────────────
def wkb_branches(m_val, hat_omA2, hat_c0sq, gamma, n_radial=1):
    hat_r = 1.0
    hat_kr = n_radial * np.pi / delta_hat_r
    hat_kth = m_val / hat_r
    hat_k2 = hat_kr**2 + hat_kth**2

    hat_R = 1.0 + gamma
    hat_S = 0.0

    A2 = 2.0 * hat_omA2 - 1.0 - 0.25 * (hat_c0sq * hat_k2 + hat_R)
    A1 = 0.0
    A0 = hat_omA2 * (hat_omA2 - 0.25 * (hat_c0sq * hat_k2 + hat_R))

    disc = A2**2 - 4.0 * A0
    if disc < 0: disc = 0.0

    X1 = (-A2 + np.sqrt(disc)) / 2.0
    X2 = (-A2 - np.sqrt(disc)) / 2.0

    oMP_p = +np.sqrt(max(X1, 0))
    oMP_m = -np.sqrt(max(X1, 0))

    # MK and MS scaling using effective parameter for consistency
    # With physical H0/r0=50/9.5e7, hat_VA2 is very small. We just use the 4*hat_omA2 proxy
    # to demonstrate the branches mathematically.
    hat_VA2_eff = 4.0 * hat_omA2 * (50.0 / 9.5e7)**2
    oMK_hat = np.sqrt(hat_c0sq + hat_VA2_eff) * hat_kth / 2.0

    oR_hat = - (1.0 + gamma) * m_val / (2.0 * (hat_k2 + 4.0 / hat_c0sq))
    oMS_hat = - hat_VA2_eff * hat_k2 / (4.0 + hat_c0sq * hat_k2)

    return oMP_p, oMP_m, oMK_hat, oR_hat, oMS_hat


# ── 3. Sweeping Logic ───────────────────────────────────────────────────────
M_max = 15
m_arr = np.arange(1, M_max + 1)
m_fine = np.linspace(1, M_max, 200)

COLORS = {'MP': '#1f77b4', 'MK': '#2ca02c', 'R': '#d62728', 'MS': '#ff7f0e', 'col': 'k'}

# New parameter grid designed to satisfy constraint and show dynamical changes
# Constraint max test: gamma=7.0 -> |1 - 3.5| = 2.5
# 0.10526 * 2.5 / 6.0 = 0.0438 <= 0.05 (Satisfied)
levels = {
    'hat_omA':  {'Low': 0.01, 'Medium': 0.5, 'High': 2.0},
    'hat_c0sq': {'Low': 6.0,  'Medium': 20.0, 'High': 50.0},
    'gamma':    {'Low': 0.5,  'Medium': 2.0,  'High': 7.0}
}

states = ['Low', 'Medium', 'High']
combinations = list(itertools.product(states, repeat=3))

print(f"Starting full-factorial sweep for Astro scale ({len(combinations)} combinations)...")

for idx, (s_omA, s_c0sq, s_gamma) in enumerate(combinations, start=1):
    hat_omA = levels['hat_omA'][s_omA]
    hat_c0sq = levels['hat_c0sq'][s_c0sq]
    gamma = levels['gamma'][s_gamma]

    hat_omA2 = hat_omA**2
    constraint_val = delta_hat_r * abs(1.0 - gamma/2.0) / hat_c0sq

    name = f"omA-{s_omA}_c0sq-{s_c0sq}_gamma-{s_gamma}"

    print(f"\n[{idx}/{len(combinations)}] {name}")
    print(f"  hat_omega_A = {hat_omA}, hat_c0^2 = {hat_c0sq}, gamma = {gamma}")
    print(f"  Constraint Value: {constraint_val:.4f} (<= 0.05)")

    if constraint_val > 0.05:
        print("  WARNING: Constraint violated!")

    max_wkb = max(abs(wkb_branches(M_max, hat_omA2, hat_c0sq, gamma)[0]),
                  abs(wkb_branches(M_max, hat_omA2, hat_c0sq, gamma)[2]))
    scan_range = (-max_wkb - 0.5, max_wkb + 0.5)

    wkb = {key: [] for key in ['MP_p','MP_m','MK','R','MS']}
    for m_val in m_fine:
        oMP_p, oMP_m, oMK, oR, oMS = wkb_branches(m_val, hat_omA2, hat_c0sq, gamma)
        wkb['MP_p'].append(oMP_p)
        wkb['MP_m'].append(oMP_m)
        wkb['MK'].append(oMK)
        wkb['R'].append(oR)
        wkb['MS'].append(oMS)

    col_eigs = {}
    for m_val in m_arr:
        eigs = find_eigenvalues(m_val, hat_omA2, hat_c0sq, gamma, omega_range=scan_range, n_scan=400)
        col_eigs[m_val] = eigs

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.6, 1]})

    # Left Panel
    ax = axes[0]
    ax.plot(m_fine, wkb['MP_p'], color=COLORS['MP'], lw=2.0, ls='-',label='Magneto-Poincaré (WKB)')
    ax.plot(m_fine, wkb['MP_m'], color=COLORS['MP'], lw=2.0, ls='--')
    ax.plot(m_fine, wkb['MK'],   color=COLORS['MK'], lw=2.0, ls='-',label='Magneto-Kelvin (WKB)')
    ax.plot(m_fine, wkb['R'],    color=COLORS['R'],  lw=2.0, ls='-',label='Rossby (WKB)')
    ax.plot(m_fine, wkb['MS'],   color=COLORS['MS'], lw=2.0, ls='-',label='Magnetostrophic (WKB)')

    for m_val in m_arr:
        eigs_n = col_eigs[m_val]
        ax.scatter([m_val] * len(eigs_n), eigs_n, s=36, color=COLORS['col'], zorder=5, alpha=0.85, marker='o')

    ax.axhline(0, color='grey', lw=0.7, ls=':')
    ax.axhline(+1, color='grey', lw=0.6, ls='--', alpha=0.5)
    ax.axhline(-1, color='grey', lw=0.6, ls='--', alpha=0.5)

    if hat_omA > 0:
        ax.axhline(+hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
        ax.axhline(-hat_omA, color='purple', lw=0.8, ls=':', alpha=0.6)
        ax.text(M_max + 0.05, hat_omA + 0.1, r'$\hat\omega = \hat\omega_A$', va='bottom', ha='left', fontsize=8, color='purple')

    ax.set_xlabel('Azimuthal wavenumber  $m$', fontsize=13)
    ax.set_ylabel(r'Normalised frequency  $\hat\omega$', fontsize=13)
    ax.set_title(f'[{name}] Global dispersion relation (dimensionless)', fontsize=12)
    ax.set_xlim(0.7, M_max + 0.4)
    ax.set_xticks(m_arr)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    col_dot = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='k', markersize=7, label='Global eigenvalues (collocation)')
    ax.legend(handles=handles + [col_dot], loc='upper left', fontsize=9, framealpha=0.9)

    # Right Panel (Zoom)
    ax2 = axes[1]
    ax2.plot(m_fine, wkb['R'],  color=COLORS['R'],  lw=2.0, ls='-',label='Rossby (WKB)')
    ax2.plot(m_fine, wkb['MS'], color=COLORS['MS'], lw=2.0, ls='-',label='Magnetostrophic (WKB)')

    for m_val in m_arr:
        # slow mode filter
        slow = [e for e in col_eigs[m_val] if abs(e) < 1.2]
        if slow:
            ax2.scatter([m_val]*len(slow), slow, s=50, color=COLORS['col'], zorder=5, marker='o')

    ax2.axhline(0,  color='grey', lw=0.7, ls=':')
    ax2.axhline(-1, color='grey', lw=0.6, ls='--', alpha=0.5, label=r'$\hat\omega = -1$')
    ax2.set_xlabel('Azimuthal wavenumber  $m$', fontsize=13)
    ax2.set_ylabel(r'$\hat\omega$', fontsize=13)
    ax2.set_title('Slow branches (zoom)', fontsize=12)
    ax2.set_xlim(0.7, M_max + 0.4)

    # Auto-adjust zoom limits based on Rossby
    min_r = min(wkb['R'])
    if min_r < -0.001:
        ax2.set_ylim(min_r * 1.5 - 0.005, abs(min_r)*0.5 + 0.005)
    else:
        ax2.set_ylim(-0.02, 0.01)

    ax2.set_xticks(m_arr)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.25)

    param_text = (
        f"$\\hat{{\\omega}}_A = {hat_omA:.2f}$\n"
        f"$\\hat{{c}}_0^2 = {hat_c0sq:.2f}$\n"
        f"$\\gamma = {gamma:.2f}$\n"
        f"Constraint = {constraint_val:.4f}"
    )
    fig.text(0.52, 0.015, param_text, fontsize=10, va='bottom', ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    fname = os.path.join(out_dir, f'swmhd_astro_{name.lower()}.png')
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()

print("\nAstro sweep completed.")
