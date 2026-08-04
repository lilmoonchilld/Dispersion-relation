import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.linalg import svd, svdvals
from scipy.optimize import minimize_scalar
import warnings
import os

warnings.filterwarnings("ignore")

# ==========================================================
# Configurable Parameters (at the top of the script)
# ==========================================================
MU0      = 4 * np.pi * 1e-7
Omega    = 0.5e-4          # rad/s
H0       = 500.0           # m
g        = 9.81            # m/s²
rho0     = 1000.0          # kg/m³
C        = 1.882e9         # m³/s² (validated: δH/H₀ = 0.15)
B0       = 8.8623e-4       # T     (gives ω_A/f = 0.50)
r1       = 0.5e6           # inner radius [m]
r2       = 1.0e6           # outer radius [m]
N        = 32              # Chebyshev resolution

# Derived quantities
r0       = 0.5 * (r1 + r2)
Delta_r  = r2 - r1
VA2      = B0**2 / (MU0 * rho0)
f        = 2.0 * Omega
hat_omA2 = VA2 / (H0**2 * f**2)   # (ω_A/f)²
hat_omA  = np.sqrt(hat_omA2)
c0sq     = g * H0
c0       = np.sqrt(c0sq)
gamma    = 2.0 * C / (Omega**2 * r0**3)
hat_c0sq = c0sq / (Omega**2 * r0**2)
hat_r1   = r1 / r0
hat_r2   = r2 / r0

a_eff    = Omega**2 * r0 - C / r0**2
dH_H0    = abs(a_eff) * Delta_r / (g * H0)

print("=" * 60)
print("  SWMHD Eigenfunction Plotter - Parameter Summary")
print("=" * 60)
print(f"  γ          : {gamma:.4f}")
print(f"  ĉ₀²        : {hat_c0sq:.4f}")
print(f"  ω_A/f      : {hat_omA:.4f}")
print(f"  δH/H₀      : {dH_H0:.4f}")
print(f"  N          : {N}")
print(f"  Domain     : hat_r ∈ [{hat_r1:.3f}, {hat_r2:.3f}]")
print("-" * 60)

# ==========================================================
# Publication Style Settings
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})

# ==========================================================
# 1. Chebyshev Machinery
# ==========================================================
def cheb(N, a, b):
    j = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N)
    c[0] = 2
    c[-1] = 2
    X = np.tile(xi, (N, 1))
    dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    sc = 2 / (b - a)
    D1 = sc * D
    D2 = D1 @ D1
    xp = 0.5 * (a + b) + 0.5 * (b - a) * xi
    xp = xp[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]
    return xp, D1, D2

xg, D1m, D2m = cheb(N, hat_r1, hat_r2)

# ==========================================================
# 2. Collocation Matrix and SVD Solver
# ==========================================================
def build_L(Oh, m):
    ws = Oh + hat_omA2 / Oh
    Pv = 1.0 / xg - (1.0 + gamma) * (xg - 1.0) / hat_c0sq
    Qv = (4.0 * Oh * (ws**2 - 1.0) / (ws * hat_c0sq)
          - (m / xg)**2
          - (1.0 + gamma) * (2.0 * xg - 1.0) / (hat_c0sq * xg)
          - m * (1.0 + gamma) * (xg - 1.0) / (ws * hat_c0sq * xg))
    L = D2m + np.diag(Pv) @ D1m + np.diag(Qv)
    for idx in [0, N - 1]:
        xb = xg[idx]
        e = np.eye(N)[idx]
        bc = -(ws * (1.0 + gamma) * (xb - 1.0) + m * hat_c0sq / xb)
        L[idx, :] = ws * hat_c0sq * D1m[idx, :] + bc * e
    return L

def rel_sigma(Oh, m):
    if abs(Oh) < 0.02:
        return np.inf
    sv = svdvals(build_L(Oh, m))
    return float(sv[-1] / sv[0])

def find_eigenvalue(Oh_init, m, bracket=0.6):
    lo = Oh_init - bracket
    hi = Oh_init + bracket
    if lo * hi < 0:
        lo = 0.02 if Oh_init > 0 else -hi
        hi = Oh_init + bracket
    res = minimize_scalar(lambda w: rel_sigma(w, m),
                          bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-12, 'maxiter': 200})
    sv = svdvals(build_L(res.x, m))
    eta, _ = extract_eta(res.x, m)
    return res.x, eta, sv[-1] / sv[0]

def extract_eta(Oh, m):
    L = build_L(Oh, m)
    _, sv, Vh = svd(L, full_matrices=False)
    eta = Vh[-1, :].copy()

    # Smooth phase alignment: ensure eta[0] is positive or matches physical sense
    # To avoid sign flips, normalize by the max absolute value and ensure the boundary or bulk value has a positive sign
    eta /= np.max(np.abs(eta))
    if eta[0] < 0:
        eta = -eta

    return eta, sv[-1]

# ==========================================================
# 3. Velocity Reconstruction (Cramér's Rule)
# ==========================================================
def reconstruct_velocity(Oh, m, eta):
    ws = Oh + hat_omA2 / Oh
    D_star = ws**2 - 4.0   # denominator

    # Radial derivative on Chebyshev grid
    deta_dx = D1m @ eta

    # RHS components (dimensionless)
    RHS_r = (hat_c0sq * deta_dx
             - (1.0 + gamma) * (xg - 1.0) * eta
             - m * (1.0 + gamma) * (xg - 1.0) * eta / (ws * xg))
    RHS_th = (m / xg) * hat_c0sq * eta

    # Cramér's rule:
    # v_r  = (ws * RHS_r  + 2 * RHS_th) / D*
    # v_th = (ws * RHS_th - 2 * RHS_r ) / D*
    v_r  = (ws * RHS_r  + 2.0 * RHS_th) / D_star
    v_th = (ws * RHS_th - 2.0 * RHS_r ) / D_star

    u0 = np.sqrt(v_r**2 + v_th**2)
    u0_max = np.max(u0) if np.max(u0) > 0 else 1.0
    v_r  /= u0_max
    v_th /= u0_max
    u0   /= u0_max

    return v_r, v_th, u0

# ==========================================================
# 4. Barycentric Interpolation
# ==========================================================
def bary_interp(x_eval, x_nodes, f_nodes):
    N_n = len(x_nodes)
    w = np.ones(N_n)
    w[::2] = -1
    w[0] /= 2
    w[-1] /= 2
    numer = np.zeros(len(x_eval), dtype=complex)
    denom = np.zeros(len(x_eval))
    for j in range(N_n):
        diff = x_eval - x_nodes[j]
        tiny = np.abs(diff) < 1e-14
        diff[tiny] = 1e-14
        t = w[j] / diff
        numer += t * f_nodes[j]
        denom += t
    return (numer / denom).real

# ==========================================================
# 5. Find and Assign Wavenumber m=2 Modes
# ==========================================================
m = 2
scan = np.linspace(-20, 20, 2000)
scan = scan[np.abs(scan) > 0.05]
rs   = [rel_sigma(w, m) for w in scan]

cands = []
for k in range(1, len(rs)-1):
    if rs[k] < rs[k-1] and rs[k] < rs[k+1] and rs[k] < 1e-6:
        cands.append(scan[k])

modes = []
for Oh_c in cands:
    Oh, eta, rel = find_eigenvalue(Oh_c, m)
    modes.append({'Oh': Oh, 'eta': eta, 'rel_sigma': rel})

# Separate positive and negative modes
pos_modes = sorted([md for md in modes if md['Oh'] > 0], key=lambda x: x['Oh'])
neg_modes = sorted([md for md in modes if md['Oh'] < 0], key=lambda x: -x['Oh'])

# Select modes based on our classification:
# Kelvin modes: no interior nodes (smooth), slow modes (smallest absolute frequencies)
# For m=2:
# Kelvin- (counter-rotating): neg_modes[0] (Oh ≈ -2.0326)
# Kelvin+ (co-rotating): pos_modes[0] (Oh ≈ 1.7434)
# Poincaré- (counter-rotating): neg_modes[1] (Oh ≈ -5.1077)
# Poincaré+ (co-rotating): pos_modes[1] (Oh ≈ 5.1080)

kelvin_minus = neg_modes[0]
kelvin_plus  = pos_modes[0]
poincare_minus = neg_modes[1]
poincare_plus  = pos_modes[1]

# Ensure eta of co-rotating Kelvin+ is phase aligned with counter-rotating Kelvin-
# Let's verify and invert if necessary so they look visually consistent (e.g. positive slope or matches style)
# We will check if the bulk structure is well-behaved.

# ==========================================================
# 6. Plotting Functions for the Figures
# ==========================================================
def draw_mode_panel(ax_main, Oh, eta, label, is_co_rotating, y_lim_eta, y_lim_u0, inset_loc, cmap='RdBu_r'):
    """
    Draws a single subpanel matching the paper's exact layout.
    """
    r_prime = (xg - hat_r1) / (hat_r2 - hat_r1)
    r_prime_fine = np.linspace(0, 1, 200)
    r_fine_hat   = r_prime_fine * (hat_r2 - hat_r1) + hat_r1

    # Interpolate fields to fine 1D grid
    eta_fine  = bary_interp(r_fine_hat, xg, eta)
    v_r, v_th, u0 = reconstruct_velocity(Oh, m, eta)
    vr_fine   = bary_interp(r_fine_hat, xg, v_r)
    vth_fine  = bary_interp(r_fine_hat, xg, v_th)
    u0_fine   = np.sqrt(vr_fine**2 + vth_fine**2)
    u0_fine  /= np.max(u0_fine) if np.max(u0_fine) > 0 else 1.0

    ax_eta = ax_main
    ax_u0  = ax_main.twinx()

    # Left y-axis: η₀ (blue)
    ax_eta.plot(r_prime_fine, eta_fine, color='#1f77b4', lw=2.2)
    ax_eta.axhline(0, color='grey', lw=0.7, ls=':')
    ax_eta.set_ylabel(r'$\eta_0$', color='#1f77b4', fontsize=12)
    ax_eta.tick_params(axis='y', labelcolor='#1f77b4')
    ax_eta.set_ylim(y_lim_eta[0], y_lim_eta[1])
    ax_eta.set_yticks(np.linspace(y_lim_eta[0], y_lim_eta[1], 5))

    # Right y-axis: u₀ (black)
    ax_u0.plot(r_prime_fine, u0_fine, color='black', lw=2.0, ls='-')
    ax_u0.set_ylabel(r'$u_0$', color='black', fontsize=12)
    ax_u0.set_ylim(y_lim_u0[0], y_lim_u0[1])
    ax_u0.set_yticks(np.linspace(y_lim_u0[0], y_lim_u0[1], 5))

    # Labels and grid
    ax_main.set_xlim(0, 1)
    ax_main.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_main.set_xlabel(r"$r'$", fontsize=12)
    ax_main.grid(False)

    # LaTeX Mode labels exactly matching paper: e.g. \sigma_{2}^{K-} = -2.0326 f_e
    wave_type = "K" if "Kelvin" in label else "P"
    sign_str = "-" if Oh < 0 else "+"
    omega_str = f"{Oh:.4f}"

    # Subscript / Superscript formatting
    if wave_type == "K":
        title_latex = r"$\sigma_{2}^{K" + sign_str + r"} = " + omega_str + r"f_e$"
    else:
        # Poincaré has radial mode n=1
        title_latex = r"$\sigma_{2,1}^{P" + sign_str + r"} = " + omega_str + r"f_e$"

    # Add the LaTeX title inside/above the panel
    ax_main.set_title(title_latex, fontsize=12, pad=10)

    # ── 2D polar inset ───────────────────────────────────────────────────────
    # We will adjust placement based on inset_loc
    # inset_loc is (x0, y0, width, height) relative to ax_main
    ax_polar = inset_axes(ax_main, width="100%", height="100%",
                          bbox_to_anchor=inset_loc,
                          bbox_transform=ax_main.transAxes,
                          loc='lower left')
    ax_polar.set_aspect('equal')

    # Build 2D fields
    N_theta = 100
    N_r = 50
    theta_arr = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    r_arr_hat = np.linspace(hat_r1, hat_r2, N_r)
    r_arr_phys = r_arr_hat * r0

    eta_col_fine = np.array([bary_interp(np.array([rh]), xg, eta) for rh in r_arr_hat])
    vr_col_fine  = np.array([bary_interp(np.array([rh]), xg, v_r) for rh in r_arr_hat])
    vth_col_fine = np.array([bary_interp(np.array([rh]), xg, v_th) for rh in r_arr_hat])

    # 2D polar meshgrid
    R2D, T2D = np.meshgrid(r_arr_phys, theta_arr, indexing='ij')
    ETA2D = np.outer(eta_col_fine, np.cos(m * theta_arr))
    VR2D  = np.outer(vr_col_fine,  np.cos(m * theta_arr))
    VTH2D = np.outer(vth_col_fine, np.sin(m * theta_arr))

    # Convert to Cartesian for plotting
    X2D = R2D * np.cos(T2D)
    Y2D = R2D * np.sin(T2D)
    VX2D = VR2D * np.cos(T2D) - VTH2D * np.sin(T2D)
    VY2D = VR2D * np.sin(T2D) + VTH2D * np.cos(T2D)

    # Normalize color limits symmetrically based on max amplitude
    eta_max = np.max(np.abs(ETA2D)) or 1.0
    norm = TwoSlopeNorm(vmin=-eta_max, vcenter=0, vmax=eta_max)

    pc = ax_polar.pcolormesh(X2D, Y2D, ETA2D, cmap=cmap, norm=norm, shading='auto', zorder=1)

    # Add vertical colorbar immediately to the right of this polar plot
    # Using inset_axes for colorbar
    ax_cb = inset_axes(ax_polar, width="8%", height="100%", loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1., 1.),
                       bbox_transform=ax_polar.transAxes,
                       borderpad=0)
    cb = plt.colorbar(pc, cax=ax_cb, orientation='vertical')
    cb.set_ticks([-eta_max, 0, eta_max])
    cb.set_ticklabels(['-1', '0', '1'])
    cb.ax.tick_params(labelsize=10)

    # Velocity vector overlays (quiver)
    skip_r = 4
    skip_t = 10
    Xq = X2D[::skip_r, ::skip_t]
    Yq = Y2D[::skip_r, ::skip_t]
    UXq = VX2D[::skip_r, ::skip_t]
    UYq = VY2D[::skip_r, ::skip_t]
    speed = np.sqrt(UXq**2 + UYq**2) + 1e-15
    ax_polar.quiver(Xq, Yq, UXq / speed, UYq / speed,
                    scale=28, width=0.004, color='black',
                    alpha=0.8, zorder=2)

    # Draw inner/outer boundaries
    circle_angles = np.linspace(0, 2 * np.pi, 300)
    for rcirc in [r1, r2]:
        ax_polar.plot(rcirc * np.cos(circle_angles), rcirc * np.sin(circle_angles),
                      'k-', lw=1.2, zorder=3)

    # Co-rotating waves should show axes coordinate labels (-1, 1), Counter-rotating are off
    if is_co_rotating:
        ax_polar.axis('on')
        # Format the axes to show ticks and labels like the reference image
        # Outer radius is 1.0e6, so ticks should be [-1.0e6, 0, 1.0e6]
        # Let's show labels as -1, -0.5, 0, 0.5, 1 (rescaling by 10^6)
        ax_polar.set_xlim(-1.1e6, 1.1e6)
        ax_polar.set_ylim(-1.1e6, 1.1e6)
        ax_polar.set_xticks([-1.0e6, -0.5e6, 0, 0.5e6, 1.0e6])
        ax_polar.set_yticks([-1.0e6, -0.5e6, 0, 0.5e6, 1.0e6])
        ax_polar.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'], fontsize=9)
        ax_polar.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'], fontsize=9)
        ax_polar.tick_params(axis='both', which='both', direction='out', length=3, width=1, colors='black')
        # Light grid
        ax_polar.grid(True, linestyle=':', alpha=0.3, color='grey')
    else:
        ax_polar.axis('off')

# ==========================================================
# 7. Generate and Save Figures
# ==========================================================
# Create output directory if not exists
os.makedirs("outputs", exist_ok=True)

# ── Figure 3: Kelvin Waves ────────────────────────────────
print("\nGenerating Figure 3 (Kelvin waves)...")
fig3 = plt.figure(figsize=(12, 5.5))
gs3 = gridspec.GridSpec(1, 2, figure=fig3, wspace=0.35, left=0.08, right=0.92, bottom=0.15, top=0.85)

# Panel (a): Counter-rotating Kelvin (Kelvin-)
ax3a = fig3.add_subplot(gs3[0, 0])
# Inset location (lower-left to avoid curve in top portion)
# x_start, y_start, width, height relative to parent axis
inset_3a = (0.05, 0.08, 0.42, 0.42)
draw_mode_panel(
    ax_main=ax3a,
    Oh=kelvin_minus['Oh'],
    eta=kelvin_minus['eta'],
    label='Kelvin-',
    is_co_rotating=False,
    y_lim_eta=[0, 2],
    y_lim_u0=[-0.2, 0.4],
    inset_loc=inset_3a
)
ax3a.text(0.5, -0.15, "(a) Counter-rotating Kelvin waves.", transform=ax3a.transAxes,
          ha='center', fontsize=12, fontweight='normal')

# Panel (b): Co-rotating Kelvin (Kelvin+)
ax3b = fig3.add_subplot(gs3[0, 1])
# Inset location (middle-right, x from 0.5 to 0.9, y from 0.08 to 0.5 to avoid curve)
inset_3b = (0.45, 0.08, 0.42, 0.42)
draw_mode_panel(
    ax_main=ax3b,
    Oh=kelvin_plus['Oh'],
    eta=kelvin_plus['eta'],
    label='Kelvin+',
    is_co_rotating=True,
    y_lim_eta=[-2, 2],
    y_lim_u0=[-0.2, 0.2],
    inset_loc=inset_3b
)
ax3b.text(0.5, -0.15, "(b) Co-rotating Kelvin waves.", transform=ax3b.transAxes,
          ha='center', fontsize=12, fontweight='normal')

plt.savefig("outputs/swmhd_kelvin_modes.pdf", dpi=200, bbox_inches='tight')
plt.savefig("outputs/swmhd_kelvin_modes.png", dpi=200, bbox_inches='tight')
print("  Saved: outputs/swmhd_kelvin_modes.pdf and .png")
plt.close(fig3)


# ── Figure 4: Poincaré Waves ──────────────────────────────
print("Generating Figure 4 (Poincaré waves)...")
fig4 = plt.figure(figsize=(12, 5.5))
gs4 = gridspec.GridSpec(1, 2, figure=fig4, wspace=0.35, left=0.08, right=0.92, bottom=0.15, top=0.85)

# Panel (a): Counter-rotating Poincaré (Poincaré-)
ax4a = fig4.add_subplot(gs4[0, 0])
# Inset location (lower-middle/right)
inset_4a = (0.28, 0.08, 0.42, 0.42)
draw_mode_panel(
    ax_main=ax4a,
    Oh=poincare_minus['Oh'],
    eta=poincare_minus['eta'],
    label='Poincare-',
    is_co_rotating=False,
    y_lim_eta=[-6, 0],
    y_lim_u0=[-0.4, 0.2],
    inset_loc=inset_4a
)
ax4a.text(0.5, -0.15, "(a) Counter-rotating Poincaré waves.", transform=ax4a.transAxes,
          ha='center', fontsize=12, fontweight='normal')

# Panel (b): Co-rotating Poincaré (Poincaré+)
ax4b = fig4.add_subplot(gs4[0, 1])
# Inset location (lower-middle/right)
inset_4b = (0.28, 0.08, 0.42, 0.42)
draw_mode_panel(
    ax_main=ax4b,
    Oh=poincare_plus['Oh'],
    eta=poincare_plus['eta'],
    label='Poincare+',
    is_co_rotating=True,
    y_lim_eta=[-6, 0],
    y_lim_u0=[-0.4, 0.2],
    inset_loc=inset_4b
)
ax4b.text(0.5, -0.15, "(b) Co-rotating Poincaré waves.", transform=ax4b.transAxes,
          ha='center', fontsize=12, fontweight='normal')

plt.savefig("outputs/swmhd_poincare_modes.pdf", dpi=200, bbox_inches='tight')
plt.savefig("outputs/swmhd_poincare_modes.png", dpi=200, bbox_inches='tight')
print("  Saved: outputs/swmhd_poincare_modes.pdf and .png")
plt.close(fig4)

print("\nAll done successfully!")
