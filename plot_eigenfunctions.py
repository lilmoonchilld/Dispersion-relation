import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import svd, svdvals
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# Publication style settings
# ==============================================================================
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

# ==============================================================================
# 0. Configurable Parameters (at the top of the script)
# ==============================================================================
MU0 = 4.0 * np.pi * 1e-7
Omega = 0.5e-4       # rad/s
H0 = 500.0           # m
g = 9.81             # m/s^2
rho0 = 1000.0        # kg/m^3
C = 1.882e9          # m^3/s^2
B0 = 8.8623e-4       # T

r1 = 0.5e6           # inner radius [m]
r2 = 1.0e6           # outer radius [m]
N = 32               # grid resolution
m = 2                # azimuthal mode number

# Derived parameters
r0 = 0.5 * (r1 + r2)
Delta_r = r2 - r1
VA2 = B0**2 / (MU0 * rho0)
f = 2.0 * Omega
hat_omA2 = VA2 / (H0**2 * f**2)
hat_omA = np.sqrt(hat_omA2)
c0sq = g * H0
gamma = 2.0 * C / (Omega**2 * r0**3)
hat_c0sq = c0sq / (Omega**2 * r0**2)
hat_r1 = r1 / r0
hat_r2 = r2 / r0

a_eff = Omega**2 * r0 - C / r0**2
dH_H0 = abs(a_eff) * Delta_r / (g * H0)

print("="*60)
print(" SWMHD Eigenfunction Generator - Parameter Summary")
print("="*60)
print(f" γ      = {gamma:.4f}")
print(f" ĉ₀²    = {hat_c0sq:.4f}")
print(f" ω_A/f  = {hat_omA:.4f}")
print(f" δH/H₀  = {dH_H0:.4f}")
print(f" N      = {N}")
print(f" m      = {m}")
print("-"*60)

# ==============================================================================
# 1. Chebyshev Machinery
# ==============================================================================
def cheb(N_grid, a, b):
    j = np.arange(N_grid)
    xi = np.cos(j * np.pi / (N_grid - 1))
    c = np.ones(N_grid)
    c[0] = 2; c[-1] = 2
    X = np.tile(xi, (N_grid, 1))
    dX = X - X.T
    D = np.zeros((N_grid, N_grid))
    for i in range(N_grid):
        for k in range(N_grid):
            if i != k:
                D[i, k] = (c[i] / c[k]) / dX[i, k]
    D -= np.diag(D.sum(axis=1))
    sc = 2.0 / (b - a)
    D1 = sc * D
    D2 = D1 @ D1
    xp = 0.5 * (a + b) + 0.5 * (b - a) * xi
    # reverse lists to go from a to b
    xp = xp[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]
    return xp, D1, D2

xg, D1m, D2m = cheb(N, hat_r1, hat_r2)

# ==============================================================================
# 2. Collocation Matrix Assembly
# ==============================================================================
def build_L(Oh, m_val):
    ws = Oh + hat_omA2 / Oh
    Pv = 1.0 / xg - (1.0 + gamma) * (xg - 1.0) / hat_c0sq
    Qv = (4.0 * Oh * (ws**2 - 1.0) / (ws * hat_c0sq)
          - (m_val / xg)**2
          - (1.0 + gamma) * (2.0 * xg - 1.0) / (hat_c0sq * xg)
          - m_val * (1.0 + gamma) * (xg - 1.0) / (ws * hat_c0sq * xg))
    L = D2m + np.diag(Pv) @ D1m + np.diag(Qv)
    for idx in [0, N - 1]:
        xb = xg[idx]
        e = np.eye(N)[idx]
        bc = -(ws * (1.0 + gamma) * (xb - 1.0) + m_val * hat_c0sq / xb)
        L[idx, :] = ws * hat_c0sq * D1m[idx, :] + bc * e
    return L

def rel_sigma(Oh, m_val):
    if abs(Oh) < 0.02:
        return np.inf
    sv = svdvals(build_L(Oh, m_val))
    return float(sv[-1] / sv[0])

# ==============================================================================
# 3. Eigenvalue Finder & Refiner
# ==============================================================================
def find_eigenvalue(Oh_init, m_val, bracket=0.4):
    lo = Oh_init - bracket
    hi = Oh_init + bracket
    if lo * hi < 0:
        lo = 0.02 if Oh_init > 0 else -bracket
    res = minimize_scalar(lambda w: rel_sigma(w, m_val), bounds=(lo, hi), method='bounded', options={'xatol': 1e-12, 'maxiter': 200})
    return res.x

def extract_eta(Oh, m_val):
    L = build_L(Oh, m_val)
    _, _, Vh = svd(L, full_matrices=False)
    eta = Vh[-1, :].copy()
    eta /= np.max(np.abs(eta))
    return eta

# ==============================================================================
# 4. Velocity Reconstruction (Cramér's Rule)
# ==============================================================================
def reconstruct_velocity(Oh, m_val, eta, target_max_u0):
    ws = Oh + hat_omA2 / Oh
    D_star = ws**2 - 4.0
    deta_dx = D1m @ eta

    RHS_r = (hat_c0sq * deta_dx
             - (1.0 + gamma) * (xg - 1.0) * eta
             - m_val * (1.0 + gamma) * (xg - 1.0) * eta / (ws * xg))
    RHS_th = (m_val / xg) * hat_c0sq * eta

    v_r = (ws * RHS_r + 2.0 * RHS_th) / D_star
    v_th = (ws * RHS_th - 2.0 * RHS_r) / D_star

    u0 = np.sqrt(v_r**2 + v_th**2)
    u0_max = np.max(u0) if np.max(u0) > 0 else 1.0
    v_r = v_r / u0_max * target_max_u0
    v_th = v_th / u0_max * target_max_u0
    u0 = u0 / u0_max * target_max_u0

    return v_r, v_th, u0

# ==============================================================================
# 5. Chebyshev Barycentric Interpolation
# ==============================================================================
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

# ==============================================================================
# 6. Locate and Polish the Four Target Modes
# ==============================================================================
# 1. Kelvin counter-rotating (approx -0.5)
Oh_K_neg = find_eigenvalue(-0.5, m)
eta_K_neg = extract_eta(Oh_K_neg, m)
if eta_K_neg[0] < 0:
    eta_K_neg *= -1

# 2. Kelvin co-rotating (approx 1.596)
Oh_K_pos = find_eigenvalue(1.596, m)
eta_K_pos = extract_eta(Oh_K_pos, m)
if eta_K_pos[0] < 0:
    eta_K_pos *= -1

# 3. Poincaré counter-rotating (approx -5.1077)
Oh_P_neg = find_eigenvalue(-5.1077, m)
eta_P_neg = extract_eta(Oh_P_neg, m)
if eta_P_neg[0] > 0:
    eta_P_neg *= -1

# 4. Poincaré co-rotating (approx 5.1080)
Oh_P_pos = find_eigenvalue(5.1080, m)
eta_P_pos = extract_eta(Oh_P_pos, m)
if eta_P_pos[0] > 0:
    eta_P_pos *= -1

print("Polished frequencies:")
print(f" 1. Kelvin Counter-Rotating:   Oh = {Oh_K_neg: .6f}")
print(f" 2. Kelvin Co-Rotating:        Oh = {Oh_K_pos: .6f}")
print(f" 3. Poincaré Counter-Rotating: Oh = {Oh_P_neg: .6f}")
print(f" 4. Poincaré Co-Rotating:      Oh = {Oh_P_pos: .6f}")
print("-"*60)

# ==============================================================================
# 7. Setup Directory for Outputs
# ==============================================================================
os.makedirs("outputs", exist_ok=True)

# ==============================================================================
# 8. Core Plotting Routine for Standalone Standalone Figures
# ==============================================================================
def create_standalone_plot(Oh, eta, m_val, label, filename, eta_lim, u0_lim, target_max_u0, show_ticks, inset_loc, cmap='RdBu_r'):
    """
    Generate and save a standalone, publication-quality 1D radial + 2D polar plot for a single mode.
    """
    fig, ax_main = plt.subplots(figsize=(6.5, 5))

    # Coordinates mapping: r' in [0, 1]
    r_prime = (xg - hat_r1) / (hat_r2 - hat_r1)
    r_prime_fine = np.linspace(0, 1, 200)
    r_fine_hat = r_prime_fine * (hat_r2 - hat_r1) + hat_r1

    # Interpolate displacement and reconstructed velocity to fine grid
    eta_fine = bary_interp(r_fine_hat, xg, eta)
    v_r, v_th, u0 = reconstruct_velocity(Oh, m_val, eta, target_max_u0)
    vr_fine = bary_interp(r_fine_hat, xg, v_r)
    vth_fine = bary_interp(r_fine_hat, xg, v_th)

    u0_fine = np.sqrt(vr_fine**2 + vth_fine**2)
    u0_fine_max = np.max(u0_fine) if np.max(u0_fine) > 0 else 1.0
    u0_fine = u0_fine / u0_fine_max * target_max_u0

    # 1D Left axis (blue): surface displacement eta
    ax_main.set_xlim(0, 1)
    ax_main.set_xlabel(r"$r'=(r-r_1)/\Delta r$", fontsize=12)

    ax_eta = ax_main
    ax_u0 = ax_main.twinx()

    ax_eta.plot(r_prime_fine, eta_fine, color='#1f77b4', lw=2.2, label=r'$\eta_0$')
    ax_eta.axhline(0, color='grey', lw=0.7, ls=':')
    ax_eta.set_ylabel(r'$\eta_0$', color='#1f77b4', fontsize=13)
    ax_eta.tick_params(axis='y', labelcolor='#1f77b4')
    ax_eta.set_ylim(eta_lim[0], eta_lim[1])

    # 1D Right axis (black): normalized velocity magnitude u0
    ax_u0.plot(r_prime_fine, u0_fine, color='black', lw=2.0, ls='-', label=r'$u_0$')
    ax_u0.set_ylabel(r'$u_0$', color='black', fontsize=13)
    ax_u0.tick_params(axis='y', labelcolor='black')
    ax_u0.set_ylim(u0_lim[0], u0_lim[1])

    # Title with LaTeX formatting
    if 'Kelvin' in label:
        sup = 'K-' if 'Counter' in label else 'K+'
        sub = f'{m_val}'
    else:
        sup = 'P-' if 'Counter' in label else 'P+'
        sub = f'{m_val},1'

    title_str = rf"$\sigma_{{{sub}}}^{{{sup}}} = {Oh:.4f}f_e$"
    ax_main.set_title(title_str, fontsize=12)

    # --------------------------------------------------------------------------
    # 2D Polar Inset
    # --------------------------------------------------------------------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_polar = inset_axes(ax_main, width="40%", height="40%",
                          loc='lower center',
                          bbox_to_anchor=inset_loc,
                          bbox_transform=ax_main.transAxes)
    ax_polar.set_aspect('equal')

    if not show_ticks:
        ax_polar.axis('off')
    else:
        # Show coordinate axis ticks as co-rotating modes do in the paper
        ax_polar.set_xticks([-0.5, 0, 0.5])
        ax_polar.set_yticks([-0.5, 0, 0.5])
        ax_polar.tick_params(axis='both', labelsize=8)
        ax_polar.grid(False)

    # Build 2D annular field
    N_theta = 100
    N_r = 50
    theta_arr = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    r_arr_hat = np.linspace(hat_r1, hat_r2, N_r)
    r_arr_phys = r_arr_hat * r0 / 1e6 # in units of 10^6 m (0.5 to 1.0)

    eta_col_fine = np.array([bary_interp(np.array([rh]), xg, eta)[0] for rh in r_arr_hat])
    vr_col_fine = np.array([bary_interp(np.array([rh]), xg, v_r)[0] for rh in r_arr_hat])
    vth_col_fine = np.array([bary_interp(np.array([rh]), xg, v_th)[0] for rh in r_arr_hat])

    R2D, T2D = np.meshgrid(r_arr_phys, theta_arr, indexing='ij')
    ETA2D = np.outer(eta_col_fine, np.cos(m_val * theta_arr))
    VR2D = np.outer(vr_col_fine, np.cos(m_val * theta_arr))
    VTH2D = np.outer(vth_col_fine, np.sin(m_val * theta_arr))

    # Cartesian conversion for quiver and pcolormesh
    X2D = R2D * np.cos(T2D)
    Y2D = R2D * np.sin(T2D)
    VX2D = VR2D * np.cos(T2D) - VTH2D * np.sin(T2D)
    VY2D = VR2D * np.sin(T2D) + VTH2D * np.cos(T2D)

    # Normalize cmap symmetrically to the absolute max value of ETA2D
    eta_max = np.max(np.abs(ETA2D)) or 1.0
    norm = TwoSlopeNorm(vmin=-eta_max, vcenter=0, vmax=eta_max)

    pc = ax_polar.pcolormesh(X2D, Y2D, ETA2D, cmap=cmap, norm=norm, shading='auto', zorder=1)

    # Add symmetrical vertical colorbar to the right of the polar plot
    cax = inset_axes(ax_polar, width="8%", height="100%", loc='right',
                     bbox_to_anchor=(0.12, 0., 1.0, 1.0),
                     bbox_transform=ax_polar.transAxes,
                     borderpad=0)
    cb = fig.colorbar(pc, cax=cax, orientation='vertical')
    cb.set_ticks([-eta_max, 0, eta_max])
    cb.set_ticklabels(['-1', '0', '1'])
    cb.ax.tick_params(labelsize=8)

    # Subsample quiver vectors
    skip_r = 4
    skip_t = 10
    Xq = X2D[::skip_r, ::skip_t]
    Yq = Y2D[::skip_r, ::skip_t]
    UXq = VX2D[::skip_r, ::skip_t]
    UYq = VY2D[::skip_r, ::skip_t]
    speed = np.sqrt(UXq**2 + UYq**2) + 1e-15
    ax_polar.quiver(Xq, Yq, UXq / speed, UYq / speed,
                    scale=25, width=0.005, color='black',
                    alpha=0.75, zorder=2)

    # Draw physical inner/outer channel walls
    circle_angles = np.linspace(0, 2*np.pi, 300)
    for rcirc in [hat_r1 * r0 / 1e6, hat_r2 * r0 / 1e6]:
        ax_polar.plot(rcirc * np.cos(circle_angles),
                      rcirc * np.sin(circle_angles),
                      'k-', lw=1.2, zorder=3)

    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# ==============================================================================
# 9. Generate the Four Standalone Figure Plots
# ==============================================================================
create_standalone_plot(
    Oh=Oh_K_neg,
    eta=eta_K_neg,
    m_val=m,
    label="Kelvin Counter-Rotating",
    filename="outputs/kelvin_counter_rotating.png",
    eta_lim=[0.0, 2.0],
    u0_lim=[-0.2, 0.4],
    target_max_u0=0.35,
    show_ticks=False,
    inset_loc=(-0.15, 0.02, 1.0, 1.0)
)

create_standalone_plot(
    Oh=Oh_K_pos,
    eta=eta_K_pos,
    m_val=m,
    label="Kelvin Co-Rotating",
    filename="outputs/kelvin_co_rotating.png",
    eta_lim=[-2.0, 2.0],
    u0_lim=[-0.2, 0.2],
    target_max_u0=0.10,
    show_ticks=True,
    inset_loc=(0.0, 0.02, 1.0, 1.0)
)

create_standalone_plot(
    Oh=Oh_P_neg,
    eta=eta_P_neg,
    m_val=m,
    label="Poincaré Counter-Rotating",
    filename="outputs/poincare_counter_rotating.png",
    eta_lim=[-6.0, 0.0],
    u0_lim=[-0.4, 0.2],
    target_max_u0=0.20,
    show_ticks=False,
    inset_loc=(-0.1, 0.08, 1.0, 1.0)
)

create_standalone_plot(
    Oh=Oh_P_pos,
    eta=eta_P_pos,
    m_val=m,
    label="Poincaré Co-Rotating",
    filename="outputs/poincare_co_rotating.png",
    eta_lim=[-6.0, 0.0],
    u0_lim=[-0.4, 0.2],
    target_max_u0=0.20,
    show_ticks=True,
    inset_loc=(0.0, 0.08, 1.0, 1.0)
)

print("\nDone generating all four standalone plots!")
