"""
plot_eigenfunctions.py

SWMHD eigenfunction structure figures generator.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import svd, svdvals
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

# =========================================================================
# Publication Style Settings
# =========================================================================
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

# Preset configurations
PRESETS = {
    "general": {
        "C": 1e10,
        "B0": 2e-5,
    },
    "reference": {
        "C": 1.882e9,
        "B0": 8.8623e-4,
    }
}

# Defaults
MU0 = 4.0 * np.pi * 1e-7
Omega = 0.5e-4       # rad/s
H0 = 500.0           # m
g = 9.81             # m/s^2
rho0 = 1000.0        # kg/m^3
r1 = 0.5e6           # inner radius [m]
r2 = 1.0e6           # outer radius [m]
N_col = 32           # Default grid resolution

def get_parameters(preset_name=None, C_val=None, B0_val=None):
    """
    Computes derived and dimensionless parameters based on preset or specific values.
    """
    p_C = C_val
    p_B0 = B0_val

    if preset_name is not None and preset_name in PRESETS:
        p_C = PRESETS[preset_name]["C"]
        p_B0 = PRESETS[preset_name]["B0"]
    elif p_C is None or p_B0 is None:
        p_C = PRESETS["general"]["C"]
        p_B0 = PRESETS["general"]["B0"]

    r0 = 0.5 * (r1 + r2)
    VA2 = p_B0**2 / (MU0 * rho0)
    omA2 = VA2 / H0**2
    omA = np.sqrt(omA2)
    c0sq = g * H0
    c0 = np.sqrt(c0sq)
    f_scale = 2.0 * Omega

    hat_r1 = r1 / r0
    hat_r2 = r2 / r0
    hat_c0sq = c0sq / (Omega**2 * r0**2)
    hat_VA2 = VA2 / (Omega**2 * r0**2)
    hat_omA = np.sqrt(VA2) / (f_scale * H0)
    hat_omA2 = hat_omA**2
    gamma_val = 2.0 * p_C / (Omega**2 * r0**3)

    return {
        "C": p_C,
        "B0": p_B0,
        "r0": r0,
        "VA2": VA2,
        "omA": omA,
        "c0sq": c0sq,
        "c0": c0,
        "f_scale": f_scale,
        "hat_r1": hat_r1,
        "hat_r2": hat_r2,
        "hat_c0sq": hat_c0sq,
        "hat_VA2": hat_VA2,
        "hat_omA": hat_omA,
        "hat_omA2": hat_omA2,
        "gamma": gamma_val,
    }

def cheb(N, r_min, r_max):
    """
    Robust Chebyshev grid and differentiation matrix generator.
    """
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
    sc = 2.0 / (r_max - r_min)
    D1 = sc * D
    D2 = D1 @ D1
    rp = 0.5 * (r_min + r_max) + 0.5 * (r_max - r_min) * xi
    rp = rp[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]
    return rp, D1, D2

def omega_star_hat(hat_omega, p):
    """
    Dimensionless modified frequency hat_omega_* = hat_omega + hat_omega_A^2 / hat_omega.
    """
    return hat_omega + p["hat_omA2"] / hat_omega

def build_L(hat_omega, m_val, x_grid, D1_mat, D2_mat, p):
    """
    Builds the collocation matrix L(hat_omega, m_val).
    """
    ws_hat = omega_star_hat(hat_omega, p)
    Pv = 1.0 / x_grid - (1.0 + p["gamma"]) * (x_grid - 1.0) / p["hat_c0sq"]

    term1 = 4.0 * hat_omega * (ws_hat**2 - 1.0) / (ws_hat * p["hat_c0sq"])
    term2 = - (m_val**2) / (x_grid**2)
    term3 = - (1.0 + p["gamma"]) * (2.0 * x_grid - 1.0) / (p["hat_c0sq"] * x_grid)
    term4 = - m_val * (1.0 + p["gamma"]) * (x_grid - 1.0) / (ws_hat * p["hat_c0sq"] * x_grid)
    Qv = term1 + term2 + term3 + term4

    L = D2_mat + np.diag(Pv) @ D1_mat + np.diag(Qv)

    # Boundary conditions
    N = len(x_grid)
    for idx in [0, N - 1]:
        rb = x_grid[idx]
        bc_eta_coeff = - (ws_hat * (1.0 + p["gamma"]) * (rb - 1.0) + m_val * p["hat_c0sq"] / rb)
        L[idx, :] = ws_hat * p["hat_c0sq"] * D1_mat[idx, :] + bc_eta_coeff * np.eye(N)[idx]

    return L

def rel_sigma(hat_omega, m_val, x_grid, D1_mat, D2_mat, p):
    """
    Computes the relative smallest singular value: sigma_min / sigma_max.
    """
    if abs(hat_omega) < 0.02:
        return np.inf
    try:
        L = build_L(hat_omega, m_val, x_grid, D1_mat, D2_mat, p)
        sv = svdvals(L, overwrite_a=True, check_finite=False)
        return float(sv[-1] / sv[0])
    except Exception:
        return np.inf

def extract_eta(hat_omega, m_val, x_grid, D1_mat, D2_mat, p):
    """
    Extracts the normalized displacement eigenfunction and returns (eta, sigma_min).
    """
    L = build_L(hat_omega, m_val, x_grid, D1_mat, D2_mat, p)
    _, s, Vh = svd(L, full_matrices=False, overwrite_a=True, check_finite=False)
    eta = Vh[-1, :].copy()
    norm = np.max(np.abs(eta))
    if norm > 0:
        eta /= norm
    return eta, float(s[-1])

def reconstruct_velocity(hat_omega, m_val, eta, x_grid, D1_mat, p):
    """
    Reconstructs dimensionless velocity components v_r and v_th using Cramer's rule.
    Returns (v_r, v_th, u0) normalized such that max(u0) = 1.0.
    """
    ws = omega_star_hat(hat_omega, p)
    D_star = ws**2 - 4.0

    deta_dx = D1_mat @ eta

    RHS_r = p["hat_c0sq"] * deta_dx - (1.0 + p["gamma"]) * (x_grid - 1.0) * eta - m_val * (1.0 + p["gamma"]) * (x_grid - 1.0) * eta / (ws * x_grid)
    RHS_th = (m_val / x_grid) * p["hat_c0sq"] * eta

    v_r = (ws * RHS_r + 2.0 * RHS_th) / D_star
    v_th = (ws * RHS_th - 2.0 * RHS_r) / D_star

    u0 = np.sqrt(v_r**2 + v_th**2)
    u0_max = np.max(u0)
    if u0_max > 0:
        v_r /= u0_max
        v_th /= u0_max
        u0 /= u0_max

    return v_r, v_th, u0

def wkb_branches(m_val, n_radial, p):
    """
    Analytical WKB predictions for a given azimuthal and radial order.
    Returns predicted frequencies for different branches.
    """
    hat_kr = n_radial * np.pi / (p["hat_r2"] - p["hat_r1"])

    # 1. Poincaré (MP+, MP-) and Magnetostrophic (MS) at r=1
    r_val = 1.0
    R_1 = p["gamma"] * (2 * r_val - 1) / r_val
    kappa2_1 = hat_kr**2 + m_val**2 / r_val**2
    K_1 = p["hat_c0sq"] * kappa2_1 + R_1

    A2 = 2.0 * p["hat_omA2"] - 1.0 - 0.25 * K_1
    A0 = p["hat_omA2"] * (p["hat_omA2"] - 0.25 * K_1)
    disc = A2**2 - 4.0 * A0
    if disc < 0:
        disc = 0.0
    X1 = (-A2 + np.sqrt(disc)) / 2.0
    oMP_p = np.sqrt(max(X1, 0)) if X1 >= 0 else np.nan
    oMP_m = -np.sqrt(max(X1, 0)) if X1 >= 0 else np.nan

    # 2. Magneto-Kelvin waves at boundaries
    # Inner boundary
    arg_in = m_val**2 * p["hat_c0sq"] / p["hat_r1"]**2 - 4.0 * p["hat_omA2"]
    oMK_in = -0.5 * np.sqrt(arg_in) if arg_in >= 0 else np.nan
    # Outer boundary
    arg_out = m_val**2 * p["hat_c0sq"] / p["hat_r2"]**2 - 4.0 * p["hat_omA2"]
    oMK_out = 0.5 * np.sqrt(arg_out) if arg_out >= 0 else np.nan

    return oMP_p, oMP_m, oMK_in, oMK_out

def select_branch_modes(m_val, p, N_grid=32):
    """
    Finds the eigenvalues of the system and assigns them to the 4 target branches:
    - Kelvin+ (MK_out)
    - Kelvin− (MK_in)
    - Poincaré+ (MP_p, n=1)
    - Poincaré− (MP_m, n=1)

    Returns a dictionary of structure:
    {
      "Kelvin+": {"omega": omega_val, "type": "Kelvin+"},
      "Kelvin−": {"omega": omega_val, "type": "Kelvin−"},
      ...
    }
    """
    x_grid, D1_mat, D2_mat = cheb(N_grid, p["hat_r1"], p["hat_r2"])

    # Perform a dense scanning of frequencies to locate candidate eigenvalues
    scan_range = np.linspace(-45, 45, 2000)
    # Exclude very small frequencies near 0 to avoid singularities
    scan_range = scan_range[np.abs(scan_range) > 0.05]

    # Compute relative smallest singular values over the scan range
    r_sigmas = []
    for w in scan_range:
        r_sigmas.append(rel_sigma(w, m_val, x_grid, D1_mat, D2_mat, p))
    r_sigmas = np.array(r_sigmas)

    # Detect local minima of the relative smallest singular value
    candidates = []
    for k in range(1, len(r_sigmas) - 1):
        if r_sigmas[k] < r_sigmas[k - 1] and r_sigmas[k] < r_sigmas[k + 1] and r_sigmas[k] < 1e-4:
            candidates.append(scan_range[k])

    # Polish candidates to find exact eigenfrequencies
    eigenvalues = []
    for w_c in candidates:
        lo = w_c - 0.5
        hi = w_c + 0.5
        # Avoid zero boundary crossing
        if lo * hi < 0:
            lo = 0.05 if w_c > 0 else -hi
            hi = w_c + 0.5
        try:
            res = minimize_scalar(
                lambda w: rel_sigma(w, m_val, x_grid, D1_mat, D2_mat, p),
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 200}
            )
            if res.success:
                val = float(res.x)
                if not any(abs(val - ex) < 1e-4 for ex in eigenvalues):
                    eigenvalues.append(val)
        except Exception:
            pass

    eigenvalues = sorted(eigenvalues)

    # WKB predictions for branch assignment
    oMP_p, oMP_m, oMK_in, oMK_out = wkb_branches(m_val, 1, p)

    # Global minimum distance matching for assignment
    wkb_preds = {}
    if not np.isnan(oMK_out):
        wkb_preds["Kelvin+"] = oMK_out
    if not np.isnan(oMK_in):
        wkb_preds["Kelvin−"] = oMK_in
    if not np.isnan(oMP_p):
        wkb_preds["Poincaré+"] = oMP_p
    if not np.isnan(oMP_m):
        wkb_preds["Poincaré−"] = oMP_m

    assigned_modes = {}
    used_eigs = set()

    # Assign the closest eigenvalue for each predicted branch
    for branch_name, pred_val in wkb_preds.items():
        if len(eigenvalues) == 0:
            continue
        # Sort eigenvalues by distance to the predicted value
        sorted_eigs = sorted(eigenvalues, key=lambda x: abs(x - pred_val))
        # Select the closest one that wasn't already used
        for eig in sorted_eigs:
            if eig not in used_eigs:
                assigned_modes[branch_name] = {
                    "omega": eig,
                    "type": branch_name
                }
                used_eigs.add(eig)
                break

    return assigned_modes

def barycentric_weights(x_nodes):
    """
    Computes barycentric weights for stable high-accuracy interpolation on Chebyshev nodes.
    """
    N = len(x_nodes)
    w = np.ones(N)
    w[::2] = -1.0
    w[0] /= 2.0
    w[-1] /= 2.0
    return w

def interpolate_to_fine_grid(x_eval, x_nodes, f_nodes):
    """
    Performs high-accuracy Barycentric Chebyshev interpolation from Chebyshev nodes to evaluative grid.
    """
    N_eval = len(x_eval)
    N_nodes = len(x_nodes)
    w = barycentric_weights(x_nodes)

    numerator = np.zeros(N_eval, dtype=complex)
    denominator = np.zeros(N_eval)

    for j in range(N_nodes):
        diff = x_eval - x_nodes[j]
        # Avoid division by zero at grid nodes
        tiny = np.abs(diff) < 1e-14
        diff[tiny] = 1e-14
        t = w[j] / diff
        numerator += t * f_nodes[j]
        denominator += t

    return (numerator / denominator).real

def build_annular_field(hat_omega, eta_nodes, m_val, x_nodes, v_r_nodes, v_th_nodes, p, N_theta=200, N_r=100):
    """
    Builds the 2D annular field for the polar inset.
    Returns:
    - R2D, T2D: Polar grid
    - X2D, Y2D: Cartesian grid coordinates
    - ETA2D: free-surface field eta(r, theta)
    - VX2D, VY2D: Cartesian velocity vector components
    """
    theta = np.linspace(0, 2.0 * np.pi, N_theta, endpoint=False)
    r_fine = np.linspace(p["hat_r1"], p["hat_r2"], N_r)

    # Interpolate displacement and velocities onto fine radial grid
    eta_fine = interpolate_to_fine_grid(r_fine, x_nodes, eta_nodes)
    vr_fine = interpolate_to_fine_grid(r_fine, x_nodes, v_r_nodes)
    vth_fine = interpolate_to_fine_grid(r_fine, x_nodes, v_th_nodes)

    # Create 2D mesh grids
    R2D, T2D = np.meshgrid(r_fine * p["r0"], theta, indexing='ij')

    # 2D Normal modes:
    # eta(r, theta) = eta_fine(r) * cos(m*theta)
    # v_r(r, theta) = v_r_fine(r) * cos(m*theta)
    # v_th(r, theta) = v_th_fine(r) * sin(m*theta)
    ETA2D = np.outer(eta_fine, np.cos(m_val * theta))
    VR2D = np.outer(vr_fine, np.cos(m_val * theta))
    VTH2D = np.outer(vth_fine, np.sin(m_val * theta))

    # Transform velocities into Cartesian coordinates
    X2D = R2D * np.cos(T2D)
    Y2D = R2D * np.sin(T2D)

    VX2D = VR2D * np.cos(T2D) - VTH2D * np.sin(T2D)
    VY2D = VR2D * np.sin(T2D) + VTH2D * np.cos(T2D)

    return R2D, T2D, X2D, Y2D, ETA2D, VX2D, VY2D

def plot_profile(branch_name, hat_omega, x_nodes, eta_nodes, u0_nodes, p, filepath):
    """
    Plots the radial eigenfunction profile using dual y-axes.
    """
    # Create fine grid for smooth plots
    r_prime_fine = np.linspace(0.0, 1.0, 300)
    # Maps r_prime fine to actual hat_r values
    r_hat_fine = r_prime_fine * (p["hat_r2"] - p["hat_r1"]) + p["hat_r1"]

    # Interpolate values
    eta_fine = interpolate_to_fine_grid(r_hat_fine, x_nodes, eta_nodes)
    u0_fine = interpolate_to_fine_grid(r_hat_fine, x_nodes, u0_nodes)

    # Re-normalize to exact max value of 1.0
    if np.max(np.abs(eta_fine)) > 0:
        eta_fine /= np.max(np.abs(eta_fine))
    if np.max(u0_fine) > 0:
        u0_fine /= np.max(u0_fine)

    fig, ax_eta = plt.subplots(figsize=(7, 5))
    ax_u0 = ax_eta.twinx()

    # Displacement plot (left axis, blue)
    ax_eta.plot(r_prime_fine, eta_fine, color='#1f77b4', lw=2.2, label=r'$\eta_0$')
    ax_eta.set_xlabel(r"$r' = (r - r_1) / \Delta r$", fontsize=12)
    ax_eta.set_ylabel(r"$\eta_0$", color='#1f77b4', fontsize=12)
    ax_eta.tick_params(axis='y', labelcolor='#1f77b4')
    ax_eta.set_ylim(-1.15, 1.15)
    ax_eta.axhline(0, color='grey', lw=0.7, ls=':')

    # Velocity magnitude plot (right axis, black)
    ax_u0.plot(r_prime_fine, u0_fine, color='black', lw=2.0, ls='-', label=r'$u_0$')
    ax_u0.set_ylabel(r"$u_0$", color='black', fontsize=12)
    ax_u0.tick_params(axis='y', labelcolor='black')
    ax_u0.set_ylim(0, 1.15)

    # Text annotation and title formatting
    formatted_name = branch_name.replace("minus", "(−)").replace("plus", "(+)")
    ax_eta.set_title(f"{formatted_name}\n" + r"$\hat\omega = $" + f"{hat_omega:.4f}", fontsize=13)

    plt.tight_layout()
    plt.savefig(filepath + ".pdf", dpi=180, bbox_inches='tight')
    plt.savefig(filepath + ".png", dpi=180, bbox_inches='tight')
    plt.close()

def plot_contour(branch_name, hat_omega, m_val, R2D, T2D, X2D, Y2D, ETA2D, VX2D, VY2D, p, filepath):
    """
    Plots the 2D annular contour plot of surface displacement and velocity vectors.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    # Displacement contour pcolormesh using symmetric RdBu_r centered at 0
    eta_max = np.max(np.abs(ETA2D)) if np.max(np.abs(ETA2D)) > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-eta_max, vcenter=0.0, vmax=eta_max)

    contour = ax.pcolormesh(X2D, Y2D, ETA2D, cmap='RdBu_r', norm=norm, shading='auto', zorder=1)

    # Adaptive subsampling for target of ~300 arrows
    # Total points = N_r * N_theta
    N_r, N_theta = ETA2D.shape
    total_points = N_r * N_theta
    target_arrows = 300

    # Choose skip intervals
    skip_r = int(np.sqrt(total_points / target_arrows))
    skip_t = skip_r

    # Adjust to ensure it is at least 1
    skip_r = max(1, skip_r)
    skip_t = max(1, skip_t)

    # Subsampled grid
    Xq = X2D[::skip_r, ::skip_t]
    Yq = Y2D[::skip_r, ::skip_t]
    UXq = VX2D[::skip_r, ::skip_t]
    UYq = VY2D[::skip_r, ::skip_t]

    # Normalize quiver arrow lengths for consistent presentation
    speed = np.sqrt(UXq**2 + UYq**2) + 1e-15
    ax.quiver(Xq, Yq, UXq / speed, UYq / speed, scale=35, width=0.0035, color='black', alpha=0.75, zorder=2)

    # Draw inner and outer solid boundaries
    r_phys_1 = p["hat_r1"] * p["r0"]
    r_phys_2 = p["hat_r2"] * p["r0"]
    circle_angles = np.linspace(0.0, 2.0 * np.pi, 400)
    for rcirc in [r_phys_1, r_phys_2]:
        ax.plot(rcirc * np.cos(circle_angles), rcirc * np.sin(circle_angles), 'k-', lw=1.5, zorder=3)

    # Title
    formatted_name = branch_name.replace("minus", "(−)").replace("plus", "(+)")
    ax.set_title(f"{formatted_name}\n" + r"$\hat\omega = $" + f"{hat_omega:.4f}", fontsize=13)

    # Add horizontal colorbar
    cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.06, shrink=0.7)
    cbar.set_label(r"$\eta$", fontsize=12)

    plt.tight_layout()
    plt.savefig(filepath + ".pdf", dpi=180, bbox_inches='tight')
    plt.savefig(filepath + ".png", dpi=180, bbox_inches='tight')
    plt.close()

def save_branch_figures(preset_name="general", m_val=2, N_grid=32):
    """
    Main orchestrator for SWMHD eigenfigure generation.
    """
    os.makedirs("outputs", exist_ok=True)
    p = get_parameters(preset_name=preset_name)

    print("=" * 65)
    print(f" Starting figure generation for Preset: '{preset_name}' (m={m_val})")
    print("=" * 65)

    # Find mode eigenvalues
    print("Finding and matching eigenvalues to wave branches...")
    assigned_modes = select_branch_modes(m_val, p, N_grid=N_grid)

    # Set up grid
    x_nodes, D1_mat, D2_mat = cheb(N_grid, p["hat_r1"], p["hat_r2"])

    # Define file suffix and naming patterns
    name_map = {
        "Kelvin+": "Kelvin_plus",
        "Kelvin−": "Kelvin_minus",
        "Poincaré+": "Poincare_plus",
        "Poincaré−": "Poincare_minus"
    }

    for b_key, b_info in assigned_modes.items():
        name_prefix = name_map[b_key]
        omega_val = b_info["omega"]
        print(f"Processing branch: {b_key} (omega={omega_val:.4f})")

        # Extract η and reconstruct velocity on collocation nodes
        eta_nodes, _ = extract_eta(omega_val, m_val, x_nodes, D1_mat, D2_mat, p)
        v_r_nodes, v_th_nodes, u0_nodes = reconstruct_velocity(omega_val, m_val, eta_nodes, x_nodes, D1_mat, p)

        # 1. Plot radial profiles
        profile_path = os.path.join("outputs", f"{name_prefix}_profile")
        plot_profile(b_key, omega_val, x_nodes, eta_nodes, u0_nodes, p, profile_path)

        # 2. Build 2D annular fields and plot contour
        R2D, T2D, X2D, Y2D, ETA2D, VX2D, VY2D = build_annular_field(
            omega_val, eta_nodes, m_val, x_nodes, v_r_nodes, v_th_nodes, p, N_theta=200, N_r=100
        )
        contour_path = os.path.join("outputs", f"{name_prefix}_contour")
        plot_contour(b_key, omega_val, m_val, R2D, T2D, X2D, Y2D, ETA2D, VX2D, VY2D, p, contour_path)

    print("\nAll figures generated successfully inside 'outputs/' directory.")

if __name__ == "__main__":
    import sys
    preset = "general"
    if len(sys.argv) > 1:
        preset = sys.argv[1]
    save_branch_figures(preset_name=preset)
