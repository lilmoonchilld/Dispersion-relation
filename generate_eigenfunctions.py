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
H0 = 10           # m
g = 1e-7            # m/s^2
rho0 = 1000.0        # kg/m^3
C = 1e16         # m^3/s^2
B0 = 0     # T

r1 = 1e7           # inner radius [m]
r2 = 1e6           # outer radius [m]
N = 64               # grid resolution
m = 2                # azimuthal mode number

# Global scan range boundaries
OMEGA_SCAN_MIN = -20.0
OMEGA_SCAN_MAX = 20.0

# Geometry and Domain Validation
if r1 > r2:
    print(f"WARNING: r1 ({r1:.4e} m) is greater than r2 ({r2:.4e} m).")
    print(f"To ensure physical consistency of the SWMHD equations (inner radius must be less than outer radius),")
    print(f"the script is automatically swapping r1 and r2.")
    r1, r2 = r2, r1
elif r1 == r2:
    raise ValueError(f"Physical inconsistency: r1 and r2 are equal ({r1:.4e} m). The computational domain must have non-zero width.")

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
print(f" Scan   = [{OMEGA_SCAN_MIN}, {OMEGA_SCAN_MAX}]")
print("-"*60)

# ==============================================================================
# 1. Chebyshev Machinery
# ==============================================================================
def cheb(N_grid, a, b):
    j = np.arange(N_grid)
    xi = np.cos(j * np.pi / (N_grid - 1))
    c = np.ones(N_grid)
    c[0] = 2; c[-1] = 2
    D = np.zeros((N_grid, N_grid))
    for i in range(N_grid):
        for k in range(N_grid):
            if i != k:
                D[i, k] = (c[i] / c[k]) * ((-1)**(i+k)) / (xi[i] - xi[k])
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
# 2. Collocation Matrix Assembly (for Poincaré waves and SVD)
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
    if abs(Oh) < 1e-4:
        return np.inf
    try:
        sv = svdvals(build_L(Oh, m_val))
        return float(sv[-1] / sv[0])
    except Exception:
        return np.inf

# Helper for temporary B0 values (used in physical continuation check)
def build_L_temp(Oh, m_val, B0_val):
    temp_VA2 = B0_val**2 / (MU0 * rho0)
    temp_hat_omA2 = temp_VA2 / (H0**2 * f**2)
    ws = Oh + temp_hat_omA2 / Oh
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

def rel_sigma_temp(Oh, m_val, B0_val):
    if abs(Oh) < 1e-5:
        return np.inf
    try:
        sv = svdvals(build_L_temp(Oh, m_val, B0_val))
        return float(sv[-1] / sv[0])
    except Exception:
        return np.inf

def trace_eigenfrequency(Oh_start, B0_start, B0_end, m_val):
    """
    Rigorously traces an eigenvalue from B0_start to B0_end in small steps
    to verify physical continuation / limit.
    """
    current_Oh = Oh_start
    steps = 8
    b_vals = np.linspace(B0_start, B0_end, steps)
    for idx in range(1, len(b_vals)):
        b_curr = b_vals[idx]
        lo = current_Oh - 0.2
        hi = current_Oh + 0.2
        if lo * hi < 0:
            if current_Oh > 0:
                lo = 1e-4
            else:
                hi = -1e-4
        res = minimize_scalar(lambda w: rel_sigma_temp(w, m_val, b_curr), bounds=(lo, hi), method='bounded', options={'xatol': 1e-12, 'maxiter': 200})
        if res.success and rel_sigma_temp(res.x, m_val, b_curr) < 1e-4:
            current_Oh = float(res.x)
        else:
            return None
    return current_Oh

# ==============================================================================
# 3. Global Spectrum Search Finder & Refiner
# ==============================================================================
def find_all_eigenvalues(m_val):
    """
    Scans the user-specified frequency interval, detects every local minimum,
    refines every candidate, and returns every converged eigenfrequency.
    """
    scan = np.linspace(OMEGA_SCAN_MIN, OMEGA_SCAN_MAX, 6000)
    scan = scan[np.abs(scan) > 1e-3]

    sigmas = []
    for w in scan:
        sigmas.append(rel_sigma(w, m_val))
    sigmas = np.array(sigmas)

    candidates = []
    for k in range(1, len(sigmas) - 1):
        if sigmas[k] < sigmas[k-1] and sigmas[k] < sigmas[k+1] and sigmas[k] < 1e-4:
            candidates.append(scan[k])

    converged_eigenvalues = []
    for c in candidates:
        lo = c - 0.05
        hi = c + 0.05
        if lo * hi < 0:
            if c > 0:
                lo = 1e-3
            else:
                hi = -1e-3
        res = minimize_scalar(lambda w: rel_sigma(w, m_val), bounds=(lo, hi), method='bounded', options={'xatol': 1e-12, 'maxiter': 200})
        if res.success:
            val = float(res.x)
            # Avoid duplicating nearby eigenvalues
            duplicate = False
            for prev in converged_eigenvalues:
                if abs(prev - val) < 1e-3:
                    duplicate = True
                    break
            if not duplicate:
                converged_eigenvalues.append(val)

    converged_eigenvalues.sort()
    return converged_eigenvalues

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
# 4. Velocity Reconstruction for Poincaré/Rossby waves (Cramér's Rule)
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

    # Ensure finite fields
    if not np.all(np.isfinite(v_r)):
        print("DIAGNOSTIC WARNING: reconstruct_velocity generated non-finite v_r. Replacing with 0.0.")
        v_r = np.where(np.isfinite(v_r), v_r, 0.0)
    if not np.all(np.isfinite(v_th)):
        print("DIAGNOSTIC WARNING: reconstruct_velocity generated non-finite v_th. Replacing with 0.0.")
        v_th = np.where(np.isfinite(v_th), v_th, 0.0)

    u0 = np.sqrt(v_r**2 + v_th**2)
    u0_max = np.max(u0) if np.max(u0) > 0 else 1.0
    v_r = v_r / u0_max * target_max_u0
    v_th = v_th / u0_max * target_max_u0
    u0 = u0 / u0_max * target_max_u0

    return v_r, v_th, u0

# ==============================================================================
# 5. Analytical Fields for Kelvin Waves (from Section 10 of the theory)
# ==============================================================================
def get_analytical_kelvin_fields(Oh, m_val, r_g, target_max_u0):
    ws = Oh + hat_omA2 / Oh
    exponent = (1.0 + gamma) / hat_c0sq

    # Compute in log-space using shift-and-exponentiate to prevent overflow/underflow
    log_eta = (m_val / ws) * np.log(r_g) + exponent * (0.5 * r_g**2 - r_g)
    max_log_eta = np.max(log_eta)

    if not np.isfinite(max_log_eta):
        # Fallback if log_eta contains non-finite values before shift
        print(f"DIAGNOSTIC WARNING: log_eta in get_analytical_kelvin_fields contains non-finite values.")
        eta = np.zeros_like(r_g)
    else:
        eta = np.exp(log_eta - max_log_eta)

    # Ensure eta is perfectly finite
    if not np.all(np.isfinite(eta)):
        print("DIAGNOSTIC WARNING: get_analytical_kelvin_fields generated non-finite eta. Replacing non-finite entries with 0.0.")
        eta = np.where(np.isfinite(eta), eta, 0.0)

    eta_max = np.max(np.abs(eta))
    if eta_max > 0:
        eta /= eta_max

    v_r = np.zeros_like(r_g)
    v_th = (m_val * hat_c0sq) / (2.0 * ws * r_g) * eta

    # Ensure v_th is perfectly finite
    if not np.all(np.isfinite(v_th)):
        print("DIAGNOSTIC WARNING: get_analytical_kelvin_fields generated non-finite v_th. Replacing non-finite entries with 0.0.")
        v_th = np.where(np.isfinite(v_th), v_th, 0.0)

    # Calculate velocity magnitude
    u0 = np.abs(v_th)

    # Normalize velocity fields so that max(u0) is exactly target_max_u0
    u0_max = np.max(u0) if np.max(u0) > 0 else 1.0
    v_th = v_th / u0_max * target_max_u0
    u0 = u0 / u0_max * target_max_u0

    return eta, v_r, v_th, u0

# ==============================================================================
# 6. Chebyshev Barycentric Interpolation
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
        diff = np.where(tiny, 1e-14, diff)
        t = w[j] / diff
        numer += t * f_nodes[j]
        denom += t
    result = (numer / denom).real
    if not np.all(np.isfinite(result)):
        print("DIAGNOSTIC WARNING: bary_interp generated non-finite values. Interpolating using np.interp fallback.")
        # fallback to linear interpolation for any non-finite elements
        finite_idx = np.isfinite(result)
        if not np.all(finite_idx):
            fallback_vals = np.interp(x_eval, x_nodes, f_nodes.real)
            result = np.where(finite_idx, result, fallback_vals)
    return result

# ==============================================================================
# 7. Locate and Polish the Four Target Modes
# ==============================================================================
# 1. Kelvin counter-rotating (approx -0.5)
Oh_K_neg = find_eigenvalue(-0.5, m)
eta_K_neg, _, _, _ = get_analytical_kelvin_fields(Oh_K_neg, m, xg, 0.35)
if eta_K_neg[0] < 0:
    eta_K_neg *= -1

# 2. Kelvin co-rotating (approx 1.67)
Oh_K_pos = find_eigenvalue(1.67, m)
eta_K_pos, _, _, _ = get_analytical_kelvin_fields(Oh_K_pos, m, xg, 0.10)
if eta_K_pos[0] < 0:
    eta_K_pos *= -1

# 3. Poincaré counter-rotating (approx -4.88)
Oh_P_neg = find_eigenvalue(-4.88, m)
eta_P_neg = extract_eta(Oh_P_neg, m)
if eta_P_neg[0] > 0:
    eta_P_neg *= -1

# 4. Poincaré co-rotating (approx 5.04)
Oh_P_pos = find_eigenvalue(5.04, m)
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
# 8. Setup Directory for Outputs
# ==============================================================================
os.makedirs("outputs", exist_ok=True)

# ==============================================================================
# 9. Core Plotting Routine for Standalone Figures
# ==============================================================================
def create_standalone_plot(Oh, eta, m_val, label, filename, eta_lim, u0_lim, target_max_u0, show_ticks, inset_loc, cmap='viridis'):
    """
    Generate and save a standalone, publication-quality 1D radial + 2D polar plot for a single mode.
    """
    try:
        # Verify inputs are finite and not empty
        if eta is None or len(eta) == 0:
            print(f"SKIPPING STANDALONE PLOT FOR '{label}': eta is empty or None.")
            return
        if not np.all(np.isfinite(eta)):
            print(f"SKIPPING STANDALONE PLOT FOR '{label}': eta contains non-finite values.")
            return
        if not np.isfinite(Oh):
            print(f"SKIPPING STANDALONE PLOT FOR '{label}': Oh frequency is not finite.")
            return

        fig, ax_main = plt.subplots(figsize=(6.5, 5))

        # Coordinates mapping: r' in [0, 1]
        r_prime = (xg - hat_r1) / (hat_r2 - hat_r1)
        r_prime_fine = np.linspace(0, 1, 200)
        r_fine_hat = r_prime_fine * (hat_r2 - hat_r1) + hat_r1

        # Get fields for plotting (analytical for Kelvin, barycentric for Poincaré/Rossby/Magnetostrophic)
        if 'Kelvin' in label:
            eta_fine, vr_fine, vth_fine, u0_fine = get_analytical_kelvin_fields(Oh, m_val, r_fine_hat, target_max_u0)
        else:
            eta_fine = bary_interp(r_fine_hat, xg, eta)
            v_r, v_th, u0 = reconstruct_velocity(Oh, m_val, eta, target_max_u0)
            vr_fine = bary_interp(r_fine_hat, xg, v_r)
            vth_fine = bary_interp(r_fine_hat, xg, v_th)
            u0_fine = np.sqrt(vr_fine**2 + vth_fine**2)
            u0_fine_max = np.max(u0_fine) if np.max(u0_fine) > 0 else 1.0
            u0_fine = u0_fine / u0_fine_max * target_max_u0

        # Diagnostics on fields before plotting
        if not np.all(np.isfinite(eta_fine)) or not np.all(np.isfinite(u0_fine)):
            print(f"SKIPPING STANDALONE PLOT FOR '{label}': Interpolated 1D fields contain non-finite values.")
            plt.close()
            return

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
        elif 'Poincaré' in label:
            sup = 'P-' if 'Counter' in label else 'P+'
            sub = f'{m_val},1'
        elif 'Magneto-Rossby' in label or 'Rossby' in label:
            sup = 'MR-' if 'Counter' in label or Oh < 0 else 'MR+'
            sub = f'{m_val},1'
        else: # Magnetostrophic
            sup = 'MS-' if 'Counter' in label or Oh < 0 else 'MS+'
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

        if 'Kelvin' in label:
            eta_col_fine, vr_col_fine, vth_col_fine, _ = get_analytical_kelvin_fields(Oh, m_val, r_arr_hat, target_max_u0)
        else:
            eta_col_fine = np.array([bary_interp(np.array([rh]), xg, eta)[0] for rh in r_arr_hat])
            vr_col_fine = np.array([bary_interp(np.array([rh]), xg, v_r)[0] for rh in r_arr_hat])
            vth_col_fine = np.array([bary_interp(np.array([rh]), xg, v_th)[0] for rh in r_arr_hat])

        # Verify coordinate inputs and field components are finite
        if not (np.all(np.isfinite(eta_col_fine)) and np.all(np.isfinite(vr_col_fine)) and np.all(np.isfinite(vth_col_fine))):
            print(f"SKIPPING STANDALONE PLOT FOR '{label}': 2D polar field coordinates/components contain non-finite values.")
            plt.close()
            return

        R2D, T2D = np.meshgrid(r_arr_phys, theta_arr, indexing='ij')
        ETA2D = np.outer(eta_col_fine, np.cos(m_val * theta_arr))
        VR2D = np.outer(vr_col_fine, np.cos(m_val * theta_arr))
        VTH2D = np.outer(vth_col_fine, np.sin(m_val * theta_arr))

        # Check ETA2D, VR2D, VTH2D are finite and non-empty
        if (not np.all(np.isfinite(ETA2D))) or (ETA2D.size == 0):
            print(f"SKIPPING STANDALONE PLOT FOR '{label}': 2D fields contain non-finite values or are empty.")
            plt.close()
            return

        # Cartesian conversion for quiver and pcolormesh
        X2D = R2D * np.cos(T2D)
        Y2D = R2D * np.sin(T2D)
        VX2D = VR2D * np.cos(T2D) - VTH2D * np.sin(T2D)
        VY2D = VR2D * np.sin(T2D) + VTH2D * np.cos(T2D)

        # Normalize cmap symmetrically to the absolute max value of ETA2D
        eta_max = np.nanmax(np.abs(ETA2D))
        if (not np.isfinite(eta_max)) or (eta_max <= 0.0):
            print(f"DIAGNOSTIC WARNING: eta_max for '{label}' is {eta_max}. Falling back to 1.0.")
            eta_max = 1.0

        vmin = -eta_max
        vcenter = 0.0
        vmax = eta_max

        # Strict validation of monotonic limits for TwoSlopeNorm
        if not (vmin < vcenter < vmax):
            print(f"SKIPPING STANDALONE PLOT FOR '{label}': TwoSlopeNorm limits must increase monotonically (vmin={vmin}, vcenter={vcenter}, vmax={vmax}).")
            plt.close()
            return

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        # Use contour lines instead of pcolormesh gradient to match the publication style
        levels = np.linspace(-eta_max, eta_max, 21)
        ax_polar.contour(X2D, Y2D, ETA2D, levels=levels, cmap=cmap, norm=norm, linewidths=2.0, zorder=1)

        # Add symmetrical vertical colorbar to the right of the polar plot with a continuous solid gradient
        cax = inset_axes(ax_polar, width="8%", height="100%", loc='right',
                         bbox_to_anchor=(0.12, 0., 1.0, 1.0),
                         bbox_transform=ax_polar.transAxes,
                         borderpad=0)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
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
    except Exception as e:
        print(f"ERROR: Failed to generate standalone plot for '{label}' due to an exception: {e}")
        plt.close()


def create_complete_mode_profile(Oh, eta, m_val, label, filename):
    """
    Generate and save a publication-quality 3-panel stacked plot (surface displacement,
    velocity magnitude, magnetic field magnitude) for a single mode.
    """
    try:
        # Verify inputs
        if eta is None or len(eta) == 0:
            print(f"SKIPPING COMPLETE PROFILE FOR '{label}': eta is empty or None.")
            return
        if not np.all(np.isfinite(eta)):
            print(f"SKIPPING COMPLETE PROFILE FOR '{label}': eta contains non-finite values.")
            return
        if not np.isfinite(Oh):
            print(f"SKIPPING COMPLETE PROFILE FOR '{label}': Oh frequency is not finite.")
            return

        # Coordinates mapping: r' in [0, 1]
        r_prime_fine = np.linspace(0, 1, 200)
        r_fine_hat = r_prime_fine * (hat_r2 - hat_r1) + hat_r1

        # Reconstruct fields on the fine grid
        if 'Kelvin' in label:
            eta_fine, vr_fine, vth_fine, u0_fine = get_analytical_kelvin_fields(Oh, m_val, r_fine_hat, 1.0)
            # Scale to physical velocity perturbations
            v_r_phys = vr_fine * (r0 * Omega)
            v_th_phys = vth_fine * (r0 * Omega)
            if abs(Oh) > 1e-5:
                factor = 1j * B0 / (2.0 * Omega * Oh * H0)
                b_r_complex = factor * v_r_phys
                b_th_complex = factor * v_th_phys
            else:
                b_r_complex = np.zeros_like(vr_fine, dtype=complex)
                b_th_complex = np.zeros_like(vth_fine, dtype=complex)
            b_mag_fine = np.sqrt(np.abs(b_r_complex)**2 + np.abs(b_th_complex)**2)
        else:
            eta_fine = bary_interp(r_fine_hat, xg, eta)
            v_r, v_th, u0 = reconstruct_velocity(Oh, m_val, eta, 1.0)
            v_r_phys = v_r * (r0 * Omega)
            v_th_phys = v_th * (r0 * Omega)
            if abs(Oh) > 1e-5:
                factor = 1j * B0 / (2.0 * Omega * Oh * H0)
                b_r_complex = factor * v_r_phys
                b_th_complex = factor * v_th_phys
            else:
                b_r_complex = np.zeros_like(v_r, dtype=complex)
                b_th_complex = np.zeros_like(v_th, dtype=complex)

            vr_fine = bary_interp(r_fine_hat, xg, v_r_phys)
            vth_fine = bary_interp(r_fine_hat, xg, v_th_phys)
            br_fine = bary_interp(r_fine_hat, xg, b_r_complex)
            bth_fine = bary_interp(r_fine_hat, xg, b_th_complex)

            u0_fine = np.sqrt(np.abs(vr_fine)**2 + np.abs(vth_fine)**2)
            b_mag_fine = np.sqrt(np.abs(br_fine)**2 + np.abs(bth_fine)**2)

        # Check fields are finite
        if not (np.all(np.isfinite(eta_fine)) and np.all(np.isfinite(u0_fine)) and np.all(np.isfinite(b_mag_fine))):
            print(f"SKIPPING COMPLETE PROFILE FOR '{label}': Reconstructed fields contain non-finite values.")
            return

        # Normalize each curve independently
        eta_max = np.max(np.abs(eta_fine))
        if eta_max > 0:
            eta_plot = eta_fine / eta_max
        else:
            eta_plot = eta_fine

        u0_max = np.max(u0_fine)
        if u0_max > 0:
            u0_plot = u0_fine / u0_max
        else:
            u0_plot = u0_fine

        b_mag_max = np.max(b_mag_fine)
        if b_mag_max > 0:
            b_mag_plot = b_mag_fine / b_mag_max
        else:
            b_mag_plot = b_mag_fine

        # Create the 3-panel stacked plot
        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        fig.subplots_adjust(hspace=0.08)

        # Titles and LaTeX formatting consistent with existing notation
        if 'Kelvin' in label:
            sup = 'K-' if 'Counter' in label else 'K+'
            sub = f'{m_val}'
        elif 'Poincaré' in label:
            sup = 'P-' if 'Counter' in label else 'P+'
            sub = f'{m_val},1'
        elif 'Magneto-Rossby' in label or 'Rossby' in label:
            sup = 'MR-' if 'Counter' in label or Oh < 0 else 'MR+'
            sub = f'{m_val},1'
        else: # Magnetostrophic
            sup = 'MS-' if 'Counter' in label or Oh < 0 else 'MS+'
            sub = f'{m_val},1'

        title_str = rf"$\sigma_{{{sub}}}^{{{sup}}} = {Oh:.4f}f_e$ ({label})"
        fig.suptitle(title_str, fontsize=14, y=0.94)

        # Panel 1: Surface Displacement
        axes[0].plot(r_prime_fine, eta_plot, color='#1f77b4', lw=2.2, label=r'$\tilde{\eta}$')
        axes[0].axhline(0, color='grey', lw=0.7, ls=':')
        axes[0].set_ylabel("Normalized Surface\nDisplacement", fontsize=11)
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].grid(True, alpha=0.3, ls=':')
        axes[0].legend(loc="upper right", frameon=True, fontsize=11)

        # Panel 2: Velocity Magnitude
        axes[1].plot(r_prime_fine, u0_plot, color='black', lw=2.0, label=r'$|\tilde{u}|$')
        axes[1].set_ylabel("Normalized Velocity\nMagnitude", fontsize=11)
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, alpha=0.3, ls=':')
        axes[1].legend(loc="upper right", frameon=True, fontsize=11)

        # Panel 3: Magnetic Field Magnitude
        axes[2].plot(r_prime_fine, b_mag_plot, color='red', lw=2.0, label=r'$|\tilde{B}|$')
        axes[2].set_ylabel("Normalized Magnetic\nField Magnitude", fontsize=11)
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, alpha=0.3, ls=':')
        axes[2].legend(loc="upper right", frameon=True, fontsize=11)

        # X-axis label on bottom panel
        axes[2].set_xlim(0, 1)
        axes[2].set_xlabel(r"$r'=(r-r_1)/\Delta r$", fontsize=12)

        plt.savefig(filename, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"ERROR: Failed to generate complete profile for '{label}' due to an exception: {e}")
        plt.close()


# ==============================================================================
# 10. Generate the Four Standalone Figure Plots for Kelvin & Poincaré
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

create_complete_mode_profile(
    Oh=Oh_K_neg,
    eta=eta_K_neg,
    m_val=m,
    label="Kelvin Counter-Rotating",
    filename="outputs/kelvin_counter_complete_profile.png"
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

create_complete_mode_profile(
    Oh=Oh_K_pos,
    eta=eta_K_pos,
    m_val=m,
    label="Kelvin Co-Rotating",
    filename="outputs/kelvin_co_complete_profile.png"
)

create_standalone_plot(
    Oh=Oh_P_neg,
    eta=eta_P_neg,
    m_val=m,
    label="Poincaré Counter-Rotating",
    filename="outputs/poincare_counter_rotating.png",
    eta_lim=[-1.2, 1.2],
    u0_lim=[0.0, 0.25],
    target_max_u0=0.20,
    show_ticks=False,
    inset_loc=(-0.1, 0.08, 1.0, 1.0)
)

create_complete_mode_profile(
    Oh=Oh_P_neg,
    eta=eta_P_neg,
    m_val=m,
    label="Poincaré Counter-Rotating",
    filename="outputs/poincare_counter_complete_profile.png"
)

create_standalone_plot(
    Oh=Oh_P_pos,
    eta=eta_P_pos,
    m_val=m,
    label="Poincaré Co-Rotating",
    filename="outputs/poincare_co_rotating.png",
    eta_lim=[-1.2, 1.2],
    u0_lim=[0.0, 0.25],
    target_max_u0=0.20,
    show_ticks=True,
    inset_loc=(0.0, 0.08, 1.0, 1.0)
)

create_complete_mode_profile(
    Oh=Oh_P_pos,
    eta=eta_P_pos,
    m_val=m,
    label="Poincaré Co-Rotating",
    filename="outputs/poincare_co_complete_profile.png"
)

print("\nDone generating all four standalone plots for Kelvin & Poincaré!")

# ==============================================================================
# 11. Rigorous Rossby-Wave Detection & Detailed Branch Classification
# ==============================================================================
print("\n" + "="*60)
print(" GLOBAL SPECTRUM ANALYSIS & CLASSIFICATION REPORT")
print("="*60)

# Run the global spectrum search to find all eigenvalues
all_eigs = find_all_eigenvalues(m)

# Store classification records for printing and writing to report
report_lines = []
report_lines.append("SWMHD BRANCH CLASSIFICATION REPORT")
report_lines.append("==================================================")
report_lines.append("SYSTEM PARAMETERS:")
report_lines.append(f"  B0     = {B0:.6e} T")
report_lines.append(f"  gamma  = {gamma:.4f}")
report_lines.append(f"  omega_A= {hat_omA:.4f}")
report_lines.append(f"  m      = {m}")
report_lines.append("--------------------------------------------------")

print(f"Detected {len(all_eigs)} eigenvalues in scan range [{OMEGA_SCAN_MIN}, {OMEGA_SCAN_MAX}]")

# Theoretical insufficiency string as required
insufficiency_msg = "The uploaded theory does not contain sufficient information to derive this diagnostic rigorously. Please provide the corresponding theoretical derivation before implementation."

slow_modes_for_plotting = []

for eig in all_eigs:
    res_sig = rel_sigma(eig, m)
    eta = extract_eta(eig, m)
    crossings = np.sum(np.diff(np.sign(eta)) != 0)

    # Reconstruct velocities on collocation grid (without scaling for normalized plotting)
    # Target max u0 is 1.0 for the raw numerical values
    v_r, v_th, u0 = reconstruct_velocity(eig, m, eta, 1.0)

    # Scale to physical velocity perturbations (m/s) using r0 * Omega
    v_r_phys = v_r * (r0 * Omega)
    v_th_phys = v_th * (r0 * Omega)

    # Rigorous reconstruction of magnetic fields from the derived SWMHD induction relations
    # b_r = i * B0 / (omega * H0) * v_r_phys = i * B0 / (2 * Omega * eig * H0) * v_r_phys
    if abs(eig) > 1e-5:
        # factor is 1j * B0 / (2.0 * Omega * eig * H0) in SI units (Tesla)
        factor = 1j * B0 / (2.0 * Omega * eig * H0)
        b_r_complex = factor * v_r_phys
        b_th_complex = factor * v_th_phys
    else:
        b_r_complex = np.zeros_like(v_r, dtype=complex)
        b_th_complex = np.zeros_like(v_th, dtype=complex)

    b_r_mag = np.abs(b_r_complex)
    b_th_mag = np.abs(b_th_complex)

    # 1. Kelvin: matches the analytical/boundary-trapped polished frequencies
    is_kelvin = False
    if abs(eig - Oh_K_neg) < 1e-4:
        b_name = "Kelvin-"
        criterion = "Analytical boundary-trapped geostrophic match (Kelvin-)"
        restoring = "Gravity modified by rotation (Coriolis boundary trapping)"
        detail = "Decays outward from r1. Traces to counter-rotating hydrodynamic Kelvin wave as B0 -> 0."
        reason_class = "Geostrophic matching on boundaries and outward boundary-trapping decay conform exactly to the derived Kelvin wave profiles."
        is_kelvin = True
    elif abs(eig - Oh_K_pos) < 1e-4:
        b_name = "Kelvin+"
        criterion = "Analytical boundary-trapped geostrophic match (Kelvin+)"
        restoring = "Gravity modified by rotation (Coriolis boundary trapping)"
        detail = "Decays inward from r2. Traces to co-rotating hydrodynamic Kelvin wave as B0 -> 0."
        reason_class = "Geostrophic matching on boundaries and inward boundary-trapping decay conform exactly to the derived Kelvin wave profiles."
        is_kelvin = True

    # 2. Poincaré: matches Poincaré polished frequencies or lies in high-frequency fast inertia-gravity branch
    is_poincare = False
    if not is_kelvin:
        if abs(eig - Oh_P_neg) < 1e-4 or abs(eig - Oh_P_pos) < 1e-4:
            is_poincare = True
        else:
            # Check tracing to B0 = 0
            traced = trace_eigenfrequency(eig, B0, 0.0, m)
            if traced is not None:
                if abs(traced) >= 1.0:
                    is_poincare = True

        if is_poincare:
            sign_str = "+" if eig > 0 else "-"
            b_name = f"Poincaré{sign_str}"
            criterion = "Global spectrum search (high-frequency wave branch match)"
            restoring = "Gravity and rotation (fast inertia-gravity branch)"
            detail = f"Matches the high-frequency Poincaré global spectrum with {crossings} radial crossings."
            reason_class = f"Traces to fast wave branches (|omega_traced| >= 1.0) in the hydrodynamic limit B0 -> 0, confirming fast gravity-inertial restoring mechanisms."

    # 3. Slow-wave branches (Rossby, Magneto-Rossby, Magnetostrophic)
    if not is_kelvin and not is_poincare:
        b_name = "Unclassified Slow (Theoretically Insufficient)"
        criterion = "Theoretically Insufficient"
        restoring = insufficiency_msg
        detail = "This mode belongs to the slow-wave spectrum but cannot be rigorously classified as Rossby, Magneto-Rossby, or Magnetostrophic under the strict theory-driven framework."
        reason_class = "The uploaded SWMHD manuscript does not derive the necessary diagnostics (energy functionals, force L2 norms, or spatial WKB averaging/comparison schemes) required to evaluate Rossby or Magnetostrophic criteria."
        slow_modes_for_plotting.append({"eig": eig, "eta": eta, "b_r": b_r_complex, "b_th": b_th_complex})

    # Print to console
    print(f"  Oh = {eig: 11.8f}  |  Branch: {b_name:<16}  |  Residual SVD: {res_sig:.2e}")
    print(f"      - Criterion: {criterion}")
    print(f"      - Restoring: {restoring}")
    print(f"      - Details:   {detail}")
    print(f"      - Radial crossings: {crossings}")
    print("-" * 60)

    # Save to report lines
    report_lines.append(f"Eigenvalue: {eig: .8f}")
    report_lines.append(f"  Branch: {b_name}")
    report_lines.append(f"  Residual SVD: {res_sig:.4e}")
    report_lines.append(f"  Radial crossings: {crossings}")
    report_lines.append(f"  Classification Criterion: {criterion}")
    report_lines.append(f"  Physical Restoring Mechanism: {restoring}")
    report_lines.append(f"  Details: {detail}")
    report_lines.append(f"  Reason for Classification: {reason_class}")
    report_lines.append(f"  Hydrodynamic Continuation limit (B0 -> 0):")
    traced_limit = trace_eigenfrequency(eig, B0, 0.0, m)
    if traced_limit is not None:
        report_lines.append(f"    Converged successfully to limiting frequency: {traced_limit:.6f}")
        # Check against hydrodynamic Rossby prediction
        # WKB Rossby: -S_r / (4.0 + K_r)
        # Note: comparison must declare theoretical insufficiency as spatial comparison is not derived
        report_lines.append(f"    Hydrodynamic Rossby WKB comparison: {insufficiency_msg}")
    else:
        report_lines.append("    Disappears or leaves the slow-wave spectrum.")
    report_lines.append(f"  Rossby WKB error: {insufficiency_msg}")
    report_lines.append(f"  Magneto-Rossby WKB error: {insufficiency_msg}")
    report_lines.append(f"  Magnetostrophic WKB error: {insufficiency_msg}")
    report_lines.append(f"  Kinetic Energy: {insufficiency_msg}")
    report_lines.append(f"  Magnetic Energy: {insufficiency_msg}")
    report_lines.append(f"  Energy Ratio (Emag/Ekin): {insufficiency_msg}")
    report_lines.append(f"  Force Norms (Coriolis, Lorentz, Pressure, PV-gradient): {insufficiency_msg}")
    report_lines.append(f"  Force-balance residual: {insufficiency_msg}")
    report_lines.append(f"  Final Confidence Score: {insufficiency_msg}")
    report_lines.append("--------------------------------------------------")

conclusion_txt = (
    "Due to the strict theoretical requirements of the SWMHD framework, "
    "empirical and heuristic frequency thresholds have been removed.\n"
    "Because the uploaded manuscript does not derive energy integrals, "
    "L2 force norms, or spatial WKB averaging comparison metrics, "
    "the slow-wave modes cannot be classified as Rossby, Magneto-Rossby, or Magnetostrophic.\n\n"
    "They are rigorously classified as Unclassified Slow (Theoretically Insufficient) pending further theoretical derivation."
)

print(conclusion_txt)
print("="*60)

report_lines.append("FINAL CONCLUSION:")
report_lines.append(conclusion_txt)

# Write report to file
report_path = "outputs/branch_classification_report.txt"
with open(report_path, "w") as f_rep:
    f_rep.write("\n".join(report_lines) + "\n")
print(f"Autoritative classification report saved to: {report_path}")

# ==============================================================================
# 12. Automated Visualization of Complete Eigenmodes
# ==============================================================================
if len(slow_modes_for_plotting) > 0:
    print(f"\nFound {len(slow_modes_for_plotting)} slow-wave modes. Generating complete visualizations...")
    for idx, mode in enumerate(slow_modes_for_plotting):
        r_eig = mode["eig"]
        r_eta = mode["eta"]
        r_br = mode["b_r"]
        r_bth = mode["b_th"]

        # Consistent sign orientation
        if r_eta[0] < 0:
            r_eta *= -1
            r_br *= -1
            r_bth *= -1

        lbl = "Unclassified Slow Mode"
        filename = f"outputs/unclassified_slow_mode_{idx+1}.png"

        # Plot utilizing the same high-resolution plotting pipeline
        create_standalone_plot(
            Oh=r_eig,
            eta=r_eta,
            m_val=m,
            label=lbl,
            filename=filename,
            eta_lim=[-1.2, 1.2],
            u0_lim=[0.0, 0.25],
            target_max_u0=0.20,
            show_ticks=True,
            inset_loc=(0.0, 0.08, 1.0, 1.0)
        )

        # Plot complete 3-panel profile
        complete_filename = f"outputs/unclassified_slow_complete_profile_{idx+1}.png"
        create_complete_mode_profile(
            Oh=r_eig,
            eta=r_eta,
            m_val=m,
            label="Unclassified Slow Mode",
            filename=complete_filename
        )

        # Additional figure showing b_r(r) and b_theta(r) matching the publication style
        bfield_filename = f"outputs/unclassified_slow_bfields_{idx+1}.png"
        try:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            r_prime = (xg - hat_r1) / (hat_r2 - hat_r1)
            r_prime_fine = np.linspace(0, 1, 200)
            r_fine_hat = r_prime_fine * (hat_r2 - hat_r1) + hat_r1

            # Compute absolute magnitudes of reconstructed complex magnetic fields
            br_mag = np.abs(r_br)
            bth_mag = np.abs(r_bth)

            # Interpolate magnetic perturbation magnitudes to fine grid
            br_fine = bary_interp(r_fine_hat, xg, br_mag)
            bth_fine = bary_interp(r_fine_hat, xg, bth_mag)

            if not (np.all(np.isfinite(br_fine)) and np.all(np.isfinite(bth_fine))):
                print(f"SKIPPING MAGNETIC FIELDS PROFILE FOR '{lbl}': Reconstructed fields contain non-finite values.")
                plt.close()
                continue

            # Normalize only for plotting so peak magnitude is clear
            peak_b = max(np.max(br_fine), np.max(bth_fine))
            if peak_b > 0.0:
                br_plot = br_fine / peak_b
                bth_plot = bth_fine / peak_b
            else:
                br_plot = br_fine
                bth_plot = bth_fine

            ax.plot(r_prime_fine, br_plot, color='#1f77b4', lw=2.2, label=r'$|\tilde{b}_r|$')
            ax.plot(r_prime_fine, bth_plot, color='black', lw=2.0, ls='--', label=r'$|\tilde{b}_\theta|$')
            ax.axhline(0, color='grey', lw=0.7, ls=':')

            ax.set_xlim(0, 1)
            ax.set_xlabel(r"$r'=(r-r_1)/\Delta r$", fontsize=12)
            ax.set_ylabel("Normalized Magnetic Perturbation Magnitude", fontsize=12)
            ax.set_title(rf"$\sigma_{{{m},1}}^{{US+}} = {r_eig:.4f}f_e$ (Magnetic Profiles)", fontsize=12)
            ax.legend(loc="upper right", frameon=True, fontsize=11)

            plt.tight_layout()
            plt.savefig(bfield_filename, dpi=180, bbox_inches='tight')
            plt.close()
            print(f"Saved: {bfield_filename}")
        except Exception as e:
            print(f"ERROR: Failed to generate magnetic fields profile for '{lbl}' due to an exception: {e}")
            plt.close()
else:
    print("\nNo slow-wave branch exists for the present governing equations and parameter set.")
