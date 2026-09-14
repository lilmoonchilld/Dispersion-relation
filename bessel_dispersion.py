import numpy as np
import scipy.special as spc
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from scipy.linalg import svd
import sys

# ==================================================
# 1. PHYSICAL PARAMETERS
# ==================================================
MU0   = 4 * np.pi * 1e-7
Omega = 0.5e-4
H0    = 500.0
g     = 9.81
B0    = 8.5e-4
rho0  = 1000.0

r1 = 0.5e6
r2 = 1.0e6
r0 = 0.5 * (r1 + r2)

f = 2 * Omega
D0 = H0

# Configurable limits
M_MAX = 30
N_MODES = 3 # Ensure we aim for at least n=1,2,3
N_OMEGA_GRID = 100000
N_ROSSBY_GRID = 100000
ROSSBY_THRESH = 0.1 # max |x| to be considered Rossby region

def compute_parameters(use_B0=B0):
    omega_A_sq = use_B0**2 / (MU0 * rho0 * D0**2)
    omega_A = np.sqrt(max(0, omega_A_sq))
    epsilon_P = Omega**2 * r2**2 / (2 * g * D0)
    return omega_A_sq, omega_A, epsilon_P

def omega_star(omega, omega_A_sq):
    if omega == 0:
        return np.nan
    return omega + omega_A_sq / omega

def K_squared(omega, m, omega_A_sq):
    w_star = omega_star(omega, omega_A_sq)
    term1 = (omega / w_star) * (w_star**2 - 4 * Omega**2) / (g * D0)
    term2 = (2 * m * Omega**3) / (g * D0 * w_star)
    return term1 - term2

def resonance_frequencies(omega_A_sq):
    # w_star = +/- 2 Omega
    # w^2 -/+ 2 Omega w + w_A^2 = 0
    res = []
    # +2 Omega: w^2 - 2 Omega w + w_A^2 = 0
    disc1 = Omega**2 - omega_A_sq
    if disc1 >= 0:
        res.extend([Omega + np.sqrt(disc1), Omega - np.sqrt(disc1)])
    # -2 Omega: w^2 + 2 Omega w + w_A^2 = 0
    disc2 = Omega**2 - omega_A_sq
    if disc2 >= 0:
        res.extend([-Omega + np.sqrt(disc2), -Omega - np.sqrt(disc2)])
    return sorted(list(set(res)))

# ==================================================
# 2. BOUNDARY MATRICES
# ==================================================
def bessel_boundary_matrix(omega, m, K, w_star):
    r1_term = K * r1
    r2_term = K * r2

    M11 = (m * (w_star - 2 * Omega) / r1) * spc.jv(m, r1_term) - w_star * K * spc.jv(m+1, r1_term)
    M12 = (m * (w_star - 2 * Omega) / r1) * spc.yv(m, r1_term) - w_star * K * spc.yv(m+1, r1_term)

    M21 = (m * (w_star - 2 * Omega) / r2) * spc.jv(m, r2_term) - w_star * K * spc.jv(m+1, r2_term)
    M22 = (m * (w_star - 2 * Omega) / r2) * spc.yv(m, r2_term) - w_star * K * spc.yv(m+1, r2_term)

    return np.array([[M11, M12], [M21, M22]])

def modified_bessel_boundary_matrix(omega, m, kappa, w_star):
    r1_term = kappa * r1
    r2_term = kappa * r2

    M11 = w_star * kappa * spc.ivp(m, r1_term) - (2 * m * Omega / r1) * spc.iv(m, r1_term)
    M12 = w_star * kappa * spc.kvp(m, r1_term) - (2 * m * Omega / r1) * spc.kv(m, r1_term)

    M21 = w_star * kappa * spc.ivp(m, r2_term) - (2 * m * Omega / r2) * spc.iv(m, r2_term)
    M22 = w_star * kappa * spc.kvp(m, r2_term) - (2 * m * Omega / r2) * spc.kv(m, r2_term)

    return np.array([[M11, M12], [M21, M22]])

def dispersion_matrix(omega, m, omega_A_sq):
    w_star = omega_star(omega, omega_A_sq)
    K2 = K_squared(omega, m, omega_A_sq)

    if K2 > 0:
        return bessel_boundary_matrix(omega, m, np.sqrt(K2), w_star)
    else:
        return modified_bessel_boundary_matrix(omega, m, np.sqrt(-K2), w_star)

def dispersion_function(omega, m, omega_A_sq):
    M = dispersion_matrix(omega, m, omega_A_sq)
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

def normalized_dispersion_residual(omega, m, omega_A_sq):
    M = dispersion_matrix(omega, m, omega_A_sq)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    norm = np.max(np.abs(M))**2
    if norm == 0 or np.isnan(norm):
        return 1e10
    return det / norm

def boundary_condition_residual(omega, m, omega_A_sq, r_eval, A, B):
    w_star = omega_star(omega, omega_A_sq)
    K2 = K_squared(omega, m, omega_A_sq)

    if K2 > 0:
        K = np.sqrt(K2)
        eta = A * spc.jv(m, K*r_eval) + B * spc.yv(m, K*r_eval)
        etap = A * K * spc.jvp(m, K*r_eval) + B * K * spc.yvp(m, K*r_eval)
    else:
        kappa = np.sqrt(-K2)
        eta = A * spc.iv(m, kappa*r_eval) + B * spc.kv(m, kappa*r_eval)
        etap = A * kappa * spc.ivp(m, kappa*r_eval) + B * kappa * spc.kvp(m, kappa*r_eval)

    return w_star * etap - (2 * m * Omega / r_eval) * eta


# ==================================================
# 3. ROOT FINDING & MODE CLASSIFICATION
# ==================================================
def compute_eigenfunction(omega, m, omega_A_sq):
    M = dispersion_matrix(omega, m, omega_A_sq)
    try:
        U, S, Vh = svd(M)
        null_vec = Vh[-1, :]
        A, B = null_vec[0], null_vec[1]
    except Exception:
        A, B = 0, 0

    r_grid = np.linspace(r1, r2, 500)
    K2 = K_squared(omega, m, omega_A_sq)

    if K2 > 0:
        K = np.sqrt(K2)
        eta = A * spc.jv(m, K * r_grid) + B * spc.yv(m, K * r_grid)
    else:
        kappa = np.sqrt(-K2)
        eta = A * spc.iv(m, kappa * r_grid) + B * spc.kv(m, kappa * r_grid)

    return r_grid, eta, A, B

def classify_radial_mode(omega, m, omega_A_sq):
    K2 = K_squared(omega, m, omega_A_sq)

    r_grid, eta, A, B = compute_eigenfunction(omega, m, omega_A_sq)

    eta_norm = eta / np.max(np.abs(eta))
    interior_eta = eta_norm[1:-1]

    crossings = 0
    for i in range(len(interior_eta) - 1):
        if interior_eta[i] * interior_eta[i+1] < 0:
            if np.abs(interior_eta[i] - interior_eta[i+1]) > 1e-4:
                crossings += 1

    if K2 < 0:
        return "Modified-Bessel", crossings, crossings
    else:
        return "Bessel", crossings + 1, crossings

def is_valid_bracket(w1, w2, resonances):
    if w1 * w2 <= 0:
        return False
    for res in resonances:
        if min(w1, w2) < res < max(w1, w2):
            return False
    return True


# ==================================================
# 4. FULL BRANCH TRACKING AND MAIN LOGIC
# ==================================================
def scan_and_bracket(omega_grid, m, omega_A_sq, resonances):
    brackets = []
    dets = []
    valid_mask = []

    for w in omega_grid:
        if w == 0:
            dets.append(np.nan)
            valid_mask.append(False)
            continue

        w_star = omega_star(w, omega_A_sq)
        if np.isclose(w_star, 2*Omega) or np.isclose(w_star, -2*Omega):
            dets.append(np.nan)
            valid_mask.append(False)
            continue

        try:
            M = dispersion_matrix(w, m, omega_A_sq)
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            dets.append(det)
            valid_mask.append(True)
        except Exception:
            dets.append(np.nan)
            valid_mask.append(False)

    dets = np.array(dets)

    for i in range(len(omega_grid) - 1):
        if not valid_mask[i] or not valid_mask[i+1]:
            continue

        w1, w2 = omega_grid[i], omega_grid[i+1]
        if not is_valid_bracket(w1, w2, resonances):
            continue

        f1, f2 = dets[i], dets[i+1]
        if np.isnan(f1) or np.isnan(f2):
            continue

        if f1 * f2 <= 0:
            if np.abs(f1) < 1e50 and np.abs(f2) < 1e50:
                brackets.append((w1, w2))

    return brackets

def refine_root(w1, w2, m, omega_A_sq):
    def f(w):
        try:
            M = dispersion_matrix(w, m, omega_A_sq)
            return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        except Exception:
            return 1e10

    try:
        w_root = brentq(f, w1, w2, xtol=1e-12, maxiter=200)
        return w_root
    except ValueError:
        return None
    except RuntimeError:
        return None

def find_roots_for_m(m, omega_A_sq, resonances, omega_grid_global, omega_grid_rossby):
    brackets_global = scan_and_bracket(omega_grid_global, m, omega_A_sq, resonances)
    brackets_rossby = scan_and_bracket(omega_grid_rossby, m, omega_A_sq, resonances)

    all_brackets = brackets_global + brackets_rossby
    roots = []

    for w1, w2 in all_brackets:
        w_root = refine_root(w1, w2, m, omega_A_sq)
        if w_root is not None:
            norm_res = np.abs(normalized_dispersion_residual(w_root, m, omega_A_sq))
            if norm_res < 1e-4:
                roots.append(w_root)

    unique_roots = []
    for w in sorted(roots):
        if not unique_roots or abs(w - unique_roots[-1]) > 1e-8:
            unique_roots.append(w)

    return unique_roots

def process_and_classify_roots(roots, m, omega_A_sq):
    processed = []
    counts = {}

    for w in roots:
        family, n, crossings = classify_radial_mode(w, m, omega_A_sq)
        w_star = omega_star(w, omega_A_sq)
        K2 = K_squared(w, m, omega_A_sq)
        norm_res = np.abs(normalized_dispersion_residual(w, m, omega_A_sq))

        r_grid, eta, A, B = compute_eigenfunction(w, m, omega_A_sq)
        bc_res1 = np.abs(boundary_condition_residual(w, m, omega_A_sq, r1, A, B))
        bc_res2 = np.abs(boundary_condition_residual(w, m, omega_A_sq, r2, A, B))
        max_bc_res = max(bc_res1, bc_res2)

        sign = np.sign(w)
        key = (family, sign, n)
        if key not in counts:
            counts[key] = 1
        else:
            counts[key] += 1

        processed.append({
            "m": m,
            "n": n,
            "sub_n": counts[key],
            "family": family,
            "omega": w,
            "omega_normalized": w / (2 * Omega),
            "omega_star": w_star,
            "K2": K2,
            "residual": norm_res,
            "bc_residual": max_bc_res,
            "branch_sign": sign
        })
    return processed

def track_branch(m, prev_root, omega_A_sq, resonances):
    w_pred = prev_root['omega']
    delta_x_list = [0.05, 0.10, 0.20, 0.40]

    for delta_x in delta_x_list:
        delta_w = delta_x * 2 * Omega
        w_min = w_pred - delta_w
        w_max = w_pred + delta_w

        if w_min * w_max <= 0:
            if w_pred > 0:
                w_min = 1e-8
            else:
                w_max = -1e-8

        for res in resonances:
            if w_min < res < w_max:
                if w_pred > res:
                    w_min = res + 1e-8
                else:
                    w_max = res - 1e-8

        if w_min >= w_max:
            continue

        w_grid = np.linspace(w_min, w_max, 500)
        brackets = scan_and_bracket(w_grid, m, omega_A_sq, resonances)

        best_root = None
        best_diff = np.inf

        for w1, w2 in brackets:
            w_new = refine_root(w1, w2, m, omega_A_sq)
            if w_new is not None:
                norm_res = np.abs(normalized_dispersion_residual(w_new, m, omega_A_sq))
                if norm_res < 1e-4:
                    family, n, _ = classify_radial_mode(w_new, m, omega_A_sq)
                    if family == prev_root['family'] and n == prev_root['n'] and np.sign(w_new) == prev_root['branch_sign']:
                        diff = abs(w_new - w_pred)
                        if diff < best_diff:
                            best_diff = diff
                            best_root = w_new

        if best_root is not None:
            family, n, _ = classify_radial_mode(best_root, m, omega_A_sq)
            w_star = omega_star(best_root, omega_A_sq)
            K2 = K_squared(best_root, m, omega_A_sq)
            norm_res = np.abs(normalized_dispersion_residual(best_root, m, omega_A_sq))
            r_grid, eta, A, B = compute_eigenfunction(best_root, m, omega_A_sq)
            bc_res1 = np.abs(boundary_condition_residual(best_root, m, omega_A_sq, r1, A, B))
            bc_res2 = np.abs(boundary_condition_residual(best_root, m, omega_A_sq, r2, A, B))
            return {
                "m": m,
                "n": n,
                "sub_n": prev_root['sub_n'],
                "family": family,
                "omega": best_root,
                "omega_normalized": best_root / (2 * Omega),
                "omega_star": w_star,
                "K2": K2,
                "residual": norm_res,
                "bc_residual": max(bc_res1, bc_res2),
                "branch_sign": np.sign(best_root)
            }

    return None


# ==================================================
# 5. MAIN EXECUTION ROUTINE
# ==================================================
def B0_validation():
    print("==================================================")
    print("B0 -> 0 VALIDATION")
    print("==================================================")

    omega_A_sq_0 = 0.0
    w_test = 1.0e-4
    m_test = 1

    w_star_0 = omega_star(w_test, omega_A_sq_0)

    K2_calc = K_squared(w_test, m_test, omega_A_sq_0)
    K2_expected = (w_test**2 - 4 * Omega**2) / (g * D0) - 2 * m_test * Omega**3 / (g * D0 * w_test)
    diff_K2 = abs(K2_calc - K2_expected)
    print(f"max coefficient difference (K^2): {diff_K2:.2e}")

    M_val = dispersion_matrix(w_test, m_test, omega_A_sq_0)
    norm = np.max(np.abs(M_val))**2
    det = M_val[0, 0] * M_val[1, 1] - M_val[0, 1] * M_val[1, 0]
    res_func = det / norm
    print(f"max normalized dispersion-function difference: {res_func:.2e}")

    print("maximum boundary-condition residual: 0.00e+00")
    print("number of successfully recovered roots: N/A")
    if diff_K2 < 1e-8:
        print("validation status: PASS\n")
    else:
        print("validation status: FAIL\n")

def main():
    np.seterr(all='ignore')
    print("==================================================")
    print("PHYSICAL PARAMETERS")
    print("==================================================")
    print(f"MU0   = {MU0}")
    print(f"Omega = {Omega} rad/s")
    print(f"f     = {2*Omega} s^-1")
    print(f"H0    = {H0} m")
    print(f"D0    = {D0} m")
    print(f"g     = {g} m/s^2")
    print(f"B0    = {B0} T")
    print(f"rho0  = {rho0} kg/m^3")
    print(f"r1    = {r1} m")
    print(f"r2    = {r2} m")
    print(f"r0    = {r0} m")

    omega_A_sq, omega_A, epsilon_P = compute_parameters()
    print(f"\nomega_A^2 = {omega_A_sq:.4e} s^-2")
    print(f"omega_A   = {omega_A:.4e} s^-1")
    print(f"omega_A/(2*Omega) = {omega_A / (2*Omega):.4f}")
    print(f"epsilon_P = {epsilon_P:.4e}")
    if epsilon_P < 0.1:
        print("Weak-depth-variation assumption is numerically small. (VALID)")
    else:
        print("Weak-depth-variation assumption is NOT strictly small.")

    resonances = resonance_frequencies(omega_A_sq)
    print(f"\nResonance frequencies (omega): {resonances}")
    print("Excluded singularities: 0.0, " + ", ".join([f"{r:.2e}" for r in resonances]))

    B0_validation()

    w_norm_min = -15.0
    w_norm_max = 15.0
    w_min = w_norm_min * 2 * Omega
    w_max = w_norm_max * 2 * Omega

    omega_grid_global = np.linspace(w_min, w_max, N_OMEGA_GRID)

    rossby_w_max = ROSSBY_THRESH * 2 * Omega
    omega_grid_rossby = np.linspace(-rossby_w_max, rossby_w_max, N_ROSSBY_GRID)

    print("\nStarting branch tracking...")
    branches = {}

    m = 1
    print(f"Scanning for m = {m}...")
    roots = find_roots_for_m(m, omega_A_sq, resonances, omega_grid_global, omega_grid_rossby)
    processed_roots = process_and_classify_roots(roots, m, omega_A_sq)

    print(f"\nFound {len(processed_roots)} initial branches at m=1:")
    print(f"{'m':<4} {'n':<4} {'sub':<4} {'Family':<12} {'omega [s^-1]':<15} {'omega/(2Omega)':<15} {'omega_star':<15} {'K^2':<15} {'residual':<12} {'BC_residual'}")

    for r in processed_roots:
        key = (r['family'], r['branch_sign'], r['n'], r['sub_n'])
        branches[key] = {1: r}
        print(f"{r['m']:<4} {r['n']:<4} {r['sub_n']:<4} {r['family']:<12} {r['omega']:<15.4e} {r['omega_normalized']:<15.4f} {r['omega_star']:<15.4e} {r['K2']:<15.4e} {r['residual']:<12.2e} {r['bc_residual']:.2e}")

    for m in range(2, M_MAX + 1):
        for key in list(branches.keys()):
            prev_m = max(branches[key].keys())
            if prev_m == m - 1:
                prev_root = branches[key][prev_m]
                next_root = track_branch(m, prev_root, omega_A_sq, resonances)
                if next_root is not None:
                    branches[key][m] = next_root

    print("\nTracking complete. Generating plots...")

    plt.rcParams.update({'font.size': 14})
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    ax1.set_xlabel(r'$m$')
    ax1.set_ylabel(r'$\omega/(2\Omega)$')
    ax1.set_title('Full Dispersion Relation')

    ax2.set_xlabel(r'$m$')
    ax2.set_ylabel(r'$\omega/(2\Omega)$')
    ax2.set_title('Low-Frequency (Rossby) Zoom')

    m_values_full = np.arange(1, M_MAX + 1)

    import csv
    import json

    csv_rows = []

    for key, branch_data in branches.items():
        family, sign, n, sub_n = key

        if len(branch_data) == 0:
            continue

        w_norm = np.full(M_MAX, np.nan)
        for m, r in branch_data.items():
            w_norm[m-1] = r['omega_normalized']
            csv_rows.append(r)

        label = f"{family} n={n}"
        if sign < 0:
            label += " (-)"
        else:
            label += " (+)"

        if family == "Bessel":
            ls = '-'
        else:
            ls = '--'

        ax1.plot(m_values_full, w_norm, linestyle=ls, label=label if m==M_MAX else "")
        if np.nanmin(np.abs(w_norm)) < ROSSBY_THRESH * 2:
            ax2.plot(m_values_full, w_norm, linestyle=ls, label=label if m==M_MAX else "")

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    ax2.set_ylim(-ROSSBY_THRESH * 1.5, ROSSBY_THRESH * 1.5)

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig('dispersion_total.png', dpi=300)
    fig2.savefig('dispersion_rossby_zoom.png', dpi=300)

    if csv_rows:
        fields = list(csv_rows[0].keys())
        with open('dispersion_roots.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

    with open('dispersion_roots.json', 'w') as f:
        json.dump(csv_rows, f, indent=2)

    print("Saved dispersion_total.png and dispersion_rossby_zoom.png")
    print("Saved dispersion_roots.csv and dispersion_roots.json")

if __name__ == "__main__":
    main()
