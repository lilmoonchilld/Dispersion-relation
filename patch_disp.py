import re

with open('swmhd_dispersion.py', 'r') as f:
    content = f.read()

# I will append the purely imaginary solver at the end of the file.

new_code = """
# ==========================================================
# 7. Purely Imaginary Modes Solver (Pure Growth/Decay)
# ==========================================================
# This section searches strictly along Re(omega) = 0 for modes
# with omega = i * gamma.
import csv

def pure_imaginary_omega(hat_gamma):
    return 1j * hat_gamma

def pure_gamma_sigma_min(hat_gamma, m_val, min_abs_gamma=1e-4):
    if abs(hat_gamma) <= min_abs_gamma:
        return np.inf, np.inf

    hat_omega = pure_imaginary_omega(hat_gamma)
    try:
        L = build_matrix(hat_omega, m_val)
        s = svdvals(L, overwrite_a=False, check_finite=False)
        if len(s) == 0:
            return np.inf, np.inf
        return float(np.real(s[-1])), float(np.real(s[0]))
    except Exception:
        return np.inf, np.inf

def pure_imaginary_objective(hat_gamma, m_val, min_abs_gamma=1e-4):
    sigma_min, sigma_max = pure_gamma_sigma_min(hat_gamma, m_val, min_abs_gamma)
    if sigma_max == np.inf or sigma_max == 0:
        return np.inf
    return sigma_min / sigma_max

def extract_complex_eigenfunction(hat_omega, m_val):
    try:
        L = build_matrix(hat_omega, m_val)
        _, s, Vh = svd(L, full_matrices=False, overwrite_a=False, check_finite=False)
        eta = Vh[-1, :].copy()
        norm = np.max(np.abs(eta))
        if norm > 0:
            eta /= norm

        # compute residual
        residual = np.linalg.norm(L @ eta) / np.linalg.norm(eta)

        return eta, float(np.real(s[-1])), residual
    except Exception:
        return None, np.inf, np.inf

def find_pure_imaginary_modes(m_val, gamma_range=(-5.0, 5.0), n_scan=2000,
                              min_abs_gamma=1e-4, sigma_rtol=1e-8, residual_tol=1e-6,
                              gamma_zero_tol=1e-8, duplicate_tol=1e-4):
    scan_points = np.linspace(gamma_range[0], gamma_range[1], n_scan)
    valid_points = scan_points[np.abs(scan_points) > min_abs_gamma]

    sigma_scan = np.array([pure_imaginary_objective(g, m_val, min_abs_gamma) for g in valid_points])

    candidates = []
    for k in range(1, len(valid_points) - 1):
        if not np.isfinite(sigma_scan[k]):
            continue
        if sigma_scan[k] <= sigma_scan[k - 1] and sigma_scan[k] <= sigma_scan[k + 1]:
            candidates.append((valid_points[k - 1], valid_points[k], valid_points[k + 1]))

    modes = []
    for left, center, right in candidates:
        try:
            res = minimize_scalar(
                pure_imaginary_objective,
                args=(m_val, min_abs_gamma),
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 200}
            )
            if res.success and np.isfinite(res.x):
                gamma_refined = float(res.x)
            else:
                gamma_refined = float(center)
        except Exception:
            gamma_refined = float(center)

        rel_sigma = pure_imaginary_objective(gamma_refined, m_val, min_abs_gamma)
        if rel_sigma > sigma_rtol:
            continue

        eta, sigma_min, residual = extract_complex_eigenfunction(pure_imaginary_omega(gamma_refined), m_val)

        if residual > residual_tol:
            continue

        # check duplicate
        is_duplicate = False
        for mode in modes:
            if abs(gamma_refined - mode['gamma_hat']) < duplicate_tol:
                is_duplicate = True
                break

        if not is_duplicate:
            if abs(gamma_refined) < gamma_zero_tol:
                stability = 'neutral'
            elif gamma_refined > 0:
                stability = 'growing'
            else:
                stability = 'decaying'

            modes.append({
                'm': m_val,
                'gamma_hat': gamma_refined,
                'gamma_dimensional': 2.0 * Omega * gamma_refined,
                'sigma_min': sigma_min,
                'relative_sigma': rel_sigma,
                'residual': residual,
                'stability': stability,
                'eta': eta
            })

    modes.sort(key=lambda x: x['gamma_hat'])
    return modes

print("\\n" + "=" * 62)
print("  Pure-imaginary stability spectrum")
print("  (Searching specifically for purely growing/decaying modes")
print("   with Re(omega) = 0)")
print("=" * 62)

all_pure_modes = []
most_unstable_per_m = {}

gamma_range_val = (-5.0, 5.0)

for m in m_arr:
    modes = find_pure_imaginary_modes(m, gamma_range=gamma_range_val)
    all_pure_modes.extend(modes)

    print(f"m = {m:2d}:")
    if len(modes) == 0:
        print("    no accepted pure-imaginary eigenmode found")
        most_unstable_per_m[m] = None
    else:
        most_unstable = max(modes, key=lambda x: x['gamma_hat'])
        most_unstable_per_m[m] = most_unstable
        print(f"    most unstable gamma_hat = {most_unstable['gamma_hat']:+.6e}")
        print(f"    gamma = {most_unstable['gamma_dimensional']:+.6e} s^-1")
        print(f"    omega = 0 + {most_unstable['gamma_hat']:+.6e} i")
        print(f"    sigma_min = {most_unstable['sigma_min']:.3e}, residual = {most_unstable['residual']:.3e}")
        print(f"    classification = {most_unstable['stability']}")

print("=" * 62)
print("  Summary Table")
print("  m       gamma_hat       gamma [s^-1]       classification")
print("-" * 62)
for m in m_arr:
    mu = most_unstable_per_m.get(m)
    if mu is None:
        print(f" {m:2d}       none found")
    else:
        print(f" {m:2d}       {mu['gamma_hat']:+.5e}     {mu['gamma_dimensional']:+.5e}      {mu['stability']}")
print("=" * 62)

# Global most unstable
global_most_unstable = None
for m, mu in most_unstable_per_m.items():
    if mu is not None:
        if global_most_unstable is None or mu['gamma_hat'] > global_most_unstable['gamma_hat']:
            global_most_unstable = mu

if global_most_unstable is not None:
    print("GLOBAL MOST UNSTABLE PURELY-IMAGINARY MODE")
    print("=" * 62)
    print(f"m = {global_most_unstable['m']}")
    print(f"gamma_hat = {global_most_unstable['gamma_hat']:+.6e}")
    print(f"gamma = {global_most_unstable['gamma_dimensional']:+.6e} s^-1")
    print(f"omega_hat = i({global_most_unstable['gamma_hat']:+.6e})")
    print(f"relative_sigma = {global_most_unstable['relative_sigma']:.3e}")
    print(f"residual = {global_most_unstable['residual']:.3e}")
    if global_most_unstable['gamma_hat'] > 0:
        print("-> The scan contains a purely growing mode.")
    else:
        print("-> No purely growing mode with Re(omega)=0 was found in the specified m and gamma search ranges.")
else:
    print("No pure-imaginary eigenmodes found across all m.")
print("=" * 62)

# Save to CSV
csv_filename = 'swmhd_pure_imaginary_modes.csv'
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['m', 'mode_index', 'omega_r_hat', 'gamma_hat', 'gamma_dimensional', 'sigma_min', 'relative_sigma', 'residual', 'stability'])

    # Sort all modes by m, then by gamma_hat
    all_pure_modes.sort(key=lambda x: (x['m'], x['gamma_hat']))

    idx_map = {}
    for mode in all_pure_modes:
        m = mode['m']
        idx_map[m] = idx_map.get(m, 0) + 1
        writer.writerow([
            m,
            idx_map[m],
            0.0,
            mode['gamma_hat'],
            mode['gamma_dimensional'],
            mode['sigma_min'],
            mode['relative_sigma'],
            mode['residual'],
            mode['stability']
        ])
print(f"Saved: {csv_filename}")

summary_csv_filename = 'swmhd_pure_imaginary_summary.csv'
with open(summary_csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['m', 'most_unstable_gamma_hat', 'most_unstable_gamma', 'stability'])
    for m in m_arr:
        mu = most_unstable_per_m.get(m)
        if mu is not None:
            writer.writerow([m, mu['gamma_hat'], mu['gamma_dimensional'], mu['stability']])
        else:
            writer.writerow([m, 'NaN', 'NaN', 'none found'])
print(f"Saved: {summary_csv_filename}")


# Plot purely imaginary dispersion relation
fig_stab, ax_stab = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax_stab.axhline(0, color='grey', lw=1.0, ls='--')

x_m = []
y_gamma = []
colors = []

for m in m_arr:
    mu = most_unstable_per_m.get(m)
    if mu is not None:
        x_m.append(m)
        y_gamma.append(mu['gamma_hat'])
        if mu['gamma_hat'] > 0:
            colors.append('r')
        elif mu['gamma_hat'] < 0:
            colors.append('b')
        else:
            colors.append('k')

ax_stab.scatter(x_m, y_gamma, c=colors, marker='o', s=60, zorder=5)
ax_stab.plot(x_m, y_gamma, color='k', lw=1.5, zorder=4, alpha=0.6)

ax_stab.set_xlabel('Azimuthal wavenumber $m$', fontsize=20)
ax_stab.set_ylabel(r'Growth/decay rate $\gamma/(2\Omega)$', fontsize=20)
ax_stab.set_xlim(0.7, M_max+0.5)
ax_stab.set_xticks(m_arr[::3])
ax_stab.grid(True, alpha=0.22)
ax_stab.set_title(r'Most unstable purely-imaginary mode', fontsize=22)

fig_stab.savefig("swmhd_pure_imaginary_dispersion.png", dpi=300, bbox_inches='tight')
print("Saved: swmhd_pure_imaginary_dispersion.png")

print("\\nNote: This calculation searches only for purely imaginary eigenfrequencies")
print("with Re(omega)=0. No purely growing mode with Re(omega)=0 was found")
print("in the specified m and gamma search ranges (if applicable).")

"""

with open('swmhd_dispersion.py', 'w') as f:
    f.write(content + "\n" + new_code)
