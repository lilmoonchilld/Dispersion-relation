import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# Publication-ready matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 18,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.size': 6,
    'ytick.major.width': 1.2,
})

# Input parameters
Omega = 0.5e-4       # Rotation rate [rad/s]
H0    = 500.0        # Reference depth [m]
g     = 9.81         # Gravitational acceleration [m/s^2]
B0    = 5e-4         # External vertical magnetic field [T]
rho0  = 1000.0       # Density [kg/m^3]
C     = 0.0          # Radial gravity constant [m^3/s^2]

r1    = 0.5e6        # Inner radius [m]
r2    = 1.0e6        # Outer radius [m]
r0    = 0.5 * (r1 + r2)

f     = 2.0 * Omega   # Coriolis parameter [rad/s]
n     = 2

mu0   = 4.0 * np.pi * 1e-7
ROOT_TOL = 1e-10

M_MIN = -20
M_MAX = 20
m_values = np.arange(M_MIN, M_MAX + 1)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def H_eq(r):
    return (
        H0
        + Omega**2 / (2.0 * g) * (r**2 - r0**2)
        + C / g * (1.0/r - 1.0/r0)
    )

def Q(r):
    return Omega**2 * r - C / r**2

def dH_eq_dr(r):
    return Q(r) / g

def omega_A2(r):
    H = H_eq(r)
    return B0**2 / (mu0 * rho0 * H**2)

def kappa_squared(m, r):
    kr = n * np.pi / (r2 - r1)
    return kr**2 + (m**2 / r**2)

def polynomial_coefficients(m, r, kappa_sq):
    H = H_eq(r)
    Q_r = Q(r)
    wA2 = omega_A2(r)

    A8 = 1.0
    A7 = 0.0

    A6 = (
        4.0 * wA2
        - 2.0 * f**2
        - kappa_sq * g * H
    )

    A5 = (
        m * f * Q_r / r
    )

    A4 = (
        6.0 * wA2**2
        - 4.0 * f**2 * wA2
        + f**4
        - kappa_sq * g * H * (3.0 * wA2 - f**2)
    )

    A3 = (
        m * f * Q_r / r
        * (6.0 * wA2 - f**2)
    )

    A2 = (
        4.0 * wA2**3
        - 2.0 * f**2 * wA2**2
        - kappa_sq * g * H * (3.0 * wA2**2 - f**2 * wA2)
    )

    A1 = (
        5.0 * m * f * Q_r * wA2**2 / r
    )

    A0 = (
        wA2**4
        - kappa_sq * g * H * wA2**3
    )

    return np.array([A8, A7, A6, A5, A4, A3, A2, A1, A0], dtype=float)

def dimensionless_coefficients(m, r, kappa_sq):
    A = polynomial_coefficients(m, r, kappa_sq)
    scale = 2.0 * Omega
    hat_A = np.zeros(9)
    for i in range(9):
        hat_A[i] = A[i] / (scale ** i)
    return hat_A

def solve_roots(m, r, kappa_sq):
    hat_A = dimensionless_coefficients(m, r, kappa_sq)
    hat_omega_roots = np.roots(hat_A)
    omega_roots = hat_omega_roots * (2.0 * Omega)
    return omega_roots

def check_consistency(m, r, omega):
    """
    Evaluates original WKB dispersion relation kappa^2 - (W1 + W2 + W3) at computed root omega.
    Returns residual.
    """
    H = H_eq(r)
    Q_r = Q(r)
    wA2 = omega_A2(r)
    kappa_sq = kappa_squared(m, r)

    if np.abs(omega) < 1e-30:
        return np.nan

    omega_star = omega + wA2 / omega

    if np.abs(omega_star**2 - f**2) < 1e-30 or np.abs(omega_star) < 1e-30:
        return np.nan

    W1 = (omega * (omega_star**2 - f**2)) / (g * H * omega_star)
    W2 = (m * f * Q_r) / (r * g * H * omega_star)
    W3 = (4.0 * m * f * B0**2 * Q_r) / (r * mu0 * rho0 * omega * g * (H**3) * (omega_star**2 - f**2))

    residual = np.abs(kappa_sq - (W1 + W2 + W3))
    return residual

def build_radial_grid():
    left = np.linspace(r1, r0, 5)
    right = np.linspace(r0, r2, 6)[1:]
    return np.concatenate((left, right))

def validate_parameters():
    if r1 <= 0:
        raise ValueError("r1 must be greater than 0")
    if r2 <= r1:
        raise ValueError("r2 must be greater than r1")
    if rho0 <= 0:
        raise ValueError("rho0 must be greater than 0")
    if g <= 0:
        raise ValueError("g must be greater than 0")
    if mu0 <= 0:
        raise ValueError("mu0 must be greater than 0")

    r_vals = build_radial_grid()
    for r in r_vals:
        if H_eq(r) <= 0:
            raise ValueError(f"H_eq at r={r} is non-positive: {H_eq(r)}")

def compute_dispersion():
    validate_parameters()
    r_values = build_radial_grid()

    # Check assertions for radial grid
    assert len(r_values) == 10
    assert np.isclose(r_values[0], r1)
    assert np.any(np.isclose(r_values, r0))
    assert np.isclose(r_values[-1], r2)
    assert len(np.unique(r_values)) == 10

    results = {}
    max_residual = 0.0

    for r in r_values:
        results[r] = {
            'm_vals': m_values,
            'roots': [],          # List of 8 roots for each m
            'residuals': [],      # List of residuals for each root for each m
            'branch_data': np.zeros((8, len(m_values))) # For plotting real branches
        }

        for m_idx, m in enumerate(m_values):
            kappa_sq = kappa_squared(m, r)
            roots = solve_roots(m, r, kappa_sq)

            # Sort roots by real part, then imaginary part
            roots_sorted = sorted(roots, key=lambda x: (x.real, x.imag))
            results[r]['roots'].append(roots_sorted)

            m_res = []
            for branch_idx, root in enumerate(roots_sorted):
                # Evaluate residual for consistency check
                res = check_consistency(m, r, root)
                m_res.append(res)
                if not np.isnan(res):
                    max_residual = max(max_residual, res)

                # Check if real under ROOT_TOL
                if np.abs(root.imag) < ROOT_TOL:
                    results[r]['branch_data'][branch_idx, m_idx] = root.real
                else:
                    results[r]['branch_data'][branch_idx, m_idx] = np.nan

            results[r]['residuals'].append(m_res)

    return results, max_residual

def plot_dispersion(results):
    r_values = build_radial_grid()
    for r in r_values:
        fig, ax = plt.subplots(figsize=(10, 8))

        branch_data = results[r]['branch_data']
        for b_idx in range(8):
            ax.plot(
                m_values,
                branch_data[b_idx, :],
                marker='o',
                markersize=4,
                linewidth=1.2,
                label=f"Branch {b_idx + 1}" if r == r_values[0] else ""  # Label first plot's branches for reference
            )

        ax.set_xlabel("Azimuthal mode number $m$")
        ax.set_ylabel(r"Dimensional frequency $\omega$ [rad s$^{-1}$]")
        ax.set_title(f"Local WKB SWMHD Dispersion Relation\n$r = {r:.4e}$ m, $n = {n}$")
        ax.grid(True, linestyle=':', alpha=0.6)

        # Format axes ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, f"dispersion_r_{r:.4e}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

def save_results(results, max_residual):
    r_values = build_radial_grid()

    # 1. Save summary CSV
    csv_filename = os.path.join(OUTPUT_DIR, "dispersion_summary.csv")
    with open(csv_filename, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "r [m]", "m", "root_index", "omega_real [rad/s]", "omega_imag [rad/s]", "omega_hat_real", "omega_hat_imag", "is_real"
        ])
        for r in r_values:
            for m_idx, m in enumerate(m_values):
                roots = results[r]['roots'][m_idx]
                for root_idx, rt in enumerate(roots):
                    is_real = np.abs(rt.imag) < ROOT_TOL
                    omega_hat = rt / (2.0 * Omega)
                    writer.writerow([
                        r, m, root_idx, rt.real, rt.imag, omega_hat.real, omega_hat.imag, int(is_real)
                    ])

    # 2. Save NPZ for complete root data structure
    npz_filename = os.path.join(OUTPUT_DIR, "dispersion_data.npz")
    npz_dict = {}
    for r_idx, r in enumerate(r_values):
        # Convert list of roots (complex array) to numpy array of shape (len(m_values), 8)
        roots_arr = np.array(results[r]['roots'])
        npz_dict[f"roots_r_{r_idx}"] = roots_arr
    npz_dict["r_values"] = r_values
    npz_dict["m_values"] = m_values
    np.savez(npz_filename, **npz_dict)

    # 3. Save diagnostic and verification report
    report_filename = os.path.join(OUTPUT_DIR, "diagnostic_report.txt")
    with open(report_filename, 'w') as f_rep:
        f_rep.write("======================================================================\n")
        f_rep.write("             WKB SWMHD LOCAL DISPERSION SOLVER REPORT\n")
        f_rep.write("======================================================================\n\n")
        f_rep.write("1. Input Parameters:\n")
        f_rep.write(f"   Rotation rate Omega = {Omega} rad/s\n")
        f_rep.write(f"   Reference depth H0 = {H0} m\n")
        f_rep.write(f"   Gravitational acceleration g = {g} m/s^2\n")
        f_rep.write(f"   External vertical magnetic field B0 = {B0} T\n")
        f_rep.write(f"   Density rho0 = {rho0} kg/m^3\n")
        f_rep.write(f"   Radial gravity constant C = {C} m^3/s^2\n")
        f_rep.write(f"   Inner radius r1 = {r1} m\n")
        f_rep.write(f"   Outer radius r2 = {r2} m\n")
        f_rep.write(f"   Reference radius r0 = {r0} m\n")
        f_rep.write(f"   Coriolis parameter f = {f} rad/s\n")
        f_rep.write(f"   Radial mode index n = {n}\n")
        f_rep.write(f"   Permeability of free space mu0 = {mu0} H/m\n")
        f_rep.write(f"   Root tolerance ROOT_TOL = {ROOT_TOL}\n\n")

        f_rep.write("2. Radial Point Diagnostics:\n")
        f_rep.write(f"{'Radius [m]':<15}{'H_eq [m]':<15}{'H_eq_prime':<15}{'Q(r)':<15}{'omega_A^2 [s^-2]':<15}\n")
        f_rep.write("-" * 75 + "\n")
        for r in r_values:
            H = H_eq(r)
            Hp = dH_eq_dr(r)
            Qr = Q(r)
            wA2 = omega_A2(r)
            f_rep.write(f"{r:<15.4e}{H:<15.4f}{Hp:<15.4e}{Qr:<15.4e}{wA2:<15.4e}\n")
        f_rep.write("\n")

        f_rep.write("3. Solver and Verification Metrics:\n")
        f_rep.write(f"   Total grid points: {len(r_values)}\n")
        f_rep.write(f"   m range: [{M_MIN}, {M_MAX}]\n")

        # Count real vs complex roots
        total_real = 0
        total_complex = 0
        for r in r_values:
            for m_idx in range(len(m_values)):
                roots = results[r]['roots'][m_idx]
                for rt in roots:
                    if np.abs(rt.imag) < ROOT_TOL:
                        total_real += 1
                    else:
                        total_complex += 1

        f_rep.write(f"   Total roots calculated: {total_real + total_complex}\n")
        f_rep.write(f"   Total numerically real roots: {total_real}\n")
        f_rep.write(f"   Total numerically complex roots: {total_complex}\n")
        f_rep.write(f"   Maximum original polynomial residual across all roots: {max_residual:.4e}\n")
        f_rep.write("======================================================================\n")

if __name__ == "__main__":
    print("Running WKB SWMHD Local Dispersion Relation Solver...")
    results, max_res = compute_dispersion()
    print("Saving figures and datasets...")
    plot_dispersion(results)
    save_results(results, max_res)
    print(f"Done! All outputs saved to '{OUTPUT_DIR}/'. Maximum residual: {max_res:.4e}")
