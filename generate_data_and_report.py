import numpy as np
from swmhd_solver import get_coefficients_dimensional, solve_m_dimensional, solve_m_dimensionless

def main():
    # Parameters
    Omega = 0.5e-4
    H0 = 500.0
    g = 9.81
    B0 = 5e-4
    rho0 = 1000.0
    C = 0.0
    r1 = 0.5e6
    r2 = 1.0e6
    r0 = 0.5 * (r1 + r2)
    mu0 = 4.0 * np.pi * 1e-7
    n = 2
    f = 2.0 * Omega

    omega_tol = 10**(-8) * Omega  # 5e-13

    # Storage for CSV rows
    # m, root_index, real_omega, imag_omega, real_omega_over_Omega, imag_omega_over_Omega, abs_residual, normalized_residual, root_class
    csv_rows = []

    # Dictionary to collect results for report sections
    report_data = {
        'm_values': [],
        'kr': 0.0,
        'kr_sq': 0.0,
        'Heq_r0': 0.0,
        'Q_r0': 0.0,
        'omega_A_sq_r0': 0.0,
        'roots_per_m': {},
        'max_residuals_per_m': {},
        'coefficients_m': {}
    }

    # Let's run the sweep m = 1 to 30
    for m in range(1, 31):
        # Solve the dimensional polynomial
        roots_dim, coeffs_dim = solve_m_dimensional(m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n)

        # Unpack local quantities (same across m, except kappa_sq)
        _, Heq, Q, f_val, omega_A_sq, kr, kr_sq, kappa_sq = get_coefficients_dimensional(
            m, r0, Omega, H0, g, B0, rho0, mu0, C, r1, r2, n
        )

        if m == 1:
            report_data['kr'] = kr
            report_data['kr_sq'] = kr_sq
            report_data['Heq_r0'] = Heq
            report_data['Q_r0'] = Q
            report_data['omega_A_sq_r0'] = omega_A_sq

        report_data['m_values'].append((m, kappa_sq))
        report_data['roots_per_m'][m] = roots_dim

        if m in [1, 10, 20, 30]:
            report_data['coefficients_m'][m] = coeffs_dim

        max_norm_res = 0.0

        # We want to keep the raw index of np.roots
        for idx, w in enumerate(roots_dim):
            # Compute residual
            # R(w) = A8 w^8 + A7 w^7 + ... + A0
            A8, A7, A6, A5, A4, A3, A2, A1, A0 = coeffs_dim
            R = (A8*w**8 + A7*w**7 + A6*w**6 + A5*w**5 + A4*w**4 + A3*w**3 + A2*w**2 + A1*w + A0)
            abs_res = abs(R)

            # Sum of term magnitudes for normalization
            term_sum = (abs(A8*w**8) + abs(A7*w**7) + abs(A6*w**6) + abs(A5*w**5) +
                        abs(A4*w**4) + abs(A3*w**3) + abs(A2*w**2) + abs(A1*w) + abs(A0))
            norm_res = abs_res / (term_sum + np.finfo(float).eps)

            if norm_res > max_norm_res:
                max_norm_res = norm_res

            # Classify root
            real_part = w.real
            imag_part = w.imag

            if abs(imag_part) <= omega_tol:
                # Treated as real
                # Handled explicitly if numerically zero
                if abs(real_part) < 1e-18:
                    root_class = 'real_zero'
                elif real_part > 0:
                    root_class = 'real_positive'
                else:
                    root_class = 'real_negative'
            else:
                # Complex
                if imag_part > 0:
                    root_class = 'complex_unstable'
                else:
                    root_class = 'complex_damped'

            csv_rows.append({
                'm': m,
                'root_index': idx,
                'real_omega': real_part,
                'imag_omega': imag_part,
                'real_omega_over_Omega': real_part / Omega,
                'imag_omega_over_Omega': imag_part / Omega,
                'abs_residual': abs_res,
                'normalized_residual': norm_res,
                'root_class': root_class
            })

        report_data['max_residuals_per_m'][m] = max_norm_res

    # Write CSV
    csv_filepath = 'outputs/dispersion_roots.csv'
    with open(csv_filepath, 'w') as f_csv:
        # Header
        f_csv.write("m,root_index,real_omega,imag_omega,real_omega_over_Omega,imag_omega_over_Omega,abs_residual,normalized_residual,root_class\n")
        for row in csv_rows:
            f_csv.write(f"{row['m']},{row['root_index']},{row['real_omega']:.18e},{row['imag_omega']:.18e},"
                        f"{row['real_omega_over_Omega']:.18e},{row['imag_omega_over_Omega']:.18e},"
                        f"{row['abs_residual']:.18e},{row['normalized_residual']:.18e},{row['root_class']}\n")

    print(f"Wrote {len(csv_rows)} root entries to {csv_filepath}")

    # Generate verification report section
    # Verification at trial frequencies
    import sympy as sp
    omega_sym = sp.Symbol('omega')
    omega_A_sq_sym, f_sym, g_sym, Heq_sym, m_sym, Q_sym, r_sym, B0_sym, mu0_sym, rho0_sym, C_sym, r0_sym = sp.symbols(
        'omega_A_sq f g Heq m Q r B0 mu0 rho0 C r0', complex=True
    )

    # Direct formula contributions
    omega_star = omega_sym + omega_A_sq_sym / omega_sym
    W1 = omega_sym * (omega_star**2 - f_sym**2) / (g_sym * Heq_sym * omega_star)
    W2 = m_sym * f_sym * Q_sym / (r_sym * g_sym * Heq_sym * omega_star)
    W3 = 4 * m_sym * f_sym * B0_sym**2 * Q_sym / (r_sym * mu0_sym * rho0_sym * omega_sym * g_sym * Heq_sym**3 * (omega_star**2 - f_sym**2))
    kappa_sq_direct = W1 + W2 + W3

    N_expr = omega_sym**4 + (2*omega_A_sq_sym - f_sym**2)*omega_sym**2 + omega_A_sq_sym**2
    D6_expr = (omega_sym**2 + omega_A_sq_sym) * N_expr
    P8_expr = N_expr**2 + (m_sym * f_sym * Q_sym / r_sym) * omega_sym * (N_expr + 4*omega_A_sq_sym*(omega_sym**2 + omega_A_sq_sym))
    kappa_sq_poly = P8_expr / (g_sym * Heq_sym * D6_expr)

    # Parameters for substitution
    subs_dict = {
        g_sym: g,
        Heq_sym: H0,
        f_sym: 2.0 * Omega,
        B0_sym: B0,
        rho0_sym: rho0,
        mu0_sym: mu0,
        C_sym: C,
        r0_sym: r0,
        r_sym: r0,
        m_sym: 10, # arbitrary
        omega_A_sq_sym: B0**2 / (mu0 * rho0 * H0**2),
        Q_sym: Omega**2 * r0 - C / r0**2
    }

    trial_omegas = [
        (0.5 + 0.2j) * Omega,
        (1.0 + 0.3j) * Omega,
        (2.0 - 0.5j) * Omega
    ]

    verif_results = []
    for idx, w_val in enumerate(trial_omegas, 1):
        v_direct = complex(kappa_sq_direct.subs(subs_dict).subs(omega_sym, w_val))
        v_poly = complex(kappa_sq_poly.subs(subs_dict).subs(omega_sym, w_val))

        err = abs(v_direct - v_poly) / max(abs(v_direct), abs(v_poly), np.finfo(float).eps)
        verif_results.append({
            'idx': idx,
            'omega': w_val,
            'direct': v_direct,
            'poly': v_poly,
            'err': err
        })

    # Write Text Report
    report_filepath = 'outputs/dispersion_report.txt'
    with open(report_filepath, 'w') as f_rep:
        f_rep.write("=======================================================================\n")
        f_rep.write("SWMHD WKB LOCAL DISPERSION RELATION REPORT AT r = r0\n")
        f_rep.write("=======================================================================\n\n")

        f_rep.write("### Section 1 — Parameters\n")
        f_rep.write(f"Rotation rate (Omega)                     = {Omega:.6e} rad/s\n")
        f_rep.write(f"Reference depth (H0)                     = {H0:.6f} m\n")
        f_rep.write(f"Gravitational acceleration (g)            = {g:.6f} m/s^2\n")
        f_rep.write(f"External vertical magnetic field (B0)     = {B0:.6e} T\n")
        f_rep.write(f"Density (rho0)                            = {rho0:.6f} kg/m^3\n")
        f_rep.write(f"Radial gravity constant (C)               = {C:.6f} m^3/s^2\n")
        f_rep.write(f"Inner radius (r1)                         = {r1:.6e} m\n")
        f_rep.write(f"Outer radius (r2)                         = {r2:.6e} m\n")
        f_rep.write(f"Coriolis parameter (f)                    = {f:.6e} rad/s\n")
        f_rep.write(f"Radial mode number (n)                    = {n}\n")
        f_rep.write(f"Vacuum magnetic permeability (mu0)        = {mu0:.6e} H/m\n\n")

        f_rep.write("### Section 2 — Reference radius\n")
        f_rep.write(f"r0 = 0.5 * (r1 + r2)                      = {r0:.6e} m\n\n")

        f_rep.write("### Section 3 — Local equilibrium quantities\n")
        f_rep.write(f"H_eq(r0)                                  = {report_data['Heq_r0']:.6e} m\n")
        f_rep.write(f"H_eq'(r0)                                 = {report_data['Q_r0']/g:.6e} m/m\n")
        f_rep.write(f"Q(r0)                                     = {report_data['Q_r0']:.6e} m^2/s^2\n")
        f_rep.write(f"omega_A^2(r0)                             = {report_data['omega_A_sq_r0']:.6e} rad^2/s^2\n\n")

        f_rep.write("### Section 4 — WKB quantization\n")
        f_rep.write(f"k_r = n * pi / (r2 - r1)                  = {report_data['kr']:.6e} m^-1\n")
        f_rep.write(f"k_r^2                                     = {report_data['kr_sq']:.6e} m^-2\n\n")

        f_rep.write("### Section 5 — kappa^2(m)\n")
        f_rep.write("m       kappa^2(m) [m^-2]\n")
        f_rep.write("----------------------------------------\n")
        for m, k_sq in report_data['m_values']:
            f_rep.write(f"{m:<8d}{k_sq:.12e}\n")
        f_rep.write("\n")

        f_rep.write("### Section 6 — Polynomial coefficients\n")
        for m in [1, 10, 20, 30]:
            f_rep.write(f"m = {m}:\n")
            coeffs = report_data['coefficients_m'][m]
            for idx, c_val in enumerate(coeffs):
                f_rep.write(f"  A_{8-idx:<2d}= {c_val:.12e}\n")
            f_rep.write("\n")

        f_rep.write("### Section 7 — All roots\n")
        f_rep.write("m       root_idx   Re(omega) [rad/s]       Im(omega) [rad/s]       Re(omega)/Omega         Im(omega)/Omega         Class\n")
        f_rep.write("------------------------------------------------------------------------------------------------------------------------\n")
        # Let's read back from csv_rows to format nicely
        for row in csv_rows:
            f_rep.write(f"{row['m']:<8d}{row['root_index']:<11d}{row['real_omega']:<24.12e}{row['imag_omega']:<24.12e}"
                        f"{row['real_omega_over_Omega']:<24.12f}{row['imag_omega_over_Omega']:<24.12f}{row['root_class']}\n")
        f_rep.write("\n")

        f_rep.write("### Section 8 — Residuals\n")
        f_rep.write("m       Max Normalized Residual\n")
        f_rep.write("----------------------------------------\n")
        for m in range(1, 31):
            f_rep.write(f"{m:<8d}{report_data['max_residuals_per_m'][m]:.12e}\n")
        f_rep.write("\n")

        f_rep.write("### Section 9 — Root classification\n")
        f_rep.write("Summary count of classes across all 240 roots:\n")
        class_counts = {}
        for row in csv_rows:
            c = row['root_class']
            class_counts[c] = class_counts.get(c, 0) + 1
        for c, count in sorted(class_counts.items()):
            f_rep.write(f"  {c:<24s}: {count}\n")
        f_rep.write("\n")

        f_rep.write("### Section 10 — Algebraic verification\n")
        f_rep.write("Verification of kappa^2 Direct vs. Polynomial at r = r0 (with m=10):\n")
        for res in verif_results:
            f_rep.write(f"Trial {res['idx']} (omega = {res['omega'] / Omega:.1f} + {res['omega'].imag / Omega:.1f}i * Omega):\n")
            f_rep.write(f"  kappa^2 Direct     = {res['direct']}\n")
            f_rep.write(f"  kappa^2 Polynomial = {res['poly']}\n")
            f_rep.write(f"  Relative error      = {res['err']:.12e}\n")
        f_rep.write("\n")

    print(f"Wrote report to {report_filepath}")

if __name__ == "__main__":
    main()
