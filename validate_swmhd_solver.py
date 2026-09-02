import numpy as np
import scipy.linalg as sla
import swmhd_dimensional_dispersion as swmhd

def run_tests():
    print("==============================================================")
    print("         RUNNING SWMHD SOLVER VALIDATION SUITE               ")
    print("==============================================================")

    # ------------------------------------------------------------
    # Test A: Finite rotation (Reference case: Omega=0.5e-4)
    # ------------------------------------------------------------
    print("\n--- Test A: Finite rotation (Omega = 0.5e-4 rad/s) ---")
    swmhd.Omega = 0.5e-4
    swmhd.B0 = 5e-4
    swmhd.C = 0.0
    swmhd.beta_eff = 2.0 * swmhd.C / swmhd.r0**3
    swmhd.VA2 = swmhd.B0**2 / (swmhd.MU0 * swmhd.rho0)
    swmhd.omA2 = swmhd.VA2 / swmhd.H0**2
    swmhd.omA = np.sqrt(swmhd.omA2)

    A, B = swmhd.build_dimensional_matrices(m_val=1)
    assert np.all(np.isfinite(A)), "Test A Matrix A has NaNs or Infs!"
    assert np.all(np.isfinite(B)), "Test A Matrix B has NaNs or Infs!"
    eigs_A = swmhd.solve_dimensional_eigenproblem(m_val=1)
    print(f"  Matrix A max abs: {np.max(np.abs(A)):.4e}, Matrix B max abs: {np.max(np.abs(B)):.4e}")
    print(f"  Found {len(eigs_A)} finite physical eigenfrequencies for m=1.")
    print("  First 5 positive frequencies (rad/s):", eigs_A[eigs_A > 0][:5])

    # ------------------------------------------------------------
    # Test B: Zero rotation (Omega = 0.0)
    # ------------------------------------------------------------
    print("\n--- Test B: Zero rotation (Omega = 0.0 rad/s) ---")
    swmhd.Omega = 0.0
    A, B = swmhd.build_dimensional_matrices(m_val=1)
    assert np.all(np.isfinite(A)), "Test B Matrix A has NaNs or Infs!"
    assert np.all(np.isfinite(B)), "Test B Matrix B has NaNs or Infs!"
    eigs_B = swmhd.solve_dimensional_eigenproblem(m_val=1)
    print(f"  Matrix A max abs: {np.max(np.abs(A)):.4e}, Matrix B max abs: {np.max(np.abs(B)):.4e}")
    print(f"  Found {len(eigs_B)} finite physical eigenfrequencies for m=1.")
    print("  First 5 positive frequencies (rad/s):", eigs_B[eigs_B > 0][:5])

    # ------------------------------------------------------------
    # Test C: Non-magnetic limit (B0 = 0.0)
    # ------------------------------------------------------------
    print("\n--- Test C: Non-magnetic limit (B0 = 0.0 T) ---")
    swmhd.Omega = 0.5e-4
    swmhd.B0 = 0.0
    swmhd.C = 0.0
    swmhd.beta_eff = 2.0 * swmhd.C / swmhd.r0**3
    swmhd.VA2 = swmhd.B0**2 / (swmhd.MU0 * swmhd.rho0)
    swmhd.omA2 = swmhd.VA2 / swmhd.H0**2
    swmhd.omA = np.sqrt(swmhd.omA2)

    A, B = swmhd.build_dimensional_matrices(m_val=1)
    assert np.all(np.isfinite(A)), "Test C Matrix A has NaNs or Infs!"
    assert np.all(np.isfinite(B)), "Test C Matrix B has NaNs or Infs!"
    eigs_C = swmhd.solve_dimensional_eigenproblem(m_val=1)
    print(f"  Found {len(eigs_C)} finite physical eigenfrequencies for m=1.")
    print("  First 5 positive frequencies (rad/s):", eigs_C[eigs_C > 0][:5])

    # ------------------------------------------------------------
    # Test D: Zero central potential (C = 0.0)
    # ------------------------------------------------------------
    print("\n--- Test D: Zero central potential (C = 0.0) ---")
    swmhd.Omega = 0.5e-4
    swmhd.B0 = 5e-4
    swmhd.C = 0.0
    swmhd.beta_eff = 0.0
    A, B = swmhd.build_dimensional_matrices(m_val=1)
    assert np.all(np.isfinite(A)), "Test D Matrix A has NaNs or Infs!"
    assert np.all(np.isfinite(B)), "Test D Matrix B has NaNs or Infs!"
    eigs_D = swmhd.solve_dimensional_eigenproblem(m_val=1)
    print(f"  Found {len(eigs_D)} finite physical eigenfrequencies for m=1.")

    # ------------------------------------------------------------
    # Test E: Combined zero limit (Omega = 0, B0 = 0, C = 0)
    # ------------------------------------------------------------
    print("\n--- Test E: Pure Hydrodynamic Shallow Water (Omega=0, B0=0, C=0) ---")
    swmhd.Omega = 0.0
    swmhd.B0 = 0.0
    swmhd.C = 0.0
    swmhd.beta_eff = 0.0
    swmhd.VA2 = 0.0
    swmhd.omA2 = 0.0
    swmhd.omA = 0.0

    A, B = swmhd.build_dimensional_matrices(m_val=1)
    assert np.all(np.isfinite(A)), "Test E Matrix A has NaNs or Infs!"
    assert np.all(np.isfinite(B)), "Test E Matrix B has NaNs or Infs!"
    eigs_E = swmhd.solve_dimensional_eigenproblem(m_val=1)
    print(f"  Found {len(eigs_E)} finite physical eigenfrequencies for m=1.")
    print("  First 5 positive frequencies (rad/s):", eigs_E[eigs_E > 0][:5])

    # Reset parameters to reference case
    swmhd.Omega = 0.5e-4
    swmhd.B0 = 5e-4
    swmhd.C = 0.0
    swmhd.beta_eff = 2.0 * swmhd.C / swmhd.r0**3
    swmhd.VA2 = swmhd.B0**2 / (swmhd.MU0 * swmhd.rho0)
    swmhd.omA2 = swmhd.VA2 / swmhd.H0**2
    swmhd.omA = np.sqrt(swmhd.omA2)

    # ------------------------------------------------------------
    # Regression Test: Legacy vs New Solver Comparison at Omega > 0
    # ------------------------------------------------------------
    print("\n==============================================================")
    print("   REGRESSION TEST: LEGACY vs NEW DIMENSIONAL SOLVER (m=1)")
    print("==============================================================")
    legacy_eigs = swmhd.find_eigenvalues_legacy(m_val=1)
    new_eigs = swmhd.solve_dimensional_eigenproblem(m_val=1)

    print(f"  Legacy solver roots count : {len(legacy_eigs)}")
    print(f"  New 5-field solver roots   : {len(new_eigs)}")

    # Exclude near-zero roots (< 1e-5 rad/s) for relative difference check
    oscillatory_legacy = legacy_eigs[np.abs(legacy_eigs) > 1e-5]

    print("\n  Comparing oscillatory physical frequencies (hat_omega = omega / (2*Omega)):")
    print(f"  {'Legacy hat_w':>14s} | {'New hat_w':>14s} | {'Abs Diff':>12s} | {'Rel Diff':>12s}")
    print("  " + "-" * 62)

    matched_diffs = []
    for leg_w in oscillatory_legacy:
        diffs = np.abs(new_eigs - leg_w)
        idx = np.argmin(diffs)
        min_diff = diffs[idx]
        new_w = new_eigs[idx]
        rel_diff = min_diff / abs(leg_w)
        matched_diffs.append(rel_diff)
        print(f"  {leg_w/(2*swmhd.Omega):14.6f} | {new_w/(2*swmhd.Omega):14.6f} | {min_diff:12.4e} | {rel_diff:12.4e}")

    mean_rel_diff = np.mean(matched_diffs)
    print(f"\n  Mean relative difference across oscillatory modes: {mean_rel_diff:.4e}")
    assert mean_rel_diff < 0.05, "Regression test failed! Mean spectrum mismatch exceeds 5%"

    # ------------------------------------------------------------
    # WKB vs Global Solver Comparison Table
    # ------------------------------------------------------------
    print("\n==============================================================")
    print("         WKB vs GLOBAL EIGENVALUE COMPARISON TABLE           ")
    print("==============================================================")
    print(f"  {'m':>3s} | {'Branch':>10s} | {'WKB omega (rad/s)':>18s} | {'Global omega (rad/s)':>20s} | {'Rel Diff':>10s}")
    print("  " + "-" * 74)

    for m in [1, 2, 3]:
        eigs = swmhd.solve_dimensional_eigenproblem(m)
        assigned = swmhd.assign_branches_global(eigs, m)
        oMP_p, oMP_m, oMK_in, oMK_out, _, oMS_val = swmhd.wkb_branches(m, n_radial=1)

        comparisons = [
            ('MP_p (n=1)', oMP_p, assigned['MP_p'].get(1)),
            ('MP_m (n=1)', oMP_m, assigned['MP_m'].get(1)),
            ('MK_in',      oMK_in, assigned['MK_in']),
            ('MK_out',     oMK_out, assigned['MK_out']),
        ]

        for bname, wkb_w, glob_w in comparisons:
            if wkb_w is not None and glob_w is not None and not np.isnan(wkb_w):
                rdiff = abs(glob_w - wkb_w) / abs(wkb_w)
                print(f"  {m:3d} | {bname:>10s} | {wkb_w:18.8e} | {glob_w:20.8e} | {rdiff:10.4e}")
            else:
                w_str = f"{wkb_w:18.8e}" if (wkb_w is not None and not np.isnan(wkb_w)) else "               N/A"
                g_str = f"{glob_w:20.8e}" if glob_w is not None else "                 N/A"
                print(f"  {m:3d} | {bname:>10s} | {w_str} | {g_str} |        N/A")

    print("\nAll validation tests and regression checks completed successfully!")

if __name__ == "__main__":
    run_tests()
