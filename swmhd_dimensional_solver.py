"""
This implementation solves the dimensional linearized SWMHD
global eigenvalue problem directly in the physical radial
coordinate r and retains eta, u_r, u_theta, b_r, b_theta
as simultaneous unknowns.

No normalization by Omega is used.

Consequently the formulation remains regular in the
Omega -> 0 limit and does not require division by omega.
"""

import numpy as np
import scipy.linalg as slab
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# Publication style settings
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})


def physical_parameters(
    Omega=0.5e-4,
    H0=500.0,
    g=9.81,
    B0=5e-4,
    rho0=1000.0,
    C=0.0,
    r1=0.5e6,
    r2=1.0e6,
    N_col=32
):
    """
    Returns a dictionary of physical parameters in SI units.
    """
    mu0 = 4.0 * np.pi * 1e-7
    r0 = 0.5 * (r1 + r2)
    beta_eff = 2.0 * C / (r0**3) if r0 > 0 else 0.0

    va2 = B0**2 / (mu0 * rho0)
    omega_A2 = va2 / (H0**2)
    omega_A = np.sqrt(omega_A2)
    c0sq = g * H0
    c0 = np.sqrt(c0sq)

    return {
        'mu0': mu0,
        'Omega': Omega,
        'H0': H0,
        'g': g,
        'B0': B0,
        'rho0': rho0,
        'C': C,
        'r1': r1,
        'r2': r2,
        'r0': r0,
        'beta_eff': beta_eff,
        'va2': va2,
        'omega_A': omega_A,
        'omega_A2': omega_A2,
        'c0sq': c0sq,
        'c0': c0,
        'N_col': N_col
    }


def chebyshev_lobatto(N, r_min, r_max):
    """
    Constructs Chebyshev-Lobatto radial grid and differentiation matrices
    in physical coordinates (meters).
    """
    j = np.arange(N)
    xi = np.cos(j * np.pi / (N - 1))
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0

    X = np.tile(xi, (N, 1))
    dX = X - X.T
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) * ((-1)**(i + k)) / dX[i, k]
    D -= np.diag(D.sum(axis=1))

    sc = 2.0 / (r_max - r_min)
    D1 = sc * D
    D2 = D1 @ D1

    r_grid = 0.5 * (r_min + r_max) + 0.5 * (r_max - r_min) * xi
    # Flip to orient from r_min to r_max
    r_grid = r_grid[::-1]
    D1 = D1[::-1, ::-1]
    D2 = D2[::-1, ::-1]

    return r_grid, D1, D2


def build_dimensional_matrices(m_val, params):
    """
    Constructs the 5-field linear operators A(m) and B(m) such that:
    A q = lambda B q
    with q = [eta, u_r, u_theta, b_r, b_theta]^T and lambda = -i * omega.
    Impermeable boundary conditions u_r(r1) = 0 and u_r(r2) = 0 are enforced.
    """
    N_col = params['N_col']
    r1 = params['r1']
    r2 = params['r2']
    r0 = params['r0']
    H0 = params['H0']
    g = params['g']
    B0 = params['B0']
    mu0 = params['mu0']
    rho0 = params['rho0']
    Omega = params['Omega']
    beta_eff = params['beta_eff']

    r_grid, D1_mat, _ = chebyshev_lobatto(N_col, r1, r2)

    I = np.eye(N_col)
    Z = np.zeros((N_col, N_col))
    r_inv = np.diag(1.0 / r_grid)
    r_diff = np.diag(r_grid - r0)

    # Continuity:
    # lambda * eta + H0 * d(u_r)/dr + H0/r * u_r + i*m*H0/r * u_theta = 0
    A_eta_ur = -H0 * D1_mat - H0 * r_inv
    A_eta_ut = -1j * m_val * H0 * r_inv

    # Radial momentum:
    # lambda * H0 * u_r + 2*Omega*H0 * u_theta + g*H0*d(eta)/dr - beta_eff*(r-r0)*eta + B0/(mu0*rho0)*b_r = 0
    A_ur_eta = -g * H0 * D1_mat + beta_eff * r_diff
    A_ur_ut  = -2.0 * Omega * H0 * I
    A_ur_br  = -(B0 / (mu0 * rho0)) * I

    # Azimuthal momentum:
    # lambda * H0 * u_theta - 2*Omega*H0 * u_r + i*m*g*H0/r * eta + B0/(mu0*rho0)*b_theta = 0
    A_ut_eta = -1j * m_val * g * H0 * r_inv
    A_ut_ur  = 2.0 * Omega * H0 * I
    A_ut_bt  = -(B0 / (mu0 * rho0)) * I

    # Radial induction:
    # lambda * H0 * b_r + B0 * u_r = 0
    A_br_ur = -B0 * I

    # Azimuthal induction:
    # lambda * H0 * b_theta + B0 * u_theta = 0
    A_bt_ut = -B0 * I

    A = np.block([
        [Z,        A_eta_ur, A_eta_ut, Z,        Z],
        [A_ur_eta, Z,        A_ur_ut,  A_ur_br,  Z],
        [A_ut_eta, A_ut_ur,  Z,        Z,        A_ut_bt],
        [Z,        A_br_ur,  Z,        Z,        Z],
        [Z,        Z,        A_bt_ut,  Z,        Z]
    ])

    B = np.block([
        [I, Z,    Z,    Z,    Z],
        [Z, H0*I, Z,    Z,    Z],
        [Z, Z,    H0*I, Z,    Z],
        [Z, Z,    Z,    H0*I, Z],
        [Z, Z,    Z,    Z,    H0*I]
    ])

    # Enforce boundary conditions u_r(r1) = 0 and u_r(r2) = 0
    # u_r block corresponds to rows [N_col, 2*N_col - 1]
    bc_indices = [N_col + 0, N_col + N_col - 1]
    for idx in bc_indices:
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        B[idx, :] = 0.0

    return A, B


def solve_dimensional_eigenproblem(m_val, params, rel_tol=1e-4):
    """
    Solves the 5-field generalized eigenvalue problem A q = lambda B q.
    Converts lambda to omega = 1j * lambda.
    Filters infinite, non-physical, and complex modes.
    Returns sorted real physical angular frequencies in rad/s along with diagnostics.
    """
    A, B = build_dimensional_matrices(m_val, params)

    assert np.all(np.isfinite(A)), "Matrix A contains non-finite entries!"
    assert np.all(np.isfinite(B)), "Matrix B contains non-finite entries!"

    vals, vecs = slab.eig(A, B)

    n_total = len(vals)
    finite_mask = np.isfinite(vals) & (np.abs(vals) < 1e8)
    n_finite = np.sum(finite_mask)

    vals = vals[finite_mask]
    vecs = vecs[:, finite_mask]

    omegas = 1j * vals

    # Distinguish approximately real frequencies
    imag_mag = np.abs(omegas.imag)
    real_mag = np.abs(omegas.real)

    real_mask = imag_mag < rel_tol * (real_mag + 1e-10)
    n_real = np.sum(real_mask)
    n_complex = n_finite - n_real
    n_discarded = n_total - n_real

    real_omegas = omegas.real[real_mask]
    real_vecs = vecs[:, real_mask]

    # Sort by frequency
    sort_idx = np.argsort(real_omegas)
    sorted_omegas = real_omegas[sort_idx]
    sorted_vecs = real_vecs[:, sort_idx]

    # Validation check: verify b = -i * B0 / (omega * H0) * v for non-zero frequency modes
    validation_residuals = []
    H0 = params['H0']
    B0 = params['B0']
    N_col = params['N_col']

    for idx_w, w in enumerate(sorted_omegas):
        if abs(w) > 1e-8:
            q = sorted_vecs[:, idx_w]
            u_r = q[N_col:2*N_col]
            u_t = q[2*N_col:3*N_col]
            b_r = q[3*N_col:4*N_col]
            b_t = q[4*N_col:5*N_col]

            b_r_pred = -1j * (B0 / (w * H0)) * u_r
            b_t_pred = -1j * (B0 / (w * H0)) * u_t

            res_r = np.linalg.norm(b_r - b_r_pred) / (np.linalg.norm(b_r) + 1e-12)
            res_t = np.linalg.norm(b_t - b_t_pred) / (np.linalg.norm(b_t) + 1e-12)
            validation_residuals.append(0.5 * (res_r + res_t))

    diagnostics = {
        'n_total': n_total,
        'n_finite': n_finite,
        'n_real': n_real,
        'n_complex': n_complex,
        'n_discarded': n_discarded,
        'validation_residuals': validation_residuals
    }

    return sorted_omegas, sorted_vecs, diagnostics


def wkb_branches_dimensional(m_val, n_radial, params):
    """
    Computes WKB branch estimates in physical angular frequencies (rad/s)
    derived from the local quartic dispersion relation.
    """
    r1 = params['r1']
    r2 = params['r2']
    r0 = params['r0']
    H0 = params['H0']
    g = params['g']
    B0 = params['B0']
    mu0 = params['mu0']
    rho0 = params['rho0']
    Omega = params['Omega']
    beta_eff = params['beta_eff']
    omega_A2 = params['omega_A2']
    c0sq = params['c0sq']

    kr = n_radial * np.pi / (r2 - r1)

    # ------------------------------------------------------------
    # Evaluate MP/MS at reference radius r = r0
    # ------------------------------------------------------------
    r_val = r0
    kappa2_1 = kr**2 + (m_val / r_val)**2

    R_1 = beta_eff * (2.0 * r_val - r0) / r_val
    K_1 = c0sq * kappa2_1 + R_1

    # Quartic at r = r0:
    # w^4 + A2*w^2 + A1*w + A0 = 0
    # At r = r0, A1 = 0.
    A2 = 2.0 * omega_A2 - 4.0 * Omega**2 + K_1
    A1 = 2.0 * Omega * m_val * beta_eff * (r_val - r0) / r_val
    A0 = omega_A2 * (omega_A2 + K_1)

    roots_r0 = np.roots([1.0, 0.0, A2, A1, A0])
    # Extract real roots
    real_roots_r0 = np.sort([r.real for r in roots_r0 if abs(r.imag) < 1e-6 * (abs(r.real) + 1e-10)])

    pos_roots = sorted([r for r in real_roots_r0 if r > 0])
    neg_roots = sorted([r for r in real_roots_r0 if r < 0])

    oMP_p = pos_roots[-1] if len(pos_roots) > 0 else np.nan
    oMP_m = neg_roots[0] if len(neg_roots) > 0 else np.nan

    oMS_p = pos_roots[0] if len(pos_roots) > 1 else (pos_roots[0] if len(pos_roots) == 1 and pos_roots[0] < oMP_p else np.nan)
    oMS_m = neg_roots[-1] if len(neg_roots) > 1 else (neg_roots[-1] if len(neg_roots) == 1 and neg_roots[0] > oMP_m else np.nan)
    oMS_val = oMS_m

    # ------------------------------------------------------------
    # Rossby branch evaluated across the radial domain
    # ------------------------------------------------------------
    r_grid = np.linspace(r1, r2, 10)
    oR_arr = []

    for r in r_grid:
        R_r = beta_eff * (2.0 * r - r0) / r
        S_r = 2.0 * Omega * m_val * beta_eff * (r - r0) / r
        kappa2_r = kr**2 + (m_val / r)**2
        K_r = c0sq * kappa2_r + R_r

        A2_r = 2.0 * omega_A2 - 4.0 * Omega**2 + K_r
        A1_r = S_r
        A0_r = omega_A2 * (omega_A2 + K_r)

        roots_r = np.roots([1.0, 0.0, A2_r, A1_r, A0_r])
        real_roots_r = [rt.real for rt in roots_r if abs(rt.imag) < 1e-6 * (abs(rt.real) + 1e-10)]

        if B0 == 0:
            if abs(4.0 * Omega**2 + K_r) > 1e-14:
                oR_val = -S_r / (4.0 * Omega**2 + K_r)
            else:
                oR_val = np.nan
        else:
            if abs(S_r) > 1e-12:
                oR_val = (4.0 * omega_A2**2 + omega_A2 * (K_r - 4.0 * Omega**2)) / S_r if abs(S_r) > 1e-12 else np.nan
            else:
                slow_r = [rt for rt in real_roots_r if abs(rt) < 0.5 * min(np.sqrt(c0sq * kappa2_r), 2*abs(Omega) + 1e-5)]
                oR_val = np.mean(slow_r) if len(slow_r) > 0 else np.nan

        oR_arr.append(oR_val)

    oR_arr = np.asarray(oR_arr)

    # ------------------------------------------------------------
    # Magneto-Kelvin branches (dimensional)
    # ------------------------------------------------------------
    c0 = params['c0']
    arg_in = (m_val * c0 / r1)**2 - 4.0 * omega_A2
    if arg_in >= 0.0:
        oMK_in = -0.5 * np.sqrt(arg_in)
    else:
        oMK_in = np.nan

    arg_out = (m_val * c0 / r2)**2 - 4.0 * omega_A2
    if arg_out >= 0.0:
        oMK_out = +0.5 * np.sqrt(arg_out)
    else:
        oMK_out = np.nan

    return oMP_p, oMP_m, oMK_in, oMK_out, oR_arr, oMS_val


def assign_branches_global_dimensional(eigs, m_val, params, N_max=20):
    """
    Assigns collocation eigenvalues to physical WKB branches using global minimum distance.
    All calculations are strictly dimensional in rad/s.
    """
    wkb_preds = {}

    _, _, oMK_in, oMK_out, _, _ = wkb_branches_dimensional(m_val, 1, params)

    if not np.isnan(oMK_in):
        wkb_preds[('MK_in', 0)] = oMK_in
    if not np.isnan(oMK_out):
        wkb_preds[('MK_out', 0)] = oMK_out

    for n in range(1, N_max + 1):
        oMP_p, oMP_m, _, _, oR_arr, oMS_val = wkb_branches_dimensional(m_val, n, params)
        if not np.isnan(oMP_p): wkb_preds[('MP_p', n)] = oMP_p
        if not np.isnan(oMP_m): wkb_preds[('MP_m', n)] = oMP_m
        if np.any(~np.isnan(oR_arr)): wkb_preds[('R', n)] = oR_arr
        if not np.isnan(oMS_val): wkb_preds[('MS', n)] = oMS_val

    distances = []
    for i, eig in enumerate(eigs):
        for branch_id, wkb_vals in wkb_preds.items():
            if isinstance(wkb_vals, np.ndarray):
                valid_vals = wkb_vals[~np.isnan(wkb_vals)]
                if len(valid_vals) > 0:
                    dist = np.min(np.abs(eig - valid_vals))
                    distances.append((dist, i, branch_id))
            else:
                dist = abs(eig - wkb_vals)
                distances.append((dist, i, branch_id))

    distances.sort(key=lambda x: x[0])

    assigned_eigs = set()
    assigned_branches = set()
    matches = {}

    for dist, eig_idx, branch_id in distances:
        if eig_idx not in assigned_eigs and branch_id not in assigned_branches:
            assigned_eigs.add(eig_idx)
            assigned_branches.add(branch_id)
            matches[branch_id] = eigs[eig_idx]

    results = {
        'MK_in': None,
        'MK_out': None,
        'MP_p': {},
        'MP_m': {},
        'R': {},
        'MS': {}
    }

    for branch_id, eig_val in matches.items():
        b_name, n = branch_id
        if b_name == 'MK_in':
            results['MK_in'] = eig_val
        elif b_name == 'MK_out':
            results['MK_out'] = eig_val
        else:
            results[b_name][n] = eig_val

    return results


def legacy_dimensionless_solver(m_val, params):
    """
    Original legacy dimensionless solver for regression testing at non-zero Omega.
    Converts returned dimensionless frequencies hat_omega back to dimensional frequencies omega = 2 * Omega * hat_omega.
    """
    Omega = params['Omega']
    if Omega == 0:
        return np.array([])

    H0 = params['H0']
    g = params['g']
    B0 = params['B0']
    rho0 = params['rho0']
    C = params['C']
    r1 = params['r1']
    r2 = params['r2']
    r0 = params['r0']
    mu0 = params['mu0']
    N_col = params['N_col']

    VA2 = B0**2 / (mu0 * rho0)
    c0sq = g * H0
    f_scale = 2.0 * Omega
    hat_r1 = r1 / r0
    hat_r2 = r2 / r0
    hat_c0sq = c0sq / (Omega**2 * r0**2)
    gamma = 2.0 * C / (Omega**2 * r0**3)
    hat_omA = np.sqrt(VA2) / (f_scale * H0)
    hat_omA2 = hat_omA**2

    hat_r_grid, D1_mat, D2_mat = chebyshev_lobatto(N_col, hat_r1, hat_r2)

    def omega_star_hat(hat_omega):
        return hat_omega + hat_omA2 / hat_omega

    def build_matrix_legacy(hat_omega):
        ws_hat = omega_star_hat(hat_omega)
        Pv = 1.0 / hat_r_grid - (gamma) * (hat_r_grid - 1.0) / hat_c0sq
        term1 = 4.0 * hat_omega * (ws_hat**2 - 1.0) / (ws_hat * hat_c0sq)
        term2 = -m_val**2 / hat_r_grid**2
        term3 = -(gamma) * (2.0 * hat_r_grid - 1.0) / (hat_c0sq * hat_r_grid)
        term4 = -m_val * (gamma) * (hat_r_grid - 1.0) / (ws_hat * hat_c0sq * hat_r_grid)
        Qv = term1 + term2 + term3 + term4

        L = D2_mat + np.diag(Pv) @ D1_mat + np.diag(Qv)
        for idx in [0, N_col - 1]:
            rb = hat_r_grid[idx]
            bc_eta_coeff = -(ws_hat * (gamma) * (rb - 1.0) + m_val * hat_c0sq / rb)
            L[idx, :] = ws_hat * hat_c0sq * D1_mat[idx, :] + bc_eta_coeff * np.eye(N_col)[idx]
        return L

    scan = np.linspace(-45, 45, 1000)
    scan = scan[np.abs(scan) > 0.01]
    ld = []
    for w in scan:
        try:
            d = np.log(np.abs(slab.det(build_matrix_legacy(w))) + 1e-300)
        except Exception:
            d = np.nan
        ld.append(d)

    roots = []
    for k in range(1, len(ld) - 1):
        if np.isfinite(ld[k]) and ld[k] < ld[k-1] and ld[k] < ld[k+1]:
            w = complex(scan[k])
            for _ in range(40):
                dw = 1e-6 * abs(w) + 1e-10
                try:
                    f0 = slab.det(build_matrix_legacy(w))
                    fp = ((slab.det(build_matrix_legacy(w+dw)) - slab.det(build_matrix_legacy(w-dw))) / (2*dw))
                    if abs(fp) < 1e-300: break
                    step = -f0 / fp
                    w += step
                    if abs(step) < 1e-10 * (abs(w) + 1): break
                except Exception:
                    break
            try:
                residual = abs(slab.det(build_matrix_legacy(w)))
                valid_ld = [x for x in ld if np.isfinite(x)]
                ld_max = max(valid_ld) if len(valid_ld) > 0 else 0
                if residual < 1e-3 * np.exp(ld_max):
                    dup = any(abs(w - wr) < 5e-3 for wr in roots)
                    if not dup:
                        roots.append(w)
            except Exception:
                pass

    hat_omegas = np.array(sorted([wr.real for wr in roots]))
    return 2.0 * Omega * hat_omegas


def run_validation_tests():
    """
    Executes mandatory validation tests A-G as specified in the requirements.
    """
    print("\n" + "=" * 70)
    print("  RUNNING MANDATORY VALIDATION TESTS A - G")
    print("=" * 70)

    # Test A: Finite rotation
    params_A = physical_parameters(Omega=0.5e-4, C=0.0)
    eigs_A, _, diag_A = solve_dimensional_eigenproblem(m_val=1, params=params_A)
    print(f"\n[Test A: Finite Rotation (Omega={params_A['Omega']} rad/s)]")
    print(f"  Total eigenvalues: {diag_A['n_total']}, Finite: {diag_A['n_finite']}, Real: {diag_A['n_real']}")
    print(f"  Sample lowest positive frequencies [rad/s]: {np.sort([e for e in eigs_A if e > 0])[:4]}")
    assert len(eigs_A) > 0, "Test A failed: no eigenvalues found!"

    # Test B: Zero rotation
    params_B = physical_parameters(Omega=0.0, C=0.0)
    eigs_B, _, diag_B = solve_dimensional_eigenproblem(m_val=1, params=params_B)
    print(f"\n[Test B: Zero Rotation (Omega=0.0)]")
    print(f"  Total eigenvalues: {diag_B['n_total']}, Finite: {diag_B['n_finite']}, Real: {diag_B['n_real']}")
    print(f"  Sample lowest positive frequencies [rad/s]: {np.sort([e for e in eigs_B if e > 0])[:4]}")
    assert len(eigs_B) > 0, "Test B failed: solver failed at zero rotation!"

    # Test C: Non-magnetic limit
    params_C = physical_parameters(B0=0.0, C=0.0)
    eigs_C, _, diag_C = solve_dimensional_eigenproblem(m_val=1, params=params_C)
    print(f"\n[Test C: Non-Magnetic Limit (B0=0.0)]")
    print(f"  Total eigenvalues: {diag_C['n_total']}, Finite: {diag_C['n_finite']}, Real: {diag_C['n_real']}")
    print(f"  Sample lowest positive frequencies [rad/s]: {np.sort([e for e in eigs_C if e > 0])[:4]}")
    assert len(eigs_C) > 0, "Test C failed: non-magnetic limit failed!"

    # Test D: Zero central potential
    params_D = physical_parameters(C=0.0)
    eigs_D, _, diag_D = solve_dimensional_eigenproblem(m_val=1, params=params_D)
    print(f"\n[Test D: Zero Central Potential (C=0.0)]")
    print(f"  Total eigenvalues: {diag_D['n_total']}, Finite: {diag_D['n_finite']}, Real: {diag_D['n_real']}")
    print(f"  beta_eff = {params_D['beta_eff']}")
    assert params_D['beta_eff'] == 0.0, "Test D failed: beta_eff non-zero!"
    assert len(eigs_D) > 0, "Test D failed: zero potential limit failed!"

    # Test E: Combined limit
    params_E = physical_parameters(Omega=0.0, B0=0.0, C=0.0)
    eigs_E, _, diag_E = solve_dimensional_eigenproblem(m_val=1, params=params_E)
    print(f"\n[Test E: Combined Limit (Omega=0, B0=0, C=0)]")
    print(f"  Total eigenvalues: {diag_E['n_total']}, Finite: {diag_E['n_finite']}, Real: {diag_E['n_real']}")
    print(f"  Sample lowest positive frequencies [rad/s]: {np.sort([e for e in eigs_E if e > 0])[:4]}")
    assert len(eigs_E) > 0, "Test E failed: combined limit failed!"

    # Test F: Legacy solver comparison at non-zero Omega
    print(f"\n[Test F: Legacy Solver vs New 5-Field Dimensional Solver Comparison]")
    legacy_eigs = legacy_dimensionless_solver(m_val=1, params=params_A)
    # Filter legacy modes > 1e-4 rad/s to exclude determinant search artifacts near w=0
    leg_physical = sorted([e for e in legacy_eigs if e > 1e-4])
    dim_physical = sorted([e for e in eigs_A if e > 1e-4])
    print(f"  Legacy physical frequencies [rad/s] (first 4): {leg_physical[:4]}")
    print(f"  New solver physical frequencies [rad/s] (first 4): {dim_physical[:4]}")

    # Match closest frequencies for verification
    diffs = []
    for leg_w in leg_physical[:4]:
        closest_dim = min(dim_physical, key=lambda x: abs(x - leg_w))
        rel_diff = abs(closest_dim - leg_w) / abs(leg_w)
        diffs.append(rel_diff)
        print(f"    Legacy: {leg_w:.6e} rad/s <-> New: {closest_dim:.6e} rad/s (Rel Diff: {rel_diff:.2e})")
    assert np.max(diffs) < 2e-2, "Test F failed: discrepancy between legacy and new dimensional solver!"

    # Test G: WKB vs Global Solver tabular comparison
    print(f"\n[Test G: WKB vs Global Solver Tabular Comparison (m=1)]")
    assigned_A = assign_branches_global_dimensional(eigs_A, m_val=1, params=params_A)
    print(f"{'m':<4} {'Branch':<10} {'WKB omega [rad/s]':<20} {'Global omega [rad/s]':<22} {'Rel Error':<12}")
    print("-" * 70)

    wkb_mp_p, wkb_mp_m, wkb_mk_in, wkb_mk_out, wkb_r_arr, wkb_ms = wkb_branches_dimensional(1, 1, params_A)

    table_rows = [
        ('MP+', wkb_mp_p, assigned_A['MP_p'].get(1)),
        ('MP-', wkb_mp_m, assigned_A['MP_m'].get(1)),
        ('MK_in', wkb_mk_in, assigned_A['MK_in']),
        ('MK_out', wkb_mk_out, assigned_A['MK_out']),
        ('MS', wkb_ms, assigned_A['MS'].get(1))
    ]

    for b_name, wkb_val, global_val in table_rows:
        if wkb_val is not None and global_val is not None and not np.isnan(wkb_val):
            rel_err = abs(global_val - wkb_val) / abs(wkb_val)
            print(f"{1:<4} {b_name:<10} {wkb_val:<20.6e} {global_val:<22.6e} {rel_err:<12.2e}")
        else:
            w_str = f"{wkb_val:.6e}" if wkb_val is not None and not np.isnan(wkb_val) else "N/A"
            g_str = f"{global_val:.6e}" if global_val is not None else "N/A"
            print(f"{1:<4} {b_name:<10} {w_str:<20} {g_str:<22} {'N/A':<12}")

    print("=" * 70 + "\n")


def plot_dimensional_spectrum(results, params, M_max=30):
    """
    Plots the dimensional dispersion spectrum:
    - Primary Y-axis: physical angular frequency omega [rad/s].
    - Secondary Y-axis: normalized frequency omega / (2*Omega) (when Omega != 0).
    - Title explicitly includes 'Dimensional'.
    """
    m_arr = np.arange(1, M_max + 1)
    m_fine = np.linspace(1, M_max, 1000)
    Omega = params['Omega']

    # Compute fine WKB curves
    WKB_MK_in = np.array([wkb_branches_dimensional(m, 1, params)[2] for m in m_fine])
    WKB_MK_out = np.array([wkb_branches_dimensional(m, 1, params)[3] for m in m_fine])
    WKB_MP_p = {}
    WKB_MP_m = {}
    WKB_R = {}
    WKB_MS = {}

    for n in range(1, 10):
        op_list, om_list, oR_list, oMS_list = [], [], [], []
        for m in m_fine:
            op, om, _, _, oR_arr, oMS = wkb_branches_dimensional(m, n, params)
            op_list.append(op)
            om_list.append(om)
            oR_list.append(oR_arr)
            oMS_list.append(oMS)

        WKB_MP_p[n] = np.array(op_list)
        WKB_MP_m[n] = np.array(om_list)
        WKB_R[n] = np.array(oR_list).T
        WKB_MS[n] = np.array(oMS_list)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.6, 1]})

    fig.suptitle(
        r'SWMHD Dimensional Global Dispersion Relation'
        '\n'
        r'Magneto-Poincaré (blue), Magneto-Kelvin (green), Rossby (red), Magnetostrophic (orange)',
        fontsize=13, fontweight='bold')

    COLORS = {
        'MP': '#1f77b4',
        'MK': '#2ca02c',
        'R':  '#d62728',
        'MS': '#ff7f0e',
    }

    def draw_panel(ax_obj, zoom=False):
        # WKB curves
        ax_obj.plot(m_fine, WKB_MK_in, color=COLORS['MK'], lw=2.5,
                    label='Magneto-Kelvin (WKB)', zorder=4)
        ax_obj.plot(m_fine, WKB_MK_out, color=COLORS['MK'], lw=2.5, zorder=4)

        for n in range(1, 9):
            lbl_MP = f'Magneto-Poincaré $n={n}$ (WKB)' if n <= 2 and not zoom else ('Magneto-Poincaré (WKB)' if n == 1 else None)
            lbl_R = f'Rossby $n={n}$ (WKB)' if n <= 2 and not zoom else ('Rossby (WKB)' if n == 1 else None)
            lbl_MS = f'Magnetostrophic $n={n}$ (WKB)' if n <= 2 and not zoom else ('Magnetostrophic (WKB)' if n == 1 else None)

            alpha_val = max(0.1, 0.4 - 0.05 * (n - 1))

            ax_obj.plot(m_fine, WKB_MP_p[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MP, zorder=3)
            ax_obj.plot(m_fine, WKB_MP_m[n], color=COLORS['MP'], lw=1.5, ls='-', alpha=alpha_val, zorder=3)
            ax_obj.plot(m_fine, WKB_MS[n], color=COLORS['MS'], lw=1.5, ls='-', alpha=alpha_val, label=lbl_MS, zorder=3)

            for i in range(1):
                lR = lbl_R if i == 0 else None
                ax_obj.plot(m_fine, WKB_R[n][i], color=COLORS['R'], lw=1.0, ls='-', alpha=alpha_val, label=lR, zorder=3)

        # Collocation points
        mk_in_data = results['mk_in_data']
        mk_out_data = results['mk_out_data']

        if mk_in_data:
            mk_m, mk_w = zip(*mk_in_data)
            ax_obj.scatter(mk_m, mk_w, color=COLORS['MK'], marker='s', s=55, zorder=6,
                           label='Magneto-Kelvin (collocation)')
        if mk_out_data:
            mk_m, mk_w = zip(*mk_out_data)
            ax_obj.scatter(mk_m, mk_w, color=COLORS['MK'], marker='s', s=55, zorder=6)

        def plot_branch_collocation(data_dict, color, label_prefix):
            for n in sorted(data_dict.keys()):
                if not data_dict[n]: continue
                mp_m, mp_w = zip(*data_dict[n])
                lbl = f'{label_prefix} (collocation)' if n == 1 else None
                ax_obj.scatter(mp_m, mp_w, color=color, marker='o', s=30, zorder=5, alpha=0.85, label=lbl)

        plot_branch_collocation(results['mp_p_data'], COLORS['MP'], 'Magneto-Poincaré')
        plot_branch_collocation(results['mp_m_data'], COLORS['MP'], 'Magneto-Poincaré')
        plot_branch_collocation(results['r_data'], COLORS['R'], 'Rossby')
        plot_branch_collocation(results['ms_data'], COLORS['MS'], 'Magnetostrophic')

        ax_obj.axhline(0, color='grey', lw=0.8, ls=':')
        if Omega != 0:
            ax_obj.axhline(+2 * Omega, color='grey', lw=0.9, ls='--', alpha=0.5)
            ax_obj.axhline(-2 * Omega, color='grey', lw=0.9, ls='--', alpha=0.5)

        omega_A = params['omega_A']
        if omega_A > 0:
            ax_obj.axhline(+omega_A, color='purple', lw=0.8, ls=':', alpha=0.6)
            ax_obj.axhline(-omega_A, color='purple', lw=0.8, ls=':', alpha=0.6)

        ax_obj.set_xlabel(r'Azimuthal wavenumber $m$', fontsize=13)
        ax_obj.set_ylabel(r'Frequency $\omega$ [rad s$^{-1}$]', fontsize=13)
        ax_obj.set_xlim(0.7, M_max + 0.5)
        ax_obj.set_xticks(m_arr[::2])
        ax_obj.grid(True, alpha=0.22)

        # Add secondary right Y-axis for normalized frequency omega / (2*Omega) if Omega != 0
        if Omega != 0:
            secax = ax_obj.secondary_yaxis(
                'right',
                functions=(lambda w: w / (2.0 * Omega), lambda hat_w: hat_w * (2.0 * Omega))
            )
            secax.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=13)

        if zoom:
            if Omega != 0:
                ax_obj.set_ylim(-1.5 * (2.0 * Omega), 1.5 * (2.0 * Omega))
            else:
                ax_obj.set_ylim(-1e-4, 1e-4)
        else:
            if Omega != 0:
                ax_obj.set_ylim(-50 * (2.0 * Omega), 50 * (2.0 * Omega))
            else:
                ax_obj.set_ylim(-0.01, 0.01)

    draw_panel(ax, zoom=False)
    ax.set_title('Full spectrum (Dimensional)', fontsize=11)

    draw_panel(ax2, zoom=True)
    ax2.set_title(r'Slow branches zoom (Rossby & MS, Dimensional)', fontsize=11)

    # Figure legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    seen = set()
    legend_handles = []
    legend_labels = []

    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l)
            legend_handles.append(h)
            legend_labels.append(l)

    fig.legend(
        legend_handles,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=13,
        frameon=True,
        borderaxespad=0.5,
        labelspacing=0.8,
        handlelength=2.2
    )

    # Parameter box displaying SI units
    pbox = (
        rf"$\Omega = {params['Omega']:.1e}\ \mathrm{{rad/s}}$" "\n"
        rf"$H_0 = {params['H0']:.1f}\ \mathrm{{m}}$" "\n"
        rf"$B_0 = {params['B0']:.1e}\ \mathrm{{T}}$" "\n"
        rf"$\rho_0 = {params['rho0']:.1f}\ \mathrm{{kg/m^3}}$" "\n"
        rf"$C = {params['C']:.1e}\ \mathrm{{m^3/s^2}}$" "\n"
        rf"$\beta_{{\mathrm{{eff}}}} = {params['beta_eff']:.1e}\ \mathrm{{s^{{-2}}m^{{-1}}}}$" "\n"
        rf"$\omega_A = {params['omega_A']:.4e}\ \mathrm{{rad/s}}$" "\n"
        rf"$r_1 = {params['r1']:.1e}\ \mathrm{{m}},\ r_2 = {params['r2']:.1e}\ \mathrm{{m}}$" "\n"
        rf"$N = {params['N_col']}$"
    )

    fig.text(0.52, 0.015, pbox, fontsize=10, va='bottom', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))

    fig.subplots_adjust(right=0.78)
    plt.tight_layout(rect=[0, 0.12, 0.78, 1])

    filename_png = f"swmhd_dimensional_dispersion_g={params['g']}_C={params['C']}_B={params['B0']}.png"
    plt.savefig(filename_png, dpi=180, bbox_inches='tight')
    print(f"\nSaved dimensional plot: {filename_png}")


def main():
    print("=" * 70)
    print("  SWMHD DIMENSIONAL GLOBAL DISPERSION SOLVER")
    print("=" * 70)

    # 1. Physical Parameters
    params = physical_parameters()

    print("  System Parameters (Dimensional SI Units):")
    print(f"    Omega    : {params['Omega']:.4e} rad/s")
    print(f"    H0       : {params['H0']:.2f} m")
    print(f"    g        : {params['g']:.2f} m/s^2")
    print(f"    B0       : {params['B0']:.4e} T")
    print(f"    rho0     : {params['rho0']:.2f} kg/m^3")
    print(f"    C        : {params['C']:.4e} m^3/s^2")
    print(f"    beta_eff : {params['beta_eff']:.4e} s^-2 m^-1")
    print(f"    omega_A  : {params['omega_A']:.4e} rad/s")
    print(f"    c0       : {params['c0']:.2f} m/s")
    print(f"    Domain   : r in [{params['r1']:.2e}, {params['r2']:.2e}] m")
    print("-" * 70)

    # 2. Run Validation Tests A - G
    run_validation_tests()

    # 3. Compute Spectrum across m = 1 .. M_max
    M_max = 30
    m_arr = np.arange(1, M_max + 1)

    print(f"Running dimensional 5-field solver N={params['N_col']}, m=1..{M_max} ...")

    results = {
        'mk_in_data': [],
        'mk_out_data': [],
        'mp_p_data': {},
        'mp_m_data': {},
        'r_data': {},
        'ms_data': {}
    }

    for m in m_arr:
        eigs, _, _ = solve_dimensional_eigenproblem(m, params)
        assigned = assign_branches_global_dimensional(eigs, m, params)

        if assigned['MK_in'] is not None:
            results['mk_in_data'].append((m, assigned['MK_in']))
        if assigned['MK_out'] is not None:
            results['mk_out_data'].append((m, assigned['MK_out']))

        for n, val in assigned['MP_p'].items():
            results['mp_p_data'].setdefault(n, []).append((m, val))
        for n, val in assigned['MP_m'].items():
            results['mp_m_data'].setdefault(n, []).append((m, val))
        for n, val in assigned['R'].items():
            results['r_data'].setdefault(n, []).append((m, val))
        for n, val in assigned['MS'].items():
            results['ms_data'].setdefault(n, []).append((m, val))

        pos = sorted([e for e in eigs if e > 0])
        neg = sorted([e for e in eigs if e < 0], reverse=True)
        if m <= 3 or m % 5 == 0:
            print(f"  m={m:2d}: +{[f'{x:.3e}' for x in pos[:4]]}... "
                  f"-{[f'{abs(x):.3e}' for x in neg[:4]]}...")

    # 4. Plot Spectrum
    plot_dimensional_spectrum(results, params, M_max=M_max)


if __name__ == "__main__":
    main()
