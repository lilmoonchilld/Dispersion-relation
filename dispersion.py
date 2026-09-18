#!/usr/bin/env python3
"""
Mathematica-style direct dispersion solver for the C != 0 annular SWMHD system.

For m = 1,...,30 the calculation is

    initial guess / previous-m root
        -> omega_*(omega)
        -> K^2(omega,m)
        -> Whittaker parameters
        -> Whittaker M/W basis
        -> 2 x 2 wall matrix
        -> determinant D(omega,m)
        -> direct mpmath FindRoot-style solve

No frequency grid is used for the eigenvalue search.
No SVD search is used for the eigenvalue search.
No determinant-minimum scan is used.
No Chebyshev collocation is used.
No finite-difference radial ODE solver is used.

Branch smoothing:
    roots are found at m=1 from direct initial guesses;
    for m=2,...,30 the previous root is used as the next FindRoot seed.
    This is direct continuation, not a frequency scan.

Radial wavenumber used for mode labelling:

    k_r^2(r) = K^2 + Lambda/r - m^2/r^2

where Lambda is the Whittaker Lambda of the present SWMHD system.
For the initial m=1 roots, n is assigned by increasing positive
k_r(r0), separately for omega>0 and omega<0. The label is then
continued with the branch.

Plots:
    1. full dispersion diagram: omega/(2 Omega) vs m
    2. slow-branch zoom: -1 <= omega/(2 Omega) <= 1

Positive-frequency branches are red.
Negative-frequency branches are blue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Numerical precision
# ============================================================
mp.mp.dps = 50


# ============================================================
# PHYSICAL SYSTEM
# ============================================================
@dataclass(frozen=True)
class Config:
    MU0: mp.mpf = mp.mpf("4.0") * mp.pi * mp.mpf("1e-7")
    Omega: mp.mpf = mp.mpf("0.5e-4")
    H0: mp.mpf = mp.mpf("500.0")
    g: mp.mpf = mp.mpf("9.81")
    B0: mp.mpf = mp.mpf("8.5e-4")
    rho0: mp.mpf = mp.mpf("1000.0")
    r1: mp.mpf = mp.mpf("0.5e6")
    r2: mp.mpf = mp.mpf("1.0e6")
    C: mp.mpf = mp.mpf("1.0e8")

    @property
    def f(self) -> mp.mpf:
        return 2 * self.Omega

    @property
    def r0(self) -> mp.mpf:
        return (self.r1 + self.r2) / 2


CFG = Config()


# ============================================================
# DIRECT FindRoot INITIAL GUESSES
# These are starting guesses only, not a frequency grid.
# They are used only to obtain the m=1 roots.
# ============================================================
INITIAL_GUESS_NORMALIZED = (
    -12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0,
    -1.5, -1.2, -0.8, -0.5, -0.25, -0.10, -0.03,
     0.03,  0.10,  0.25,  0.5,  0.8, 1.2, 1.5,
     2.0,  3.0,  4.0,  6.0, 8.0, 10.0, 12.0,
)

SECOND_GUESS_OFFSET = mp.mpf("1e-5")

N_RADIAL_MODES = 12
REAL_ROOT_IMAG_TOL = mp.mpf("1e-8")
ROOT_DUPLICATE_TOL = mp.mpf("1e-7")
ROOT_RESIDUAL_TOL = mp.mpf("1e-12")
PRINT_DIGITS = 14


# ============================================================
# BASIC SYSTEM QUANTITIES
# ============================================================
def omega_A_squared(cfg: Config) -> mp.mpf:
    return cfg.B0**2 / (cfg.MU0 * cfg.rho0 * cfg.H0**2)


def omega_star(omega: mp.mpc, cfg: Config) -> mp.mpc:
    if omega == 0:
        raise ZeroDivisionError("omega = 0")
    return omega + omega_A_squared(cfg) / omega


def beta_eff(cfg: Config) -> mp.mpf:
    return cfg.Omega**2 + 2 * cfg.C / cfg.r0**3


def G_factor(omega: mp.mpc, cfg: Config) -> mp.mpc:
    os = omega_star(omega, cfg)
    denom = omega * (os**2 - cfg.f**2)
    if denom == 0:
        raise ZeroDivisionError("singular omega")
    return 1 + 4 * omega_A_squared(cfg) * os / denom


# ============================================================
# EXACT SYSTEM K^2 AND WHITTAKER LAMBDA
# ============================================================
def K_squared(omega: mp.mpc, m: int, cfg: Config) -> mp.mpc:
    os = omega_star(omega, cfg)
    if os == 0:
        raise ZeroDivisionError("omega_star = 0")

    G = G_factor(omega, cfg)

    return (
        omega * (os**2 - cfg.f**2)
        - m * cfg.f * beta_eff(cfg) * G
    ) / (
        cfg.g * cfg.H0 * os
    )


def Lambda_whittaker(omega: mp.mpc, m: int, cfg: Config) -> mp.mpc:
    os = omega_star(omega, cfg)
    if os == 0:
        raise ZeroDivisionError("omega_star = 0")

    G = G_factor(omega, cfg)

    return (
        3 * m * cfg.f * cfg.C * G
        / (
            cfg.g
            * cfg.H0
            * os
            * cfg.r0**2
        )
    )


# ============================================================
# WHITTAKER PARAMETERS
# NOTE: kappa_r here is the auxiliary Whittaker transformation
# parameter. It is NOT the physical local k_r in the thesis.
# ============================================================
def whittaker_parameters(omega: mp.mpc, m: int, cfg: Config):
    k2 = K_squared(omega, m, cfg)
    lam = Lambda_whittaker(omega, m, cfg)

    if abs(k2) == 0:
        raise ZeroDivisionError("K^2 = 0 Whittaker transition")

    if abs(mp.im(k2)) < mp.mpf("1e-30") and mp.re(k2) > 0:
        kappa_r = 1j * mp.sqrt(mp.re(k2))
    elif abs(mp.im(k2)) < mp.mpf("1e-30") and mp.re(k2) < 0:
        kappa_r = mp.sqrt(-mp.re(k2))
    else:
        kappa_r = mp.sqrt(-k2)

    kappa_W = lam / (2 * kappa_r)

    return k2, kappa_r, kappa_W, lam


# ============================================================
# WHITTAKER M/W AND DERIVATIVES
# ============================================================
def whittaker_M(kappa_W, mu: int, x):
    a = mu - kappa_W + mp.mpf("0.5")
    b = 2 * mu + 1
    return (
        mp.exp(-x / 2)
        * x ** (mu + mp.mpf("0.5"))
        * mp.hyp1f1(a, b, x)
    )


def whittaker_W(kappa_W, mu: int, x):
    a = mu - kappa_W + mp.mpf("0.5")
    b = 2 * mu + 1
    return (
        mp.exp(-x / 2)
        * x ** (mu + mp.mpf("0.5"))
        * mp.hyperu(a, b, x)
    )


def whittaker_M_x(kappa_W, mu: int, x):
    a = mu - kappa_W + mp.mpf("0.5")
    b = 2 * mu + 1

    base = mp.hyp1f1(a, b, x)
    dbase = (a / b) * mp.hyp1f1(a + 1, b + 1, x)
    pref = mp.exp(-x / 2) * x ** (mu + mp.mpf("0.5"))

    return pref * (
        dbase
        + ((mu + mp.mpf("0.5")) / x - mp.mpf("0.5")) * base
    )


def whittaker_W_x(kappa_W, mu: int, x):
    a = mu - kappa_W + mp.mpf("0.5")
    b = 2 * mu + 1

    base = mp.hyperu(a, b, x)
    dbase = -a * mp.hyperu(a + 1, b + 1, x)
    pref = mp.exp(-x / 2) * x ** (mu + mp.mpf("0.5"))

    return pref * (
        dbase
        + ((mu + mp.mpf("0.5")) / x - mp.mpf("0.5")) * base
    )


# ============================================================
# ANALYTICAL RADIAL BASIS
# ============================================================
def whittaker_basis(omega, m: int, cfg: Config, r):
    _, kappa_r, kappa_W, lam = whittaker_parameters(
        omega, m, cfg
    )

    x = 2 * kappa_r * r

    M = whittaker_M(kappa_W, m, x)
    W = whittaker_W(kappa_W, m, x)
    Mx = whittaker_M_x(kappa_W, m, x)
    Wx = whittaker_W_x(kappa_W, m, x)

    return M, W, Mx, Wx, kappa_r, kappa_W, lam


# ============================================================
# 2 x 2 ANALYTICAL BOUNDARY MATRIX
# ============================================================
def whittaker_boundary_matrix(omega, m: int, cfg: Config):
    os = omega_star(omega, cfg)
    rows = []

    for r in (cfg.r1, cfg.r2):
        M, W, Mx, Wx, kappa_r, _, _ = whittaker_basis(
            omega, m, cfg, r
        )

        prefactor = (os / 2 + m * cfg.f) / r

        BM = (
            2 * kappa_r * os * Mx
            - prefactor * M
        )

        BW = (
            2 * kappa_r * os * Wx
            - prefactor * W
        )

        rows.append((BM, BW))

    return rows


# ============================================================
# MATHEMATICA DISPERSION FUNCTION
# ============================================================
def dispersion_determinant(omega, m: int, cfg: Config):
    M = whittaker_boundary_matrix(omega, m, cfg)

    M11, M12 = M[0]
    M21, M22 = M[1]

    return M11 * M22 - M12 * M21


# ============================================================
# DIRECT FindRoot EQUIVALENT
# ============================================================
def direct_findroot(omega_guess, m: int, cfg: Config):
    z0 = mp.mpc(omega_guess)
    z1 = z0 * (1 + SECOND_GUESS_OFFSET)

    root = mp.findroot(
        lambda z: dispersion_determinant(z, m, cfg),
        (z0, z1),
        solver="secant",
        tol=mp.mpf("1e-25"),
        maxsteps=100,
        verify=False,
    )

    return mp.mpc(root)


# ============================================================
# ROOT UTILITIES
# ============================================================
def is_finite_complex(z) -> bool:
    try:
        return mp.isfinite(mp.re(z)) and mp.isfinite(mp.im(z))
    except Exception:
        return False


def root_is_valid(root) -> bool:
    return is_finite_complex(root)


def normalized_frequency(omega, cfg: Config):
    return omega / cfg.f


def deduplicate_roots(roots, cfg: Config):
    roots = sorted(
        roots,
        key=lambda z: float(mp.re(z) / cfg.f),
    )

    unique = []

    for root in roots:
        if not unique:
            unique.append(root)
            continue

        if abs((root - unique[-1]) / cfg.f) > ROOT_DUPLICATE_TOL:
            unique.append(root)

    return unique


# ============================================================
# PHYSICAL EFFECTIVE RADIAL WAVENUMBER
#
#     k_r^2(r) = K^2 + Lambda/r - m^2/r^2
# ============================================================
def k_eff_squared(r, omega, m: int, cfg: Config):
    K2 = K_squared(omega, m, cfg)
    lam = Lambda_whittaker(omega, m, cfg)

    return (
        K2
        + lam / r
        - mp.mpf(m) ** 2 / r**2
    )


def effective_radial_wavenumber_at_r0(root, m: int, cfg: Config):
    try:
        kr2 = k_eff_squared(
            cfg.r0,
            root,
            m,
            cfg,
        )
    except Exception:
        return None

    if abs(mp.im(kr2)) > mp.mpf("1e-8"):
        return None

    kr2 = mp.re(kr2)

    if kr2 <= 0:
        return None

    return mp.sqrt(kr2)


@dataclass
class RootRecord:
    m: int
    omega: mp.mpc
    omega_norm: mp.mpc
    omega_real: mp.mpf
    omega_imag: mp.mpf
    sign: int
    n: int | None
    mode_family: str
    radial_behavior: str
    K2: mp.mpc
    Lambda: mp.mpc
    k_eff2_r1: mp.mpc | None
    k_eff2_r0: mp.mpc | None
    k_eff2_r2: mp.mpc | None
    radial_k_r0: mp.mpc | None
    radial_nodes: int | None
    residual: mp.mpf


# ============================================================
# FIND ROOTS AT m=1 ONLY
# ============================================================
# ============================================================
# NEW: POINCARE INITIAL GUESSES
# ============================================================
# ============================================================
# NEW: RADIAL BEHAVIOR AND NODE COUNTING
# ============================================================
def generate_poincare_seeds(m: int, cfg: Config):
    seeds = []
    L = cfg.r2 - cfg.r1
    for n in range(1, N_RADIAL_MODES + 1):
        kr_n = n * mp.pi / L

        # omega0^2 = f^2 + omega_A^2 + g H0 [ k_r,n^2 + m^2/r0^2 ]
        omega0_sq = cfg.f**2 + omega_A_squared(cfg) + cfg.g * cfg.H0 * (kr_n**2 + (m/cfg.r0)**2)
        omega0 = mp.sqrt(omega0_sq)

        for factor in (mp.mpf("0.9"), mp.mpf("1.0"), mp.mpf("1.1")):
            seeds.append(+omega0 * factor)
            seeds.append(-omega0 * factor)

    return seeds

# ============================================================
# NEW: RADIAL BEHAVIOR AND NODE COUNTING
# ============================================================
def classify_radial_behavior(omega, m: int, cfg: Config):
    try:
        r_vals = mp.linspace(cfg.r1, cfg.r2, 50)
        has_positive = False
        has_negative = False

        for r in r_vals:
            kr2 = k_eff_squared(r, omega, m, cfg)

            if abs(mp.im(kr2)) > 1e-8:
                return "complex"

            kr2_re = mp.re(kr2)
            if kr2_re > 0:
                has_positive = True
            elif kr2_re < 0:
                has_negative = True

        if has_positive and not has_negative:
            return "propagating"
        elif has_negative and not has_positive:
            return "evanescent"
        elif has_positive and has_negative:
            return "turning"
        else:
            return "complex"
    except Exception:
        return "complex"

def count_interior_nodes(omega, m: int, cfg: Config):
    try:
        # Construct eigenfunction: eta(r) = A*M(r) + B*W(r)
        # Using the wall matrix at r1:
        # M11*A + M12*B = 0 => A = M12, B = -M11
        M_mat = whittaker_boundary_matrix(omega, m, cfg)
        M11, M12 = M_mat[0]

        A = M12
        B = -M11

        # Sample on fine grid
        r_vals = mp.linspace(cfg.r1, cfg.r2, 100)
        eta_vals_complex = []
        for r in r_vals:
            M, W, _, _, _, _, _ = whittaker_basis(omega, m, cfg, r)
            eta = A*M + B*W
            eta_vals_complex.append(eta)

        # Remove an arbitrary complex phase before node counting
        max_val = max(eta_vals_complex, key=abs)
        if abs(max_val) == 0:
            return None
        phase = max_val / abs(max_val)

        eta_vals = [mp.re(v / phase) for v in eta_vals_complex]

        # Filter noise and count crossings (interior only)
        max_amp = max(abs(v) for v in eta_vals)
        if max_amp == 0:
            return None

        threshold = max_amp * mp.mpf('1e-5')

        # Remove boundary points
        eta_interior = eta_vals[1:-1]

        # Filter values
        filtered = [v for v in eta_interior if abs(v) > threshold]

        if not filtered:
            return 0

        nodes = 0
        current_sign = mp.sign(filtered[0])
        for v in filtered[1:]:
            s = mp.sign(v)
            if s != current_sign:
                nodes += 1
                current_sign = s

        return nodes
    except Exception:
        return None

def seed_roots_m1(cfg: Config):
    roots = []

    # 1. Low-frequency / Kelvin-like / user seeds
    seeds = [mp.mpf(str(g)) * cfg.f for g in INITIAL_GUESS_NORMALIZED if g != 0]

    # 2. Poincare seeds
    poincare = generate_poincare_seeds(1, cfg)
    seeds.extend(poincare)

    for omega_guess in seeds:
        try:
            root = direct_findroot(
                omega_guess,
                1,
                cfg,
            )
        except Exception:
            continue

        if not root_is_valid(root):
            continue

        try:
            residual = abs(
                dispersion_determinant(root, 1, cfg)
            )
        except Exception:
            continue

        if residual > ROOT_RESIDUAL_TOL:
            continue

        roots.append(root)

    return deduplicate_roots(roots, cfg)


# ============================================================
# MAKE INITIAL RADIAL BRANCHES FROM m=1
# n is ordered using the physical k_r(r0), not sqrt(K^2).
# ============================================================
def initialize_branches(cfg: Config):
    roots = seed_roots_m1(cfg)

    initial_tuples = []

    # We will deduplicate Poincare branches that get the same n and sign.
    # We want to keep the one with the smallest residual.
    poincare_candidates = {} # (n, sign) -> (root, residual)

    other_roots = []

    for root in roots:
        sign = 1 if mp.re(root) > 0 else -1

        behavior = classify_radial_behavior(root, 1, cfg)

        try:
            residual = abs(dispersion_determinant(root, 1, cfg))
        except Exception:
            continue

        n = None
        mode_family = "low-frequency"

        # A root is a Poincare branch candidate only when:
        # abs(Im(omega)/f) < REAL_ROOT_IMAG_TOL AND radial_behavior == "propagating" AND radial node counting succeeds.
        is_poincare = False
        if abs(mp.im(root)/cfg.f) < REAL_ROOT_IMAG_TOL and behavior == "propagating":
            nodes = count_interior_nodes(root, 1, cfg)
            if nodes is not None:
                n = nodes + 1
                mode_family = "Poincare"
                is_poincare = True

                key = (n, sign)
                if key not in poincare_candidates or residual < poincare_candidates[key][1]:
                    poincare_candidates[key] = (root, residual)

        if not is_poincare:
            if behavior == "evanescent":
                mode_family = "evanescent"
            elif behavior == "turning":
                mode_family = "turning"

            # For low frequency, if not selected as Poincare branch but abs(Re(omega)/f) <= 1
            if mode_family == "low-frequency" and abs(mp.re(root)/cfg.f) > 1.0:
                # Still keep them per requirement: "Do not discard them... Retain the root."
                mode_family = "other"
            other_roots.append((None, sign, root, mode_family))

    # Add back the best Poincare candidates
    for (n, sign), (root, res) in poincare_candidates.items():
        initial_tuples.append((n, sign, root, "Poincare"))

    initial_tuples.extend(other_roots)

    return initial_tuples


# ============================================================
# CONTINUE ONE DIRECT-ROOT BRANCH IN m
# ============================================================
def continue_branch(
    n,
    sign,
    root_m1,
    mode_family,
    cfg: Config,
):
    branch_records = []

    omega_previous = mp.mpc(root_m1)
    omega_previous_previous = None

    for m in range(1, 31):

        if m == 1:
            root = omega_previous
        else:
            try:
                if omega_previous_previous is None:
                    omega_guess = omega_previous
                else:
                    omega_guess = omega_previous + (omega_previous - omega_previous_previous)

                root = direct_findroot(
                    omega_guess,
                    m,
                    cfg,
                )
            except Exception:
                break

        if not root_is_valid(root):
            break

        # Preserve branch sign.
        if sign > 0 and mp.re(root) <= 0:
            break
        if sign < 0 and mp.re(root) >= 0:
            break

        try:
            residual = abs(dispersion_determinant(root, m, cfg))
            K2 = K_squared(root, m, cfg)
            lam = Lambda_whittaker(root, m, cfg)
            k2_r1 = k_eff_squared(cfg.r1, root, m, cfg)
            k2_r0 = k_eff_squared(cfg.r0, root, m, cfg)
            k2_r2 = k_eff_squared(cfg.r2, root, m, cfg)
        except Exception:
            break

        if residual > ROOT_RESIDUAL_TOL:
            break

        kr0 = None
        if abs(mp.im(k2_r0)) <= mp.mpf("1e-8") and mp.re(k2_r0) > 0:
            kr0 = mp.sqrt(mp.re(k2_r0))

        behavior = classify_radial_behavior(root, m, cfg)

        nodes = None
        if behavior == "propagating" and mode_family == "Poincare":
            nodes = count_interior_nodes(root, m, cfg)

        guess_str = mp.nstr(omega_guess / cfg.f, 6) if m > 1 else "N/A"
        print(f"m={m}, n={n}, sign={sign}, guess={guess_str}")

        branch_records.append(
            RootRecord(
                m=m,
                omega=root,
                omega_norm=normalized_frequency(root, cfg),
                omega_real=mp.re(root),
                omega_imag=mp.im(root),
                sign=sign,
                n=n,
                mode_family=mode_family,
                radial_behavior=behavior,
                K2=K2,
                Lambda=lam,
                k_eff2_r1=k2_r1,
                k_eff2_r0=k2_r0,
                k_eff2_r2=k2_r2,
                radial_k_r0=kr0,
                radial_nodes=nodes,
                residual=residual,
            )
        )

        omega_previous_previous = omega_previous
        omega_previous = root

    return branch_records


# ============================================================
# SOLVE ALL BRANCHES m = 1,...,30
# ============================================================
def solve_all_branches(cfg: Config):
    initial = initialize_branches(cfg)

    records = []

    print(f"Total m=1 roots found: {len(initial)}")
    print("\n{:<6} {:<25} {:<25} {:<20} {:<20} {:<20} {:<20} {:<15} {:<6} {:<15} {:<10}".format(
        "Index", "Re[omega]/(2Omega)", "Im[omega]/(2Omega)", "K^2", "k_r^2(r1)", "k_r^2(r0)", "k_r^2(r2)", "Behavior", "Nodes", "Family", "|D|"))
    print("-" * 180)

    for idx, (n, sign, root_m1, mode_family) in enumerate(initial):
        try:
            residual = abs(dispersion_determinant(root_m1, 1, cfg))
            K2 = K_squared(root_m1, 1, cfg)
            lam = Lambda_whittaker(root_m1, 1, cfg)
            k2_r1 = k_eff_squared(cfg.r1, root_m1, 1, cfg)
            k2_r0 = k_eff_squared(cfg.r0, root_m1, 1, cfg)
            k2_r2 = k_eff_squared(cfg.r2, root_m1, 1, cfg)
        except Exception:
            continue

        behavior = classify_radial_behavior(root_m1, 1, cfg)
        nodes = count_interior_nodes(root_m1, 1, cfg) if behavior == "propagating" and mode_family == "Poincare" else None

        nodes_str = str(nodes) if nodes is not None else "None"

        print("{:<6} {:<25.9e} {:<25.3e} {:<20.3e} {:<20.3e} {:<20.3e} {:<20.3e} {:<15} {:<6} {:<15} {:<10.3e}".format(
            idx, float(mp.re(root_m1/cfg.f)), float(mp.im(root_m1/cfg.f)), float(mp.re(K2)), float(mp.re(k2_r1)), float(mp.re(k2_r0)), float(mp.re(k2_r2)), behavior, nodes_str, mode_family, float(residual)))

        branch = continue_branch(
            n=n,
            sign=sign,
            root_m1=root_m1,
            mode_family=mode_family,
            cfg=cfg,
        )
        records.extend(branch)

    return records


# ============================================================
# PRINT
# ============================================================
def print_parameter_summary(cfg: Config):
    print("=" * 80)
    print("MATHEMATICA-STYLE DIRECT SWMHD DISPERSION SOLVER")
    print("=" * 80)
    print(f"Omega       = {mp.nstr(cfg.Omega, PRINT_DIGITS)}")
    print(f"2 Omega     = {mp.nstr(cfg.f, PRINT_DIGITS)}")
    print(f"H0          = {mp.nstr(cfg.H0, PRINT_DIGITS)}")
    print(f"g           = {mp.nstr(cfg.g, PRINT_DIGITS)}")
    print(f"B0          = {mp.nstr(cfg.B0, PRINT_DIGITS)}")
    print(f"rho0        = {mp.nstr(cfg.rho0, PRINT_DIGITS)}")
    print(f"r1          = {mp.nstr(cfg.r1, PRINT_DIGITS)}")
    print(f"r2          = {mp.nstr(cfg.r2, PRINT_DIGITS)}")
    print(f"r0          = {mp.nstr(cfg.r0, PRINT_DIGITS)}")
    print(f"C           = {mp.nstr(cfg.C, PRINT_DIGITS)}")
    print("m range     = 1 ... 30")
    # print(f"n retained  = 1 ... {N_RADIAL_MODES}")
    print("root search = direct FindRoot + continuation")
    print("radial k_r^2 = K^2 + Lambda/r - m^2/r^2")
    print("=" * 80)


def print_root_table(records, cfg: Config):
    print("\nFinal Summary")

    m1_count = len([r for r in records if r.m == 1])
    real_count = len([r for r in records if abs(r.omega_imag / cfg.f) < REAL_ROOT_IMAG_TOL])
    complex_count = len(records) - real_count
    prop_count = len([r for r in records if r.radial_behavior == "propagating"])
    evan_count = len([r for r in records if r.radial_behavior == "evanescent"])
    turn_count = len([r for r in records if r.radial_behavior == "turning"])
    poinc_branches = len(set(r.n for r in records if r.mode_family == "Poincare" and r.n is not None))
    max_poinc_n = max([r.n for r in records if r.mode_family == "Poincare" and r.n is not None] + [0])

    print(f"Number of m=1 roots found: {m1_count}")
    print(f"Number of real roots: {real_count}")
    print(f"Number of complex roots: {complex_count}")
    print(f"Number of propagating roots: {prop_count}")
    print(f"Number of evanescent roots: {evan_count}")
    print(f"Number of turning-point roots: {turn_count}")
    print(f"Number of Poincare branches: {poinc_branches}")
    print(f"Maximum Poincare n reached: {max_poinc_n}")



# ============================================================
# PLOT
# ============================================================
def plot_dispersion(records, cfg: Config):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
        constrained_layout=True,
    )

    colors = {
        +1: "red",
        -1: "blue",
    }

    # Filter for mostly real roots for the main plotting
    real_records = [r for r in records if abs(r.omega_imag / cfg.f) < REAL_ROOT_IMAG_TOL]

    # 1. Poincare propagating branches
    poincare_recs = [r for r in real_records if r.mode_family == "Poincare"]
    unique_n = sorted(list(set(r.n for r in poincare_recs if r.n is not None)))

    for n in unique_n:
        for sign in (-1, +1):
            pts = [r for r in poincare_recs if r.n == n and r.sign == sign]
            pts.sort(key=lambda r: r.m)

            if len(pts) < 1:
                continue

            m_vals = np.asarray([r.m for r in pts], dtype=float)
            w_vals = np.asarray([float(r.omega_real / cfg.f) for r in pts], dtype=float)

            label = rf"$n={n}$, " + (r"$\omega>0$" if sign > 0 else r"$\omega<0$")

            ax1.plot(
                m_vals, w_vals,
                color=colors[sign],
                linestyle="-",
                linewidth=1.8,
                marker=None, # requested to not have arbitrary space but not scatter either
                solid_capstyle="round",
                label=label,
            )

            ax2.plot(
                m_vals, w_vals,
                color=colors[sign],
                linestyle="-",
                linewidth=1.8,
                marker=None,
                solid_capstyle="round",
            )

    # 2. Slow branches / low-frequency (evanescent / other)
    # They should also be plotted via Re(omega)/(2Omega)
    # The requirement says "Plot retained low-frequency roots using Re(omega)/(2Omega) including roots that are real, weakly complex, evanescent, turning-point."
    slow_recs = [r for r in records if r.mode_family != "Poincare"]
    for sign in (-1, +1):
        # We can just scatter plot these slow branches because they don't have n branch tracking strictly
        # Oh actually we do have a branch sequence. Let's group them by the original m=1 branch.
        # But wait, they all have n=None. We can group by sign and plot dots, or we can find unique initial branches and track them.
        pts = [r for r in slow_recs if r.sign == sign]
        pts.sort(key=lambda r: r.m)
        if len(pts) > 0:
            m_vals = [r.m for r in pts]
            w_vals = [float(r.omega_real / cfg.f) for r in pts]

            ax2.scatter(
                m_vals, w_vals,
                color=colors[sign],
                marker="x",
                s=20,
                alpha=0.6,
                label="Slow / Evanescent" if sign == 1 else None
            )

    # ------------------------------------------------------------
    # Full diagram
    # ------------------------------------------------------------
    ax1.axhline(0.0, linewidth=0.8, color="black")
    ax1.set_xlim(1, 30)
    ax1.set_ylabel(r"$\omega/(2\Omega)$")
    ax1.set_title("Annular SWMHD dispersion diagram")
    ax1.grid(True, linewidth=0.4, alpha=0.30)

    # Handle legends carefully
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8, ncol=2)

    # ------------------------------------------------------------
    # Slow-branch zoom
    # ------------------------------------------------------------
    ax2.axhline(0.0, linewidth=0.8, color="black")
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(1, 30)
    ax2.set_xlabel(r"Azimuthal mode number $m$")
    ax2.set_ylabel(r"$\omega/(2\Omega)$")
    ax2.set_title(r"Slow branches: $-1 \leq \omega/(2\Omega) \leq 1$")
    ax2.set_xticks(np.arange(1, 31, 2))
    ax2.grid(True, linewidth=0.4, alpha=0.30)

    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)

    output = Path("SWMHD_dispersion_Mathematica_style.png")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"\nDispersion diagram saved to: {output.resolve()}")


# ============================================================
# MAIN
# ============================================================
def main():
    if CFG.C == 0:
        raise ValueError(
            "This code is for the C != 0 Whittaker system."
        )

    print_parameter_summary(CFG)

    records = solve_all_branches(CFG)

    print_root_table(records, CFG)

    plot_dispersion(records, CFG)

    return records


if __name__ == "__main__":
    main()
