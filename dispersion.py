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

N_RADIAL_MODES = 6
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


def root_is_real(root, cfg: Config) -> bool:
    return (
        is_finite_complex(root)
        and abs(mp.im(root) / cfg.f) <= REAL_ROOT_IMAG_TOL
    )


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
    sign: int
    radial_k: mp.mpf | None
    n: int | None
    K2: mp.mpc
    k_eff2_r0: mp.mpc | None
    Lambda: mp.mpc
    residual: mp.mpf


# ============================================================
# FIND ROOTS AT m=1 ONLY
# ============================================================
def seed_roots_m1(cfg: Config):
    roots = []

    for guess_norm in INITIAL_GUESS_NORMALIZED:
        omega_guess = mp.mpf(str(guess_norm)) * cfg.f

        if omega_guess == 0:
            continue

        try:
            root = direct_findroot(
                omega_guess,
                1,
                cfg,
            )
        except Exception:
            continue

        if not root_is_real(root, cfg):
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

    positive = []
    negative = []

    for root in roots:
        kr = effective_radial_wavenumber_at_r0(
            root,
            1,
            cfg,
        )

        if kr is None:
            continue

        if mp.re(root) > 0:
            positive.append((kr, root))
        elif mp.re(root) < 0:
            negative.append((kr, root))

    positive.sort(key=lambda q: float(q[0]))
    negative.sort(key=lambda q: float(q[0]))

    positive = positive[:N_RADIAL_MODES]
    negative = negative[:N_RADIAL_MODES]

    initial = []

    for n, (_, root) in enumerate(positive, start=1):
        initial.append((n, +1, root))

    for n, (_, root) in enumerate(negative, start=1):
        initial.append((n, -1, root))

    return initial


# ============================================================
# CONTINUE ONE DIRECT-ROOT BRANCH IN m
# ============================================================
def continue_branch(
    n: int,
    sign: int,
    root_m1,
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

        if not root_is_real(root, cfg):
            break

        # Preserve branch sign.
        if sign > 0 and mp.re(root) <= 0:
            break
        if sign < 0 and mp.re(root) >= 0:
            break

        try:
            residual = abs(
                dispersion_determinant(root, m, cfg)
            )
            K2 = K_squared(root, m, cfg)
            lam = Lambda_whittaker(root, m, cfg)
            kr2_r0 = k_eff_squared(
                cfg.r0,
                root,
                m,
                cfg,
            )
        except Exception:
            break

        if residual > ROOT_RESIDUAL_TOL:
            break

        kr0 = None
        if abs(mp.im(kr2_r0)) <= mp.mpf("1e-8") and mp.re(kr2_r0) > 0:
            kr0 = mp.sqrt(mp.re(kr2_r0))

        branch_records.append(
            RootRecord(
                m=m,
                omega=root,
                omega_norm=normalized_frequency(root, cfg),
                sign=sign,
                radial_k=kr0,
                n=n,
                K2=K2,
                k_eff2_r0=kr2_r0,
                Lambda=lam,
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

    for n, sign, root_m1 in initial:
        branch = continue_branch(
            n=n,
            sign=sign,
            root_m1=root_m1,
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
    print(f"n retained  = 1 ... {N_RADIAL_MODES}")
    print("root search = direct FindRoot + continuation")
    print("radial k_r^2 = K^2 + Lambda/r - m^2/r^2")
    print("=" * 80)


def print_root_table(records, cfg: Config):
    print("\nAccepted roots")
    print(
        "m   sign   n   Re[omega]/(2Omega)   Im[omega]/(2Omega)   "
        "k_r(r0) [m^-1]       K^2 [m^-2]          Lambda        |D|"
    )
    print("-" * 125)

    for rec in sorted(
        records,
        key=lambda q: (
            q.m,
            q.sign,
            q.n if q.n is not None else 999,
        ),
    ):
        kr = (
            mp.nstr(rec.radial_k, 8)
            if rec.radial_k is not None
            else "--"
        )

        print(
            f"{rec.m:2d}  "
            f"{rec.sign:+d}    "
            f"{rec.n:2d}   "
            f"{float(mp.re(rec.omega_norm)): .9e}   "
            f"{float(mp.im(rec.omega_norm)): .3e}   "
            f"{kr:>15s}   "
            f"{float(mp.re(rec.K2)): .6e}   "
            f"{float(mp.re(rec.Lambda)): .6e}   "
            f"{float(rec.residual):.3e}"
        )


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

    for n in range(1, N_RADIAL_MODES + 1):
        for sign in (-1, +1):

            pts = [
                rec
                for rec in records
                if rec.n == n
                and rec.sign == sign
                and root_is_real(rec.omega, cfg)
            ]

            pts.sort(key=lambda rec: rec.m)

            if len(pts) < 2:
                continue

            m_vals = np.asarray(
                [rec.m for rec in pts],
                dtype=float,
            )

            w_vals = np.asarray(
                [
                    float(mp.re(rec.omega_norm))
                    for rec in pts
                ],
                dtype=float,
            )

            label = (
                rf"$n={n}$, "
                + (r"$\omega>0$" if sign > 0 else r"$\omega<0$")
            )

            ax1.plot(
                m_vals,
                w_vals,
                color=colors[sign],
                linestyle="-",
                linewidth=1.8,
                marker=None,
                solid_capstyle="round",
                label=label,
            )

            ax2.plot(
                m_vals,
                w_vals,
                color=colors[sign],
                linestyle="-",
                linewidth=1.8,
                marker=None,
                solid_capstyle="round",
            )

    # ------------------------------------------------------------
    # Full diagram
    # ------------------------------------------------------------
    ax1.axhline(
        0.0,
        linewidth=0.8,
        color="black",
    )

    ax1.set_xlim(1, 30)
    ax1.set_ylabel(r"$\omega/(2\Omega)$")
    ax1.set_title("Annular SWMHD dispersion diagram")
    ax1.grid(True, linewidth=0.4, alpha=0.30)
    ax1.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    # ------------------------------------------------------------
    # Slow-branch zoom
    # ------------------------------------------------------------
    ax2.axhline(
        0.0,
        linewidth=0.8,
        color="black",
    )

    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(1, 30)
    ax2.set_xlabel(r"Azimuthal mode number $m$")
    ax2.set_ylabel(r"$\omega/(2\Omega)$")
    ax2.set_title(
        r"Slow branches: $-1 \leq \omega/(2\Omega) \leq 1$"
    )
    ax2.set_xticks(np.arange(1, 31, 1))
    ax2.grid(True, linewidth=0.4, alpha=0.30)

    output = Path(
        "SWMHD_dispersion_Mathematica_style.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

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
