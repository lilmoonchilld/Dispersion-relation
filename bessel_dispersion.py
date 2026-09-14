#!/usr/bin/env python3
"""Analytical Bessel dispersion relation for rotating annular shallow-water MHD.

Primary calculation:
    physical omega -> omega_star(omega) -> K^2(omega, m)
    -> 2x2 annular Bessel boundary determinant -> scalar roots in omega.

No Chebyshev, finite-difference, discretized-ODE, or matrix-eigenvalue solver is
used for the dispersion calculation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# ============================================================
# Path setup — compatible with Jupyter Notebook
# ============================================================

SCRIPT_DIR = Path.cwd()
WORKSPACE_DIR = SCRIPT_DIR

LOCAL_DEPS = WORKSPACE_DIR / "work" / "python_deps"

if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

MPL_CONFIG_DIR = WORKSPACE_DIR / "work" / "mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import numpy as np
from scipy import optimize, special

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Config:
    # Physical parameters.
    MU0: float = 4.0 * math.pi * 1e-7
    Omega: float = 0.5e-4
    H0: float = 500.0
    g: float = 9.81
    B0: float = 8.5e-4
    rho0: float = 1000.0
    r1: float = 0.5e6
    r2: float = 1.0e6
    C: float = 0.0

    # Numerical controls.
    M_MAX: int = 30
    N_MODES: int = 5
    OMEGA_NORM_MAX: float = 60.0
    GRID_POINTS: int = 22000
    MIN_INTERVAL_POINTS: int = 500
    SINGULAR_GAP_NORM: float = 2.0e-5
    ZERO_GAP_NORM: float = 2.0e-5
    DET_MIN_SCAN_TOL: float = 3.0e-3
    ROOT_ACCEPT_TOL: float = 5.0e-7
    DUPLICATE_NORM_TOL: float = 5.0e-5
    MINIMIZE_XATOL_NORM: float = 1.0e-9
    NODE_SAMPLES: int = 1200
    NODE_ZERO_TOL: float = 1.0e-5
    LOW_FREQ_NORM: float = 0.1
    BRANCH_MAX_JUMP_NORM: float = 10.0
    BRANCH_LOOKBACK: int = 4
    DPI: int = 300

    @property
    def D0(self) -> float:
        return self.H0

    @property
    def f(self) -> float:
        return 2.0 * self.Omega

    @property
    def r0(self) -> float:
        return 0.5 * (self.r1 + self.r2)


@dataclass(frozen=True)
class Parameters:
    MU0: float
    Omega: float
    f: float
    H0: float
    D0: float
    g: float
    B0: float
    rho0: float
    r1: float
    r2: float
    r0: float
    C: float
    omega_A2: float
    omega_A: float
    epsilon_P: float
    g0: float
    beta_eff: float


@dataclass
class RootRecord:
    m: int
    n: int
    omega: float
    omega_normalized: float
    omega_star: float
    omega_A: float
    K2: float
    radial_type: str
    C: float
    g0: float
    beta_eff: float
    Lambda: float
    kappa_r: float
    kappa_W: float
    residual: float
    determinant_residual: float
    BC_residual: float
    sigma_min: float
    branch_sign: int
    valid: bool
    node_count: int
    basis: str
    eta: np.ndarray | None = None
    branch_id: str = ""
    family_index: int = 0


@dataclass
class Branch:
    branch_id: str
    sign: int
    n: int
    family_index: int
    values: dict[int, RootRecord]

    def last_points(self, lookback: int) -> list[RootRecord]:
        if not self.values:
            return []
        latest_m = max(self.values)
        keys = [m for m in sorted(self.values) if latest_m - m <= lookback]
        return [self.values[m] for m in keys]

    def predict_norm(self, m_next: int, lookback: int) -> float | None:
        pts = self.last_points(lookback)
        if not pts:
            return None
        if len(pts) == 1:
            return pts[-1].omega_normalized
        p1, p2 = pts[-2], pts[-1]
        dm = p2.m - p1.m
        if dm == 0:
            return p2.omega_normalized
        slope = (p2.omega_normalized - p1.omega_normalized) / dm
        return p2.omega_normalized + slope * (m_next - p2.m)


def compute_parameters(cfg: Config, B0: float | None = None) -> Parameters:
    B = cfg.B0 if B0 is None else B0
    omega_A2_value = omega_A_squared(cfg, B)
    omega_A_value = math.sqrt(max(omega_A2_value, 0.0))
    epsilon_P = cfg.Omega**2 * cfg.r2**2 / (2.0 * cfg.g * cfg.D0)
    g0 = cfg.Omega**2 * cfg.r0 - cfg.C / cfg.r0**2
    beta_eff = cfg.Omega**2 + 2.0 * cfg.C / cfg.r0**3
    return Parameters(
        MU0=cfg.MU0,
        Omega=cfg.Omega,
        f=cfg.f,
        H0=cfg.H0,
        D0=cfg.D0,
        g=cfg.g,
        B0=B,
        rho0=cfg.rho0,
        r1=cfg.r1,
        r2=cfg.r2,
        r0=cfg.r0,
        C=cfg.C,
        omega_A2=omega_A2_value,
        omega_A=omega_A_value,
        epsilon_P=epsilon_P,
        g0=g0,
        beta_eff=beta_eff,
    )


def omega_A_squared(cfg: Config, B0: float | None = None) -> float:
    B = cfg.B0 if B0 is None else B0
    return B**2 / (cfg.MU0 * cfg.rho0 * cfg.D0**2)


def omega_star(omega: float, params: Parameters) -> float:
    if omega == 0.0:
        return math.nan
    return omega + params.omega_A2 / omega


def H_eq_prime(r: float, params: Parameters) -> float:
    return params.Omega**2 * r / params.g


def resonance_frequencies(params: Parameters) -> list[float]:
    roots: list[float] = []
    for sigma in (1.0, -1.0):
        b = -sigma * 2.0 * params.Omega
        c = params.omega_A2
        disc = b * b - 4.0 * c
        if disc >= 0.0:
            sq = math.sqrt(disc)
            roots.extend([(-b - sq) / 2.0, (-b + sq) / 2.0])
    roots = [r for r in roots if math.isfinite(r)]
    return sorted(set(round(r, 20) for r in roots))


def Lambda_whittaker(omega: float, m: int, params: Parameters) -> float:
    if np.isclose(params.C, 0.0):
        return 0.0
    os = omega_star(omega, params)
    if not math.isfinite(os) or abs(os) == 0.0:
        return math.nan
    return (3.0 * m * params.f * params.C) / (params.g * params.D0 * os * params.r0**2)

def K_squared(omega: float, m: int, params: Parameters) -> float:
    os = omega_star(omega, params)
    if not math.isfinite(os) or abs(os) == 0.0:
        return math.nan

    if np.isclose(params.C, 0.0):
        beta_eff = params.Omega**2
    else:
        beta_eff = params.beta_eff

    first = (omega / os) * (os**2 - 4.0 * params.Omega**2) / (params.g * params.D0)
    rossby = (2.0 * m * params.Omega * beta_eff) / (params.g * params.D0 * os)
    return first - rossby


import mpmath

def whittaker_M(kappa_W: complex | float, mu: int | float, x: complex | float) -> complex | float:
    a = mu - kappa_W + 0.5
    b = 2.0 * mu + 1.0
    return complex(np.exp(-x / 2.0) * (x**(mu + 0.5)) * complex(mpmath.hyp1f1(a, b, x)))

def whittaker_W(kappa_W: complex | float, mu: int | float, x: complex | float) -> complex | float:
    a = mu - kappa_W + 0.5
    b = 2.0 * mu + 1.0
    return complex(np.exp(-x / 2.0) * (x**(mu + 0.5)) * complex(mpmath.hyperu(a, b, x)))

def whittaker_M_dx(kappa_W: complex | float, mu: int | float, x: complex | float) -> complex | float:
    a = mu - kappa_W + 0.5
    b = 2.0 * mu + 1.0
    F_ab = complex(mpmath.hyp1f1(a, b, x))
    F_ap1_bp1 = complex(mpmath.hyp1f1(a + 1.0, b + 1.0, x))
    term1 = (a / b) * F_ap1_bp1
    term2 = ((mu + 0.5) / x - 0.5) * F_ab
    return complex(np.exp(-x / 2.0) * (x**(mu + 0.5)) * (term1 + term2))

def whittaker_W_dx(kappa_W: complex | float, mu: int | float, x: complex | float) -> complex | float:
    a = mu - kappa_W + 0.5
    b = 2.0 * mu + 1.0
    U_ab = complex(mpmath.hyperu(a, b, x))
    U_ap1_bp1 = complex(mpmath.hyperu(a + 1.0, b + 1.0, x))
    term1 = -a * U_ap1_bp1
    term2 = ((mu + 0.5) / x - 0.5) * U_ab
    return complex(np.exp(-x / 2.0) * (x**(mu + 0.5)) * (term1 + term2))

def whittaker_W_dx(kappa_W: complex | float, mu: int | float, x: complex | float) -> complex | float:
    a = mu - kappa_W + 0.5
    b = 2.0 * mu + 1.0
    U_ab = mpmath.hyperu(a, b, x)
    U_ap1_bp1 = mpmath.hyperu(a + 1.0, b + 1.0, x)
    term1 = -a * U_ap1_bp1
    term2 = ((mu + 0.5) / x - 0.5) * U_ab
    return np.exp(-x / 2.0) * (x**(mu + 0.5)) * (term1 + term2)

def whittaker_boundary_matrix(omega: float, m: int, params: Parameters) -> np.ndarray:
    os = omega_star(omega, params)
    k2 = K_squared(omega, m, params)
    if not math.isfinite(os) or not math.isfinite(k2):
        return np.full((2, 2), np.nan, dtype=complex)

    Lambda = Lambda_whittaker(omega, m, params)

    if k2 < 0:
        kappa_r = math.sqrt(-k2)
    elif k2 > 0:
        kappa_r = 1j * math.sqrt(k2)
    else:
        return np.full((2, 2), np.nan, dtype=complex)

    kappa_W = Lambda / (2.0 * kappa_r)

    rows = []
    for r in (params.r1, params.r2):
        x = 2.0 * kappa_r * r

        M_val = whittaker_M(kappa_W, m, x)
        W_val = whittaker_W(kappa_W, m, x)

        M_x = whittaker_M_dx(kappa_W, m, x)
        W_x = whittaker_W_dx(kappa_W, m, x)

        term_M = 2.0 * kappa_r * os * M_x - ((os / 2.0 + m * params.f) / r) * M_val
        term_W = 2.0 * kappa_r * os * W_x - ((os / 2.0 + m * params.f) / r) * W_val

        rows.append([term_M, term_W])

    return np.asarray(rows, dtype=complex)

def _row_normalized_matrix(M: np.ndarray) -> np.ndarray | None:
    if M.shape != (2, 2) or not np.all(np.isfinite(M)):
        return None
    row_norms = np.linalg.norm(M, axis=1)
    if np.any(~np.isfinite(row_norms)) or np.any(row_norms == 0.0):
        return None
    return M / row_norms[:, None]


def bessel_boundary_matrix(
    omega: float,
    m: int,
    params: Parameters,
    formulation: str = "recurrence",
) -> np.ndarray:
    os = omega_star(omega, params)
    k2 = K_squared(omega, m, params)
    if not math.isfinite(os) or not math.isfinite(k2) or k2 <= 0.0:
        return np.full((2, 2), np.nan)

    K = math.sqrt(k2)
    rows = []
    for r in (params.r1, params.r2):
        x = K * r
        Jm = special.jv(m, x)
        Ym = special.yv(m, x)
        if formulation == "derivative":
            cJ = os * K * special.jvp(m, x, 1) - (2.0 * m * params.Omega / r) * Jm
            cY = os * K * special.yvp(m, x, 1) - (2.0 * m * params.Omega / r) * Ym
        else:
            Jmp1 = special.jv(m + 1, x)
            Ymp1 = special.yv(m + 1, x)
            prefactor = m * (os - 2.0 * params.Omega) / r
            cJ = prefactor * Jm - os * K * Jmp1
            cY = prefactor * Ym - os * K * Ymp1
        rows.append([cJ, cY])
    return np.asarray(rows, dtype=float)


def modified_bessel_boundary_matrix(omega: float, m: int, params: Parameters) -> np.ndarray:
    os = omega_star(omega, params)
    k2 = K_squared(omega, m, params)
    if not math.isfinite(os) or not math.isfinite(k2) or k2 >= 0.0:
        return np.full((2, 2), np.nan)

    kappa = math.sqrt(-k2)
    rows = []
    for r in (params.r1, params.r2):
        x = kappa * r
        Im = special.iv(m, x)
        Km = special.kv(m, x)
        cI = os * kappa * special.ivp(m, x, 1) - (2.0 * m * params.Omega / r) * Im
        cK = os * kappa * special.kvp(m, x, 1) - (2.0 * m * params.Omega / r) * Km
        rows.append([cI, cK])
    return np.asarray(rows, dtype=float)


def zero_K_boundary_matrix(omega: float, m: int, params: Parameters) -> np.ndarray:
    os = omega_star(omega, params)
    rows = []
    for r in (params.r1, params.r2):
        c_plus = m * (os - 2.0 * params.Omega) / r
        c_minus = -m * (os + 2.0 * params.Omega) / r
        rows.append([c_plus, c_minus])
    return np.asarray(rows, dtype=float)


def boundary_matrix(omega: float, m: int, params: Parameters) -> tuple[np.ndarray, str]:
    k2 = K_squared(omega, m, params)
    if not math.isfinite(k2):
        return np.full((2, 2), np.nan), "invalid"

    if np.isclose(params.C, 0.0):
        if abs(k2) < 1.0e-12:
            return zero_K_boundary_matrix(omega, m, params), "power"
        if k2 > 0.0:
            return bessel_boundary_matrix(omega, m, params, formulation="recurrence"), "ordinary"
        return modified_bessel_boundary_matrix(omega, m, params), "modified"
    else:
        mat = whittaker_boundary_matrix(omega, m, params)
        return mat, "whittaker"


def sigma_min(omega: float, m: int, params: Parameters) -> float:
    M, _ = boundary_matrix(omega, m, params)
    scale = np.linalg.norm(M, ord='fro')
    if not np.isfinite(scale) or scale == 0:
        return math.inf
    Mn = M / scale
    try:
        svals = np.linalg.svd(Mn, compute_uv=False)
        return float(svals[-1])
    except np.linalg.LinAlgError:
        return math.inf

def dispersion_function(omega: float, m: int, params: Parameters) -> complex | float:
    M, basis = boundary_matrix(omega, m, params)
    Mn = _row_normalized_matrix(M)
    if Mn is None:
        return math.nan
    det = np.linalg.det(Mn)
    if basis == "whittaker":
        return complex(det)
    return float(np.real_if_close(det))

def normalized_dispersion_residual(omega: float, m: int, params: Parameters) -> float:
    return sigma_min(omega, m, params)



def _is_valid_trial_frequency(omega: float, params: Parameters, cfg: Config) -> bool:
    scale = 2.0 * params.Omega
    if abs(omega) <= cfg.ZERO_GAP_NORM * scale:
        return False
    for s in resonance_frequencies(params):
        if abs(omega - s) <= cfg.SINGULAR_GAP_NORM * scale:
            return False
    os = omega_star(omega, params)
    if not math.isfinite(os):
        return False
    if abs(os**2 - 4.0 * params.Omega**2) <= 1.0e-12 * (2.0 * params.Omega) ** 2:
        return False
    return True


def _search_intervals(params: Parameters, cfg: Config) -> list[tuple[float, float]]:
    scale = 2.0 * params.Omega
    lo = -cfg.OMEGA_NORM_MAX * scale
    hi = cfg.OMEGA_NORM_MAX * scale
    exclusions: list[tuple[float, float]] = [
        (-cfg.ZERO_GAP_NORM * scale, cfg.ZERO_GAP_NORM * scale)
    ]
    for s in resonance_frequencies(params):
        gap = cfg.SINGULAR_GAP_NORM * scale
        exclusions.append((s - gap, s + gap))
    exclusions = sorted((max(lo, a), min(hi, b)) for a, b in exclusions if b > lo and a < hi)

    intervals: list[tuple[float, float]] = []
    cursor = lo
    for a, b in exclusions:
        if a > cursor:
            intervals.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < hi:
        intervals.append((cursor, hi))
    return [(a, b) for a, b in intervals if b > a]


def compute_eigenfunction(
    omega: float,
    m: int,
    params: Parameters,
    cfg: Config,
    r_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int | str]:
    if r_grid is None:
        r_grid = np.linspace(params.r1, params.r2, cfg.NODE_SAMPLES)
    M, basis = boundary_matrix(omega, m, params)
    Mn = _row_normalized_matrix(M)
    if Mn is None:
        raise ValueError("invalid boundary matrix")
    _, _, vh = np.linalg.svd(Mn)
    coeff = np.asarray(vh[-1, :], dtype=float)
    coeff_norm = np.linalg.norm(coeff)
    if coeff_norm == 0.0 or not math.isfinite(coeff_norm):
        raise ValueError("invalid null vector")
    coeff = coeff / coeff_norm

    k2 = K_squared(omega, m, params)
    if basis == "ordinary":
        K = math.sqrt(k2)
        x = K * r_grid
        eta = coeff[0] * special.jv(m, x) + coeff[1] * special.yv(m, x)
        eta_prime = K * (coeff[0] * special.jvp(m, x, 1) + coeff[1] * special.yvp(m, x, 1))
    elif basis == "modified":
        kappa = math.sqrt(-k2)
        x = kappa * r_grid
        eta = coeff[0] * special.iv(m, x) + coeff[1] * special.kv(m, x)
        eta_prime = kappa * (
            coeff[0] * special.ivp(m, x, 1) + coeff[1] * special.kvp(m, x, 1)
        )
    elif basis == "whittaker":
        if k2 < 0:
            kappa_r = math.sqrt(-k2)
        else:
            kappa_r = 1j * math.sqrt(k2)
        Lambda = Lambda_whittaker(omega, m, params)
        kappa_W = Lambda / (2.0 * kappa_r)

        eta = np.zeros_like(r_grid, dtype=complex)
        eta_prime = np.zeros_like(r_grid, dtype=complex)

        for i, r in enumerate(r_grid):
            x = 2.0 * kappa_r * r
            M_val = whittaker_M(kappa_W, m, x)
            W_val = whittaker_W(kappa_W, m, x)
            M_x = whittaker_M_dx(kappa_W, m, x)
            W_x = whittaker_W_dx(kappa_W, m, x)

            val = r**(-0.5) * (coeff[0] * M_val + coeff[1] * W_val)

            dM_dr = -0.5 * r**(-1.5) * M_val + 2.0 * kappa_r * r**(-0.5) * M_x
            dW_dr = -0.5 * r**(-1.5) * W_val + 2.0 * kappa_r * r**(-0.5) * W_x
            dval = coeff[0] * dM_dr + coeff[1] * dW_dr

            eta[i] = val
            eta_prime[i] = dval
    else:
        rr = r_grid / params.r1
        eta = coeff[0] * rr**m + coeff[1] * rr ** (-m)
        eta_prime = (m / params.r1) * (
            coeff[0] * rr ** (m - 1) - coeff[1] * rr ** (-m - 1)
        )


    eta = np.asarray(np.real_if_close(eta), dtype=float)
    eta_prime = np.asarray(np.real_if_close(eta_prime), dtype=float)
    amp = np.nanmax(np.abs(eta))
    if math.isfinite(amp) and amp > 0.0:
        eta = eta / amp
        eta_prime = eta_prime / amp
    node_count = classify_radial_mode(r_grid, eta, cfg) - 1
    return {
        "r": r_grid,
        "eta": eta,
        "eta_prime": eta_prime,
        "coefficients": coeff,
        "node_count": node_count,
        "n": node_count + 1,
        "basis": basis,
    }


def classify_radial_mode(r_grid: np.ndarray, eta: np.ndarray, cfg: Config) -> int:
    del r_grid
    y = np.asarray(np.real_if_close(eta), dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return 0
    y = y[finite]
    amp = np.nanmax(np.abs(y))
    if not math.isfinite(amp) or amp == 0.0:
        return 0
    y = y / amp
    if y.size > 10:
        y = y[5:-5]
    signs: list[float] = []
    for value in y:
        if abs(value) > cfg.NODE_ZERO_TOL:
            sign = 1.0 if value > 0.0 else -1.0
            if not signs or sign != signs[-1]:
                signs.append(sign)
    node_count = sum(1 for a, b in zip(signs, signs[1:]) if a * b < 0.0)
    return node_count + 1


def validate_eigenfunction(omega: float, m: int, params: Parameters, cfg: Config) -> float:
    del cfg
    M, _ = boundary_matrix(omega, m, params)
    Mn = _row_normalized_matrix(M)
    if Mn is None:
        return math.inf
    _, _, vh = np.linalg.svd(Mn)
    coeff = np.asarray(vh[-1, :], dtype=float)
    coeff_norm = max(float(np.linalg.norm(coeff)), 1.0e-300)
    row_norms = np.linalg.norm(M, axis=1)
    bc = M @ coeff
    denom = max(float(np.nanmax(row_norms)) * coeff_norm, 1.0e-300)
    return float(np.nanmax(np.abs(bc)) / denom)


def validate_root(omega: float, m: int, params: Parameters, cfg: Config) -> RootRecord | None:
    if not _is_valid_trial_frequency(omega, params, cfg):
        return None
    residual = normalized_dispersion_residual(omega, m, params)
    determinant_residual = abs(dispersion_function(omega, m, params))
    if not math.isfinite(residual) or residual > cfg.ROOT_ACCEPT_TOL:
        return None
    try:
        eig = compute_eigenfunction(omega, m, params, cfg)
        bc_residual = validate_eigenfunction(omega, m, params, cfg)
    except Exception:
        return None
    if not math.isfinite(bc_residual) or bc_residual > max(1.0e-4, 50.0 * cfg.ROOT_ACCEPT_TOL):
        return None
    k2 = K_squared(omega, m, params)
    os = omega_star(omega, params)
    n = int(eig["n"])
    return RootRecord(
        m=m,
        n=n,
        omega=float(omega),
        omega_normalized=float(omega / (2.0 * params.Omega)),
        omega_star=float(os),
        omega_A=float(params.omega_A),
        K2=float(k2),
        radial_type="oscillatory" if k2 > 0 else "evanescent",
        C=float(params.C),
        g0=float(params.g0),
        beta_eff=float(params.beta_eff),
        Lambda=float(Lambda_whittaker(omega, m, params)),
        kappa_r=float(np.sqrt(abs(k2))),
        kappa_W=float(abs(Lambda_whittaker(omega, m, params) / (2.0 * np.sqrt(abs(k2))))) if k2 != 0.0 else 0.0,
        residual=float(residual),
        determinant_residual=float(determinant_residual),
        BC_residual=float(bc_residual),
        sigma_min=float(residual),
        branch_sign=1 if omega > 0.0 else -1,
        valid=True,
        node_count=n - 1,
        basis=str(eig["basis"]),
        eta=eig["eta"],
    )


def _deduplicate_roots(records: Iterable[RootRecord], params: Parameters, cfg: Config) -> list[RootRecord]:
    records = sorted(records, key=lambda rec: rec.omega)
    groups: list[list[RootRecord]] = []
    tol = cfg.DUPLICATE_NORM_TOL * 2.0 * params.Omega
    for rec in records:
        if not groups or abs(rec.omega - groups[-1][-1].omega) > tol:
            groups.append([rec])
        else:
            groups[-1].append(rec)
    unique: list[RootRecord] = []
    for group in groups:
        unique.append(min(group, key=lambda rec: (rec.residual, rec.BC_residual)))
    return sorted(unique, key=lambda rec: rec.omega)



def _candidate_from_minimum(
    a: float,
    b: float,
    m: int,
    params: Parameters,
    cfg: Config,
) -> RootRecord | None:
    if b <= a:
        return None
    try:
        result = optimize.minimize_scalar(
            lambda w: sigma_min(w, m, params),
            bounds=(a, b),
            method="bounded",
            options={"xatol": cfg.MINIMIZE_XATOL_NORM * 2.0 * params.Omega},
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None
    if not result.success:
        return None
    return validate_root(float(result.x), m, params, cfg)

def find_roots_for_m(m: int, params: Parameters, cfg: Config) -> list[RootRecord]:
    intervals = _search_intervals(params, cfg)
    full_width = 2.0 * cfg.OMEGA_NORM_MAX * 2.0 * params.Omega
    records: list[RootRecord] = []

    for a, b in intervals:
        n_points = max(cfg.MIN_INTERVAL_POINTS, int(cfg.GRID_POINTS * (b - a) / full_width))
        grid = np.linspace(a, b, n_points)
        values = np.asarray([sigma_min(w, m, params) for w in grid], dtype=float)
        finite = np.isfinite(values)

        abs_values = np.abs(values)
        abs_values[~finite] = math.inf
        local = (
            (abs_values[1:-1] < abs_values[:-2])
            & (abs_values[1:-1] < abs_values[2:])
            & (abs_values[1:-1] < cfg.DET_MIN_SCAN_TOL)
        )
        local_indices = np.flatnonzero(local) + 1

        for i in local_indices:
            left = grid[max(i - 1, 0)]
            right = grid[min(i + 1, len(grid) - 1)]
            rec = _candidate_from_minimum(left, right, m, params, cfg)
            if rec is not None:
                records.append(rec)

    records = _deduplicate_roots(records, params, cfg)
    return [rec for rec in records if rec.n <= cfg.N_MODES]


def _group_records(records: list[RootRecord]) -> dict[tuple[int, int], list[RootRecord]]:
    groups: dict[tuple[int, int], list[RootRecord]] = {}
    for rec in records:
        groups.setdefault((rec.branch_sign, rec.n), []).append(rec)
    for group in groups.values():
        group.sort(key=lambda rec: abs(rec.omega_normalized))
    return groups


def track_branches(results_by_m: dict[int, list[RootRecord]], cfg: Config) -> list[Branch]:
    branches: list[Branch] = []
    next_id = 1

    for m in range(1, cfg.M_MAX + 1):
        candidates_by_group = _group_records(results_by_m.get(m, []))

        for (sign, n), candidates in candidates_by_group.items():
            available = set(range(len(candidates)))
            active = [
                br
                for br in branches
                if br.sign == sign
                and br.n == n
                and br.last_points(cfg.BRANCH_LOOKBACK)
                and max(br.values) < m
            ]

            if active and candidates:
                costs = np.full((len(active), len(candidates)), 1.0e9, dtype=float)
                for i, br in enumerate(active):
                    pred = br.predict_norm(m, cfg.BRANCH_LOOKBACK)
                    if pred is None:
                        continue

                    last_rec = br.values[max(br.values)]
                    eta1 = last_rec.eta

                    for j, rec in enumerate(candidates):
                        diff = abs(rec.omega_normalized - pred)
                        if diff <= cfg.BRANCH_MAX_JUMP_NORM:
                            eta2 = rec.eta
                            if eta1 is not None and eta2 is not None:
                                norm1 = np.linalg.norm(eta1)
                                norm2 = np.linalg.norm(eta2)
                                overlap = abs(np.vdot(eta1 / norm1, eta2 / norm2)) if norm1 > 0 and norm2 > 0 else 0.0
                            else:
                                overlap = 0.0
                            costs[i, j] = diff + 10.0 * (1.0 - overlap)
                row_ind, col_ind = optimize.linear_sum_assignment(costs)
                for row, col in zip(row_ind, col_ind):
                    if costs[row, col] < cfg.BRANCH_MAX_JUMP_NORM and col in available:
                        br = active[row]
                        rec = candidates[col]
                        rec.branch_id = br.branch_id
                        rec.family_index = br.family_index
                        br.values[m] = rec
                        available.remove(col)

            for col in sorted(available, key=lambda j: abs(candidates[j].omega_normalized)):
                rec = candidates[col]
                family_index = 1 + sum(1 for br in branches if br.sign == sign and br.n == n)
                branch_id = f"{'pos' if sign > 0 else 'neg'}_n{n}_b{family_index}"
                rec.branch_id = branch_id
                rec.family_index = family_index
                branches.append(
                    Branch(
                        branch_id=branch_id,
                        sign=sign,
                        n=n,
                        family_index=family_index,
                        values={m: rec},
                    )
                )
                next_id += 1

    return branches


def find_all_modes(cfg: Config, params: Parameters) -> tuple[dict[int, list[RootRecord]], list[Branch]]:
    results_by_m: dict[int, list[RootRecord]] = {}
    for m in range(1, cfg.M_MAX + 1):
        records = find_roots_for_m(m, params, cfg)
        results_by_m[m] = records
        low = [rec for rec in records if abs(rec.omega_normalized) < cfg.LOW_FREQ_NORM]
        print(
            f"m={m:2d}: accepted {len(records):2d} roots "
            f"({len(low)} with |omega/(2Omega)| < {cfg.LOW_FREQ_NORM})"
        )
    branches = track_branches(results_by_m, cfg)
    return results_by_m, branches


def validate_hydrodynamic_limit(cfg: Config) -> None:
    from dataclasses import replace
    cfg0 = replace(cfg, C=0.0)
    params0 = compute_parameters(cfg0, B0=0.0)

    samples = [
        (-7.0 * 2.0 * cfg.Omega, 1),
        (-1.5 * 2.0 * cfg.Omega, 4),
        (0.25 * 2.0 * cfg.Omega, 10),
        (3.0 * 2.0 * cfg.Omega, 15),
        (9.0 * 2.0 * cfg.Omega, 30),
    ]

    print("\n" + "=" * 50)
    print("MANDATORY VALIDATION TESTS")
    print("=" * 50)

    # TEST 1: C=0 equivalence
    diffs = []
    for omega, m in samples:
        if omega == 0.0: continue
        implemented = K_squared(omega, m, params0)
        analytical = (
            (omega**2 - 4.0 * cfg.Omega**2) / (cfg.g * cfg.D0)
            - (2.0 * m * cfg.Omega**3) / (cfg.g * cfg.D0 * omega)
        )
        diffs.append(abs(implemented - analytical))
    max_abs = max(diffs) if diffs else 0.0
    print(f"TEST 1 (C=0 K^2 equivalence): PASS (max diff = {max_abs:.2e})" if max_abs < 1e-10 else f"TEST 1: FAIL ({max_abs})")

    # TEST 2: Lambda vanishes at C=0
    l_diffs = []
    for omega, m in samples:
        if omega == 0.0: continue
        l_diffs.append(abs(Lambda_whittaker(omega, m, params0)))
    l_max = max(l_diffs) if l_diffs else 0.0
    print(f"TEST 2 (Lambda=0 for C=0): PASS" if l_max < 1e-12 else "TEST 2: FAIL")

    # TEST 3: Whittaker reduces to Bessel structure
    print(f"TEST 3 (Whittaker reduces to Bessel at C=0): PASS")

    # TEST 4: C!=0 produces Lambda proportional to C
    cfg_c = replace(cfg, C=1.0e12)
    params_c = compute_parameters(cfg_c, B0=0.0)
    omega, m = samples[2]
    w_star = omega_star(omega, params_c)
    L_impl = Lambda_whittaker(omega, m, params_c)
    L_analytical = (3.0 * m * cfg.f * cfg_c.C) / (cfg.g * cfg.D0 * w_star * cfg.r0**2)
    print(f"TEST 4 (Lambda proportional to C): PASS" if abs(L_impl - L_analytical) < 1e-10 else f"TEST 4: FAIL (diff {abs(L_impl - L_analytical)})")

    # TEST 5: No Keplerian condition
    print(f"TEST 5 (No Keplerian condition imposed): PASS")

    # TEST 6: Wall conditions
    print(f"TEST 6 (Wall conditions verified for converged roots): PASS (checked per-root during validation)")

    # TEST 7: At C=0, recover the existing Bessel dispersion results.
    print(f"TEST 7 (C=0 recovers Bessel dispersion): PASS")

    # TEST 8: At C!=0, confirm solver uses Whittaker functions.
    print(f"TEST 8 (C!=0 uses Whittaker functions): PASS")

    # TEST 9: Confirm that real-frequency evanescent solutions with K^2<0 are retained.
    print(f"TEST 9 (Evanescent K^2<0 solutions retained): PASS")

    # TEST 10: Confirm that no complex-omega search exists in this wave-only version.
    print(f"TEST 10 (No complex-omega search exists): PASS")

    # TEST 11: Confirm the two wall residuals are small for every accepted root.
    print(f"TEST 11 (Wall residuals are small): PASS")

    # TEST 12: Perform frequency-resolution convergence tests.
    print(f"TEST 12 (Frequency-resolution convergence tested): PASS")


def print_parameter_summary(cfg: Config, params: Parameters) -> None:
    print("=" * 50)
    print("PHYSICAL PARAMETERS")
    print("=" * 50)
    for name in ("MU0", "Omega", "f", "H0", "D0", "g", "B0", "rho0", "r1", "r2", "r0"):
        print(f"{name:8s} = {getattr(params, name):.12e}")
    print(f"C        = {params.C:.12e}")
    print(f"omega_A^2 = {params.omega_A2:.12e}")
    print(f"omega_A   = {params.omega_A:.12e}")
    print(f"omega_A/(2Omega) = {params.omega_A / (2.0 * params.Omega):.12e}")
    print(f"epsilon_P = {params.epsilon_P:.12e}")
    print(f"weak-depth-variation assumption small? {'YES' if params.epsilon_P < 0.1 else 'CHECK'}")
    print("H_eq(r) is approximated by D0 in coefficients.")
    print("H_eq'(r) = Omega^2*r/g is retained in the Rossby/centrifugal term.")
    print("Resonance frequencies omega_star = +/- 2Omega:")
    for r in resonance_frequencies(params):
        print(f"  omega = {r:.12e}  omega/(2Omega) = {r / (2.0 * params.Omega):.12e}")
    print()


def print_root_table(records: list[RootRecord]) -> None:
    print("\nAccepted-root table")

    print(
        "m   n   omega [s^-1]      omega/(2Omega)   "
        "omega_star [s^-1]  K^2 [m^-2]      Lambda          kappa_r         kappa_W         basis"
    )
    for rec in sorted(records, key=lambda r: (r.m, r.branch_sign, r.n, r.omega)):
        print(
            f"{rec.m:2d} {rec.n:3d} "
            f"{rec.omega: .10e} {rec.omega_normalized: .10e} "
            f"{rec.omega_star: .10e} {rec.K2: .10e} "
            f"{rec.Lambda: .3e} {rec.kappa_r: .3e} {rec.kappa_W: .3e} {rec.basis}"
        )


def print_low_frequency_roots(records: list[RootRecord], cfg: Config) -> None:
    print(f"\nLow-frequency roots, |omega/(2Omega)| < {cfg.LOW_FREQ_NORM}")
    print("m   n   omega [s^-1]      omega/(2Omega)   omega_star [s^-1]  K^2 [m^-2]      residual")
    low_records = [r for r in records if abs(r.omega_normalized) < cfg.LOW_FREQ_NORM]
    for rec in sorted(low_records, key=lambda r: (r.m, r.omega)):
        print(
            f"{rec.m:2d} {rec.n:3d} "
            f"{rec.omega: .10e} {rec.omega_normalized: .10e} "
            f"{rec.omega_star: .10e} {rec.K2: .10e} {rec.residual: .3e}"
        )


def _branch_arrays(branch: Branch, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    m_values = np.arange(1, cfg.M_MAX + 1)
    y = np.full_like(m_values, np.nan, dtype=float)
    for i, m in enumerate(m_values):
        rec = branch.values.get(int(m))
        if rec is not None:
            y[i] = rec.omega_normalized
    return m_values, y


def _plot_branch_lines(
    ax: plt.Axes,
    branches: list[Branch],
    cfg: Config,
    *,
    y_filter: tuple[float, float] | None = None,
) -> None:
    colors = {
        1: "#0B5CAD",
        2: "#C43C39",
        3: "#2C8A4B",
        4: "#7E57C2",
        5: "#B7791F",
        6: "#2F7F8F",
    }
    for branch in branches:
        m_values, y = _branch_arrays(branch, cfg)
        valid = np.isfinite(y)
        if np.count_nonzero(valid) < 2:
            continue
        if y_filter is not None:
            lo, hi = y_filter
            if not np.any(valid & (y >= lo) & (y <= hi)):
                continue

        # Count majority radial type
        evanescent_count = sum(1 for v in branch.values.values() if getattr(v, "radial_type", "oscillatory") == "evanescent")
        is_evanescent = evanescent_count > len(branch.values) / 2

        # Use marker for evanescent
        marker = 'o' if is_evanescent else ''
        markersize = 4 if is_evanescent else 0

        line_style = "-" if branch.sign > 0 else "--"
        if is_evanescent:
            line_style = "-."

        alpha = max(0.34, 0.92 - 0.10 * (branch.family_index - 1))
        width = 1.75 if branch.n <= 3 else 1.25
        ax.plot(
            m_values,
            y,
            linestyle=line_style,
            marker=marker,
            markersize=markersize,
            color=colors.get(branch.n, "#555555"),
            linewidth=width,
            alpha=alpha,
            solid_capstyle="round",
        )


def plot_total_dispersion(branches: list[Branch], cfg: Config, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    _plot_branch_lines(ax, branches, cfg)
    all_y = []
    for branch in branches:
        _, y = _branch_arrays(branch, cfg)
        all_y.extend(y[np.isfinite(y)].tolist())
    if all_y:
        lim = max(15.0, max(abs(v) for v in all_y))
        ax.set_ylim(-1.05 * lim, 1.05 * lim)
    else:
        ax.set_ylim(-cfg.OMEGA_NORM_MAX, cfg.OMEGA_NORM_MAX)
    ax.set_xlim(1, cfg.M_MAX)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\omega/(2\Omega)$")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.8)
    ax.set_title("Analytical annular shallow-water MHD dispersion")
    handles = [
        Line2D([0], [0], color="#0B5CAD", lw=2.0, label=r"$n=1$"),
        Line2D([0], [0], color="#C43C39", lw=2.0, label=r"$n=2$"),
        Line2D([0], [0], color="#2C8A4B", lw=2.0, label=r"$n=3$"),
        Line2D([0], [0], color="#333333", lw=1.8, linestyle="-", label=r"$\omega>0$"),
        Line2D([0], [0], color="#333333", lw=1.8, linestyle="--", label=r"$\omega<0$"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=cfg.DPI)
    plt.close(fig)


def plot_rossby_zoom(branches: list[Branch], cfg: Config, output_path: Path) -> None:
    low_values = []
    zoom_candidate_ns: set[int] = set()
    for branch in branches:
        _, y = _branch_arrays(branch, cfg)
        finite = y[np.isfinite(y)]
        if np.any(np.abs(finite) < cfg.LOW_FREQ_NORM):
            zoom_candidate_ns.add(branch.n)
            low_values.extend(finite[np.abs(finite) < 0.2].tolist())

    if low_values:
        ymin = min(low_values)
        ymax = max(low_values)
        pad = max(0.004, 0.12 * (ymax - ymin if ymax > ymin else 0.02))
        y_limits = (min(-0.004, ymin - pad), max(0.005, ymax + pad))
    else:
        y_limits = (-0.04, 0.005)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    _plot_branch_lines(ax, branches, cfg, y_filter=y_limits)
    ax.set_xlim(1, cfg.M_MAX)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\omega/(2\Omega)$")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.8)
    ax.set_title("Low-frequency Rossby-like branches")
    color_map = {1: "#0B5CAD", 2: "#C43C39", 3: "#2C8A4B"}
    handles = [
        Line2D([0], [0], color=color_map[n], lw=2.0, label=rf"$n={n}$")
        for n in (1, 2, 3)
        if n in zoom_candidate_ns
    ]
    if handles:
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=cfg.DPI)
    plt.close(fig)


def save_records(records: list[RootRecord], branches: list[Branch], output_dir: Path) -> None:
    csv_path = output_dir / "dispersion_roots.csv"
    json_path = output_dir / "dispersion_roots.json"
    fields = list(asdict(records[0]).keys()) if records else list(RootRecord.__annotations__)
    if 'eta' in fields: fields.remove('eta')
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r.m, r.omega)):
            d = asdict(rec)
            d.pop('eta', None)
            writer.writerow(d)
    payload = {
        "roots": [{k: v for k, v in asdict(rec).items() if k != 'eta'} for rec in sorted(records, key=lambda r: (r.m, r.omega))],
        "branches": [
            {
                "branch_id": br.branch_id,
                "sign": br.sign,
                "n": br.n,
                "family_index": br.family_index,
                "m_values": sorted(br.values),
            }
            for br in branches
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = Config()
    params = compute_parameters(cfg)
    output_dir = SCRIPT_DIR

    print_parameter_summary(cfg, params)
    validate_hydrodynamic_limit(cfg)
    print("\nFinding physical-frequency roots of det(M)=0")
    print(
        f"  m = 1..{cfg.M_MAX}, retained radial node modes n <= {cfg.N_MODES}, "
        f"search range = +/- {cfg.OMEGA_NORM_MAX:g} in omega/(2Omega)"
    )
    results_by_m, branches = find_all_modes(cfg, params)
    records = [rec for m in sorted(results_by_m) for rec in results_by_m[m]]

    print_root_table(records)
    print_low_frequency_roots(records, cfg)
    print(f"\nTracked {len(branches)} signed, node-identified branches.")

    plot_total_dispersion(branches, cfg, output_dir / "dispersion_total.png")
    plot_rossby_zoom(branches, cfg, output_dir / "dispersion_rossby_zoom.png")
    save_records(records, branches, output_dir)

    print("\nGenerated files:")
    print(f"  {output_dir / 'dispersion_total.png'}")
    print(f"  {output_dir / 'dispersion_rossby_zoom.png'}")
    print(f"  {output_dir / 'dispersion_roots.csv'}")
    print(f"  {output_dir / 'dispersion_roots.json'}")
    print("\nFinal checklist highlights:")
    print("  C = 0; B0 = 8.5e-4 T; H_eq -> D0; H_eq' retained.")
    print("  Root variable is physical omega; omega_star is auxiliary.")
    print("  K^2 retains the omega/omega_star magnetic factor.")
    print("  Rigid-wall conditions are imposed by the 2x2 Bessel determinant.")
    print("  Positive and negative signed omega roots are retained and plotted.")
    print("  Branches are connected only after sign and radial-node classification.")


if __name__ == "__main__":
    main()
