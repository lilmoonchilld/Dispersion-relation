import re

with open('dispersion.py', 'r') as f:
    content = f.read()

old_summary = """def print_parameter_summary(cfg: Config):
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
    print(f"m range     = 1 ... 20")
    print(f"n retained  = 1 ... {N_RADIAL_MODES}")
    print("root search = direct FindRoot + continuation")
    print("radial k_r^2 = K^2 + Lambda/r - m^2/r^2")
    print("=" * 80)"""

new_summary = """def print_parameter_summary(cfg: Config):
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
    print(f"m range     = 1 ... 30")
    print("root search = direct FindRoot + continuation")
    print("radial k_r^2 = K^2 + Lambda/r - m^2/r^2")
    print("=" * 80)"""

content = content.replace(old_summary, new_summary)

old_table = """def print_root_table(records, cfg: Config):
    print("\\nAccepted roots")
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
        )"""

new_table = """def print_root_table(records, cfg: Config):
    print("\\nFinal Summary")

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
"""

content = content.replace(old_table, new_table)

old_plot = """def plot_dispersion(records, cfg: Config):
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

    ax1.set_xlim(1, 20)
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

    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlim(1, 20)
    ax2.set_xlabel(r"Azimuthal mode number $m$")
    ax2.set_ylabel(r"$\omega/(2\Omega)$")
    ax2.set_title(
        r"Slow branches: $-1 \leq \omega/(2\Omega) \leq 1$"
    )
    ax2.set_xticks(np.arange(1, 21, 1))
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

    print(f"\\nDispersion diagram saved to: {output.resolve()}")"""

new_plot = """def plot_dispersion(records, cfg: Config):
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

    # Filter for mostly real roots
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
                marker="o",
                markersize=3,
                label=label,
            )

    # 2. Slow branches / Kelvin (evanescent / other)
    slow_recs = [r for r in real_records if r.mode_family != "Poincare"]
    # Group by mode_family or sign for slow branches
    for sign in (-1, +1):
        pts = [r for r in slow_recs if r.sign == sign]
        pts.sort(key=lambda r: r.m)

        # Connect branches only if m step is 1 to avoid crossing lines
        if len(pts) > 0:
            m_vals = []
            w_vals = []
            for r in pts:
                m_vals.append(r.m)
                w_vals.append(float(r.omega_real / cfg.f))

            ax2.scatter(
                m_vals, w_vals,
                color=colors[sign],
                marker="x",
                s=20,
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
    ax1.legend(frameon=False, fontsize=8, ncol=2)

    # ------------------------------------------------------------
    # Slow-branch zoom
    # ------------------------------------------------------------
    ax2.axhline(0.0, linewidth=0.8, color="black")
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlim(1, 30)
    ax2.set_xlabel(r"Azimuthal mode number $m$")
    ax2.set_ylabel(r"$\omega/(2\Omega)$")
    ax2.set_title(r"Slow branches: $-1 \leq \omega/(2\Omega) \leq 1$")
    ax2.set_xticks(np.arange(1, 31, 2))
    ax2.grid(True, linewidth=0.4, alpha=0.30)
    ax2.legend(frameon=False, fontsize=8)

    output = Path("SWMHD_dispersion_Mathematica_style.png")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"\\nDispersion diagram saved to: {output.resolve()}")"""

content = content.replace(old_plot, new_plot)

with open('dispersion.py', 'w') as f:
    f.write(content)
print("Updated plotting and summary.")
