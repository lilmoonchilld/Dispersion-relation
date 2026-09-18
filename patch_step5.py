import re

with open('dispersion.py', 'r') as f:
    content = f.read()

# Replace seed_roots_m1
old_seed_roots = """def seed_roots_m1(cfg: Config):
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

    return deduplicate_roots(roots, cfg)"""

new_seed_roots = """def seed_roots_m1(cfg: Config):
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

    return deduplicate_roots(roots, cfg)"""

content = content.replace(old_seed_roots, new_seed_roots)

# Replace initialize_branches
old_init_branches = """def initialize_branches(cfg: Config):
    roots = seed_roots_m1(cfg)

    positive = [
        root for root in roots
        if mp.re(root) > 0
    ]

    negative = [
        root for root in roots
        if mp.re(root) < 0
    ]

    # Order branches by real part of frequency.
    positive.sort(
        key=lambda z: float(mp.re(z / cfg.f))
    )

    negative.sort(
        key=lambda z: float(mp.re(z / cfg.f))
    )

    positive = positive[:N_RADIAL_MODES]
    negative = negative[:N_RADIAL_MODES]

    initial = []

    for n, root in enumerate(positive, start=1):
        initial.append((n, +1, root))

    for n, root in enumerate(negative, start=1):
        initial.append((n, -1, root))

    return initial"""

new_init_branches = """def initialize_branches(cfg: Config):
    roots = seed_roots_m1(cfg)

    initial = []

    for root in roots:
        sign = 1 if mp.re(root) > 0 else -1

        behavior = classify_radial_behavior(root, 1, cfg)

        n = None
        mode_family = "other"
        nodes = None

        if behavior == "propagating":
            nodes = count_interior_nodes(root, 1, cfg)
            if nodes is not None:
                n = nodes + 1
                mode_family = "Poincare"
        elif behavior == "evanescent":
            mode_family = "evanescent"
        elif behavior == "turning":
            mode_family = "turning"

        # Optional: fallback for slow/Kelvin modes if low frequency
        if mode_family == "other" and abs(mp.re(root)/cfg.f) < 2.0:
            mode_family = "Kelvin/slow"

        try:
            K2 = K_squared(root, 1, cfg)
            lam = Lambda_whittaker(root, 1, cfg)
            k2_r1 = k_eff_squared(cfg.r1, root, 1, cfg)
            k2_r0 = k_eff_squared(cfg.r0, root, 1, cfg)
            k2_r2 = k_eff_squared(cfg.r2, root, 1, cfg)
            residual = abs(dispersion_determinant(root, 1, cfg))
        except Exception:
            continue

        kr0 = None
        if abs(mp.im(k2_r0)) <= mp.mpf("1e-8") and mp.re(k2_r0) > 0:
            kr0 = mp.sqrt(mp.re(k2_r0))

        rec = RootRecord(
            m=1,
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
            residual=residual
        )
        initial.append(rec)

    return initial"""

content = content.replace(old_init_branches, new_init_branches)

with open('dispersion.py', 'w') as f:
    f.write(content)
print("Replaced seed_roots and initialize_branches.")
