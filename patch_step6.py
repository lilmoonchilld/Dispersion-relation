import re

with open('dispersion.py', 'r') as f:
    content = f.read()

# Replace solve_all_branches to pass rec
old_solve = """def solve_all_branches(cfg: Config):
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

    return records"""

new_solve = """def solve_all_branches(cfg: Config):
    initial = initialize_branches(cfg)

    # Print m=1 summary
    print(f"Total m=1 roots found: {len(initial)}")

    print("\\nm=1 Roots Summary")
    print(f"{'Index':<6} {'Re[omega]/(2Omega)':<25} {'Im[omega]/(2Omega)':<25} {'K^2':<20} {'k_r^2(r0)':<20} {'Behavior':<15} {'Nodes':<6} {'Family':<15} {'|D|':<10}")
    print("-" * 150)
    for i, rec in enumerate(initial):
        nodes_str = str(rec.radial_nodes) if rec.radial_nodes is not None else "--"
        print(f"{i:<6} {float(mp.re(rec.omega_norm)):<25.9e} {float(mp.im(rec.omega_norm)):<25.3e} {float(mp.re(rec.K2)):<20.3e} {float(mp.re(rec.k_eff2_r0)):<20.3e} {rec.radial_behavior:<15} {nodes_str:<6} {rec.mode_family:<15} {float(rec.residual):<10.3e}")

    records = []
    records.extend(initial) # Keep m=1 roots

    for rec in initial:
        branch = continue_branch(rec, cfg)
        records.extend(branch)

    return records"""

content = content.replace(old_solve, new_solve)

old_continue = """def continue_branch(
    n: int,
    sign: int,
    root_m1,
    cfg: Config,
):
    branch_records = []

    omega_previous = mp.mpc(root_m1)
    omega_previous_previous = None

    for m in range(1, 21):

        if m == 1:
            root = omega_previous
        else:
            try:
                if omega_previous_previous is None:
                    omega_guess = omega_previous
                else:
                    omega_guess = omega_previous + (omega_previous - omega_previous_previous)
                print(f"m=1, initial guess={mp.nstr(omega_guess / cfg.f, 6)}")
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

    return branch_records"""

new_continue = """def continue_branch(rec_m1: RootRecord, cfg: Config):
    branch_records = []
    omega_previous = mp.mpc(rec_m1.omega)
    omega_previous_previous = None

    n = rec_m1.n
    sign = rec_m1.sign
    mode_family = rec_m1.mode_family

    for m in range(2, 31):
        try:
            if omega_previous_previous is None:
                omega_guess = omega_previous
            else:
                omega_guess = omega_previous + (omega_previous - omega_previous_previous)

            # Use diagnostic print correctly
            # print(f"m={m}, n={n}, sign={sign}, guess={mp.nstr(omega_guess / cfg.f, 6)}")
            root = direct_findroot(omega_guess, m, cfg)
        except Exception:
            break

        if not root_is_valid(root):
            break

        if sign > 0 and mp.re(root) <= 0:
            break
        if sign < 0 and mp.re(root) >= 0:
            break

        # Check jump
        if omega_previous_previous is not None:
            if abs((root - omega_previous) / cfg.f) > 5.0: # arbitrary large jump threshold
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

        branch_records.append(
            RootRecord(
                m=m,
                omega=root,
                omega_norm=normalized_frequency(root, cfg),
                omega_real=mp.re(root),
                omega_imag=mp.im(root),
                sign=sign,
                n=n,
                mode_family=mode_family, # preserve initial family classification
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

    return branch_records"""

content = content.replace(old_continue, new_continue)

with open('dispersion.py', 'w') as f:
    f.write(content)
print("Updated branch continuation.")
