with open('dispersion.py', 'r') as f:
    lines = f.readlines()

out = []
for line in lines:
    if line.startswith('def seed_roots_m1(cfg: Config):'):
        out.append("""# ============================================================
# NEW: RADIAL BEHAVIOR AND NODE COUNTING
# ============================================================
def classify_radial_behavior(omega, m: int, cfg: Config):
    try:
        r_vals = mp.linspace(cfg.r1, cfg.r2, 100)
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
        r_vals = mp.linspace(cfg.r1, cfg.r2, 200)
        eta_vals = []
        for r in r_vals:
            M, W, _, _, _, _, _ = whittaker_basis(omega, m, cfg, r)
            eta = A*M + B*W
            # Use real part of eta for node counting since for propagating modes it should be mainly real or have a fixed phase
            eta_vals.append(mp.re(eta))

        # Filter noise and count crossings (interior only)
        # First find max amplitude to set threshold
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

""")
    out.append(line)

with open('dispersion.py', 'w') as f:
    f.writelines(out)
print("Inserted diagnostic functions.")
