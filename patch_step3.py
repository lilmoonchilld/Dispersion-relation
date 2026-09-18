with open('dispersion.py', 'r') as f:
    lines = f.readlines()

out = []
in_seed_section = False
for line in lines:
    if line.startswith('def seed_roots_m1(cfg: Config):'):
        # Insert poincare seeds before this function
        out.append("""# ============================================================
# NEW: POINCARE INITIAL GUESSES
# ============================================================
def generate_poincare_seeds(m: int, cfg: Config):
    seeds = []
    L = cfg.r2 - cfg.r1
    for n in range(1, 16):
        kr_n = n * mp.pi / L

        # omega0^2 = f^2 + omega_A^2 + g H0 [ k_r,n^2 + m^2/r0^2 ]
        omega0_sq = cfg.f**2 + omega_A_squared(cfg) + cfg.g * cfg.H0 * (kr_n**2 + (m/cfg.r0)**2)
        omega0 = mp.sqrt(omega0_sq)

        for factor in (mp.mpf("0.9"), mp.mpf("1.0"), mp.mpf("1.1")):
            seeds.append(+omega0 * factor)
            seeds.append(-omega0 * factor)

    return seeds

""")
    out.append(line)

with open('dispersion.py', 'w') as f:
    f.writelines(out)
print("Inserted Poincare seeds.")
