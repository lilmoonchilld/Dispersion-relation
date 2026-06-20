import mpmath
import plot_dispersion as pd

def find_roots(m, guesses):
    roots = []
    print(f"\n--- Finding roots for m={m} ---")
    for g in guesses:
        try:
            root = mpmath.findroot(lambda w: pd.D_determinant(w, m),
                                   g, solver='muller', tol=1e-10, maxsteps=50)
            if not mpmath.isnan(root.real) and not mpmath.isnan(root.imag):
                val = abs(pd.D_determinant(root, m))
                if val < 1e-3:
                    # check for duplicates
                    is_dup = False
                    for r in roots:
                        if abs(r - complex(root)) < 1e-4:
                            is_dup = True
                            break
                    if not is_dup:
                        roots.append(complex(root))
                        print(f"Found root from guess {g}: {root} (D={val:.2e})")
        except Exception as e:
            pass
    return roots

if __name__ == "__main__":
    g_m1 = [complex(-2.89, 0), complex(-0.79, -0.08), complex(-0.79, 0.08), complex(0.79, 0), complex(3.42, 0)]
    roots_m1 = find_roots(1, g_m1)

    g_m2 = [complex(-2.89, 0), complex(-0.79, -0.08), complex(-0.79, 0.08), complex(0.79, 0), complex(3.42, 0)]
    roots_m2 = find_roots(2, g_m2)

    # Let's save the roots for the tracing step
    with open('roots_m1_m2.txt', 'w') as f:
        f.write("m=1\n")
        for r in roots_m1:
            f.write(f"{r.real},{r.imag}\n")
        f.write("m=2\n")
        for r in roots_m2:
            f.write(f"{r.real},{r.imag}\n")
