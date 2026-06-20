import mpmath
import numpy as np
import plot_dispersion as pd

def find_roots_for_m(m, guesses):
    print(f"Finding roots for m={m}")
    roots = []

    # mpmath findroot parameters:
    # We use solver='muller' for complex root finding without derivatives.
    # We provide a tuple of 3 points for Muller's method, or just one point
    # and let mpmath generate the others. We'll use one point as the initial guess.

    for guess in guesses:
        try:
            # Setting a slightly wider tolerance since these functions can be very steep
            root = mpmath.findroot(lambda w: pd.D_determinant(w, m),
                                   guess, solver='muller', tol=1e-10, maxsteps=50)

            # Check if root is valid (not nan/inf)
            if not mpmath.isnan(root.real) and not mpmath.isnan(root.imag):
                # We can also verify it's a root
                val = abs(pd.D_determinant(root, m))
                if val < 1e-3: # Loose check just to ensure it didn't diverge
                    roots.append(complex(root))
                    print(f"Found root for guess {guess}: {root} (D val: {val:.2e})")
                else:
                    print(f"Root found for guess {guess} but D value is large: {root} (D val: {val:.2e})")
            else:
                print(f"Failed to find root for guess {guess}: resulted in NaN")
        except Exception as e:
            print(f"Error finding root for guess {guess}: {e}")

    return roots

if __name__ == "__main__":
    # Extracted initial guesses based on typical analytical/physical intuition for such systems
    # For m=1:
    # - Kelvin mode ~ +Omega (2.0)
    # - Poincare modes (inertial-gravity) ~ +/- sqrt(f^2 + k^2 c^2) (typically > f = 4.0)
    # - Rossby mode ~ low frequency (< 0.5)

    guesses_m1 = [
        # Poincare-like
        complex(4.5, 0.0), complex(-4.5, 0.0),
        # Kelvin-like
        complex(2.5, 0.0), complex(-2.5, 0.0),
        # Rossby-like
        complex(0.1, 0.0), complex(-0.1, 0.0)
    ]

    roots_m1 = find_roots_for_m(1, guesses_m1)

    guesses_m2 = [
        # Poincare-like
        complex(5.0, 0.0), complex(-5.0, 0.0),
        # Kelvin-like
        complex(3.0, 0.0), complex(-3.0, 0.0),
        # Rossby-like
        complex(0.2, 0.0), complex(-0.2, 0.0)
    ]

    roots_m2 = find_roots_for_m(2, guesses_m2)

    print("\nSummary of Roots Found:")
    print("m=1 roots:", roots_m1)
    print("m=2 roots:", roots_m2)

    # Save the roots to a file for the next step
    with open('roots_initial.txt', 'w') as f:
        f.write("m=1\n")
        for r in roots_m1:
            f.write(f"{r.real},{r.imag}\n")
        f.write("m=2\n")
        for r in roots_m2:
            f.write(f"{r.real},{r.imag}\n")
