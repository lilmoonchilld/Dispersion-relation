import numpy as np
import mpmath
import plot_dispersion as pd

def fast_grid_minima(m, re_bounds, im_bounds, grid_size=20):
    re_vals = np.linspace(re_bounds[0], re_bounds[1], grid_size)
    im_vals = np.linspace(im_bounds[0], im_bounds[1], grid_size)
    Z_abs = np.zeros((grid_size, grid_size))

    for i, r in enumerate(re_vals):
        for j, im in enumerate(im_vals):
            w = complex(r, im)
            if abs(w) < 1e-4:
                Z_abs[i, j] = 1e10
                continue
            try:
                val = float(mpmath.fabs(pd.D_determinant(w, m)))
                Z_abs[i, j] = val if not np.isnan(val) else 1e10
            except:
                Z_abs[i, j] = 1e10

    guesses = []
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            val = Z_abs[i, j]
            if val < 1e9:
                neighbors = [
                    Z_abs[i-1, j], Z_abs[i+1, j],
                    Z_abs[i, j-1], Z_abs[i, j+1]
                ]
                if all(val < n for n in neighbors):
                    if val < 10.0: # relaxed constraint
                        guesses.append(complex(re_vals[i], im_vals[j]))
    return guesses

if __name__ == "__main__":
    print("Finding m=1 minima...")
    g_m1 = fast_grid_minima(1, [-5, 5], [-0.5, 0.5], 20)
    print("m=1 guesses:", g_m1)

    print("Finding m=2 minima...")
    g_m2 = fast_grid_minima(2, [-5, 5], [-0.5, 0.5], 20)
    print("m=2 guesses:", g_m2)

    with open('guesses.txt', 'w') as f:
        f.write(f"{g_m1}\n")
        f.write(f"{g_m2}\n")
