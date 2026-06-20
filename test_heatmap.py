import mpmath
import numpy as np
import time
import plot_dispersion as pd

def test_grid():
    re_bounds = [-5, 5]
    im_bounds = [-1, 1]
    grid_size = 20 # much smaller grid for testing speed

    re_vals = np.linspace(re_bounds[0], re_bounds[1], grid_size)
    im_vals = np.linspace(im_bounds[0], im_bounds[1], grid_size)

    Re, Im = np.meshgrid(re_vals, im_vals)

    start_time = time.time()
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            omega_val = complex(Re[i, j], Im[i, j])
            if abs(omega_val) >= 1e-4:
                val = pd.D_determinant(omega_val, 1)
                count += 1
    end_time = time.time()
    print(f"Evaluated {count} points in {end_time - start_time:.2f} seconds.")
    print(f"Average time per point: {(end_time - start_time)/count:.4f} seconds.")

if __name__ == "__main__":
    test_grid()
