import numpy as np
import mpmath
import matplotlib.pyplot as plt
import plot_dispersion as pd
from multiprocessing import Pool
import sys

def evaluate_point(args):
    re_val, im_val, m = args
    omega_val = complex(re_val, im_val)
    if abs(omega_val) < 1e-4:
        return (re_val, im_val, np.nan)
    try:
        d_val = pd.D_determinant(omega_val, m)
        if d_val != d_val:
            return (re_val, im_val, np.nan)
        return (re_val, im_val, float(mpmath.fabs(d_val)))
    except Exception as e:
        return (re_val, im_val, np.nan)

def generate_heatmap_parallel(m, re_bounds, im_bounds, grid_size, title, filename):
    print(f"Generating heatmap for m={m}, Re: {re_bounds}, Im: {im_bounds}")

    re_vals = np.linspace(re_bounds[0], re_bounds[1], grid_size)
    im_vals = np.linspace(im_bounds[0], im_bounds[1], grid_size)

    Re, Im = np.meshgrid(re_vals, im_vals)
    Z_abs = np.zeros_like(Re, dtype=float)

    tasks = []
    for i in range(grid_size):
        for j in range(grid_size):
            tasks.append((Re[i, j], Im[i, j], m))

    # Use multiprocessing to speed up
    with Pool() as p:
        results = p.map(evaluate_point, tasks)

    for i in range(grid_size):
        for j in range(grid_size):
            # Results are in order of tasks appended
            idx = i * grid_size + j
            _, _, val = results[idx]
            Z_abs[i, j] = val

    plt.figure(figsize=(10, 8))
    log_Z = np.log10(np.where((Z_abs == 0) | np.isnan(Z_abs), 1e-15, Z_abs))

    plt.contourf(Re, Im, log_Z, levels=50, cmap='viridis')
    plt.colorbar(label='log10 |D(omega, m)|')
    plt.xlabel('Re(omega)')
    plt.ylabel('Im(omega)')
    plt.title(title)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    # We use a grid size of 50 to keep the compute time manageable (2500 points = ~160 seconds per map with 1 core, much faster with Pool)
    grid_sz = 50
    generate_heatmap_parallel(m=1, re_bounds=[-5, 5], im_bounds=[-1, 1], grid_size=grid_sz,
                    title='Determinant Magnitude |D| for m=1 (Kelvin/Poincare)',
                    filename='heatmap_m1_kelvin_poincare.png')

    generate_heatmap_parallel(m=2, re_bounds=[-5, 5], im_bounds=[-1, 1], grid_size=grid_sz,
                    title='Determinant Magnitude |D| for m=2 (Kelvin/Poincare)',
                    filename='heatmap_m2_kelvin_poincare.png')

    generate_heatmap_parallel(m=1, re_bounds=[-0.5, 0.5], im_bounds=[-1, 1], grid_size=grid_sz,
                    title='Determinant Magnitude |D| for m=1 (Rossby)',
                    filename='heatmap_m1_rossby.png')

    generate_heatmap_parallel(m=2, re_bounds=[-0.5, 0.5], im_bounds=[-1, 1], grid_size=grid_sz,
                    title='Determinant Magnitude |D| for m=2 (Rossby)',
                    filename='heatmap_m2_rossby.png')
