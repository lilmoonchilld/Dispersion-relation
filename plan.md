1. **Extend `swmhd_dispersion.py` with purely imaginary modes solver:**
   - Define a function `pure_imaginary_omega(hat_gamma)` returning `1j * hat_gamma`.
   - Define an objective `pure_gamma_sigma_min(hat_gamma, m_val)` computing the ratio of minimum to maximum singular values of the collocation matrix $L(1j \cdot \hat{\gamma}, m)$. Skip the point if $|\hat{\gamma}| \le 10^{-4}$ (or similar `min_abs_gamma`).
   - Implement `find_pure_imaginary_eigenmodes(m_val, gamma_range, n_scan)` which scans over a range like `(-5, 5)` (this should be configurable, maybe use arguments or top-level constants). It finds local minima, refines them using `minimize_scalar` within bounds, computes the actual residual $||L \eta|| / ||\eta||$, and accepts roots that pass singular value ratio (`< 1e-8`) and residual (`< 1e-6`) checks. Store $\hat{\gamma}$, $\gamma$, $\sigma_{rel}$, $residual$, and $\eta$.
2. **Compute most unstable mode:**
   - For each azimuthal mode $m = 1, \dots, M_{max}$:
     - Call the solver to find accepted pure-imaginary modes.
     - Identify the mode with the maximum $\hat{\gamma}$.
     - Print out the results for this $m$, including the most unstable mode or "no purely-imaginary mode found".
3. **Data export and printing:**
   - Compute global most unstable mode across all $m$.
   - Print a summary table of most unstable modes.
   - Save the results into a `.csv` file.
4. **Create a new stability plot:**
   - Plot $m$ versus $\hat{\gamma}_{max}$ (the most unstable mode's dimensionless growth rate).
   - Draw a horizontal line at $\hat{\gamma} = 0$.
   - Save the plot to `swmhd_pure_imaginary_dispersion.png`.
5. **Run tests**
   - Execute the code to confirm the original real modes run correctly and the new purely-imaginary branch also correctly executes without errors.
6. **Pre commit checks**
   - Run pre-commit instructions.
