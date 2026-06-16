# Numerical Solution of the Annular SWMHD Dispersion Relation

## 1. Introduction
This document explains the numerical implementation used to compute and plot the global dispersion relation for Shallow Water Magnetohydrodynamics (SWMHD) in a Keplerian annulus.

The problem is cast as a 1D global eigenvalue problem in the radial direction $r$. Rather than solving the highly nonlinear 2nd-order ODE analytically (which relies on root-finding for $\omega$), we use a standard **Uniform Finite Difference Method (FDM)** on the complete first-order system to form a generalized linear eigenvalue problem.

## 2. Background State and Parameters
The computational domain covers a radial range $[r_1, r_2]$. The parameters adopted to model the system are:
- $G = 6.67430 \times 10^{-11} \text{ m}^3\text{kg}^{-1}\text{s}^{-2}$
- $M = 5.683 \times 10^{26} \text{ kg}$ (Saturn mass)
- Mean radius $r_0 = 1 \times 10^8 \text{ m}$
- $H_0 = 10.0 \text{ m}$
- $\rho = 100.0 \text{ kg m}^{-3}$
- $\mu_0 = 4\pi \times 10^{-7} \text{ T m A}^{-1}$
- $B_0 = 2 \times 10^{-5} \text{ T}$
- Boundaries: $r_1 = 8 \times 10^7 \text{ m}$, $r_2 = 1.2 \times 10^8 \text{ m}$.

From the relation $H(r) = c_g(r)/\Omega(r)$ and $\Omega(r) = \sqrt{GM/r^3}$, the local sound speed squared is determined rigorously as:
$$ c_g^2 = \frac{GMH_0^2}{r_0^3} = \text{const} $$
The thickness profile is thus given by $H(r) = H_0(r/r_0)^{3/2}$.

## 3. The First-Order Linearized System
The state vector of perturbations is chosen as $X = [\hat{\eta}, \hat{V}_r, \hat{V}_\theta, \hat{B}_r, \hat{B}_\theta]^T$. By dividing out $H(r)$ appropriately and applying the structural gradients, the system is expressed as $A X = \omega C X$.

The governing equations implemented in the code are:

1. **Continuity Equation:**
   $$ -i\omega\hat{\eta} + \frac{d\hat{V}_r}{dr} + \frac{5}{2r}\hat{V}_r + \frac{im}{r}\hat{V}_\theta = 0 $$

2. **$r$-Momentum Equation:**
   $$ -i\omega\hat{V}_r + \frac{c_g^2}{H}\frac{d\hat{\eta}}{dr} - 2\Omega\hat{V}_\theta - \frac{B_0}{\mu_0\rho H}\hat{B}_r = 0 $$

3. **$\theta$-Momentum Equation:**
   $$ -i\omega\hat{V}_\theta + \frac{im c_g^2}{r H}\hat{\eta} + 2\Omega\hat{V}_r - \frac{B_0}{\mu_0\rho H}\hat{B}_\theta = 0 $$

4. **$r$-Induction Equation:**
   $$ -i\omega\hat{B}_r = -\frac{B_0}{H}\hat{V}_r $$

5. **$\theta$-Induction Equation:**
   $$ -i\omega\hat{B}_\theta = -\frac{B_0}{H}\hat{V}_\theta $$

## 4. Discretization and Matrix Construction
The physical radial domain $r \in [r_1, r_2]$ is divided into $N$ evenly spaced nodes with grid spacing $\Delta r$. We construct a 2nd-order Central Finite Difference derivative matrix $D$. The matrix takes the form:
- Interior nodes: $D_{i, i-1} = -\frac{1}{2\Delta r}$, $D_{i, i+1} = \frac{1}{2\Delta r}$
- Left boundary: 2nd-order forward difference coefficients $(-3/2, 4/2, -1/2)$
- Right boundary: 2nd-order backward difference coefficients $(1/2, -4/2, 3/2)$

The matrices $A$ and $C$ are built using $5 \times 5$ blocks of size $N \times N$. Diagonal blocks represent local algebraic operations, and blocks incorporating the derivative matrix $D$ represent radial gradients.

The right-hand side mass matrix $C$ is strictly the identity matrix $\mathbf{I}$, except at boundary rows.

## 5. Boundary Conditions
Rigid radial boundaries dictate that the radial velocity must vanish at the inner and outer edges:
$$ \hat{V}_r(r_1) = 0, \quad \hat{V}_r(r_2) = 0 $$
In the algebraic system, the first and last rows corresponding to the $\hat{V}_r$ variable block in $A$ are overwritten with `1.0` on the diagonal, and `0.0` everywhere else. In matrix $C$, these rows are zeroed out completely. This forces $\omega \cdot 0 = 1 \cdot \hat{V}_r \implies \hat{V}_r = 0$.

## 6. Computing and Plotting the Dispersion Relation
The SciPy `scipy.linalg.eig(A, C)` function solves the generalized eigenvalue problem for a given azimuthal mode number $m$.
1. A loop runs over $m = 1, 2, 3, 4, 5$.
2. The complex frequencies $\omega$ are generated. Purely physical propagating modes have purely real frequencies (or mostly real frequencies given slight numerical roundoff error). Modes with massive imaginary components are mathematically discarded.
3. The valid real frequencies are normalized by $2\Omega(r_0)$ (the characteristic Coriolis parameter at the mean radius).
4. The normalized frequencies are plotted against $m$, yielding the final output file `dispersion_relation.png`.

## 7. Wave Mode Analysis
Based on the distribution of the normalized eigenvalues $\tilde{\omega} = \omega / (2\Omega_0)$, we can clearly identify the spectrum of SWMHD wave branches captured by the numerical solver:

1. **Poincare / Fast Gravity Waves ($|\tilde{\omega}| > 1$):**
   These are high-frequency acoustic-gravity modes, modified by rotation. They appear as the dense continuum of points stretching vertically at $|\tilde{\omega}| > 1$.

2. **Rossby Waves ($-1 < \tilde{\omega} < 0$):**
   Rossby waves are typically low-frequency, retrograde modes restoring via the potential vorticity gradient. They occupy the space below the axis but bounded by the Poincare modes.

3. **Kelvin & Prograde Modes ($0 < \tilde{\omega} < 1$):**
   These are equatorially (or boundary) trapped modes traveling in the prograde direction. They are clearly visible in the positive, sub-Poincare frequency domain.

4. **Magnetostrophic & Slow MHD Modes ($|\tilde{\omega}| \approx 0$):**
   With the introduction of the magnetic field terms $B_r$ and $B_\theta$, a new family of extremely low-frequency modes emerges. These magnetic modes rely on the Alfven restoring force (which is very weak compared to the Coriolis force, given $B_0 = 2 \times 10^{-5}$ T). They cluster heavily around $\tilde{\omega} \approx 0$.
