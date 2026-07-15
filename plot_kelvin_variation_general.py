import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Publication style settings
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

# Parameters based on swmhd.py
MU0   = 4 * np.pi * 1e-7
Omega = 0.5e-4        # rad/s
f_e = 2 * Omega
H0    = 500.0         # m
g     = 9.81          # m/s^2
rho0  = 1000.0        # kg/m^3
c0sq = g * H0
c0 = np.sqrt(c0sq)
Rd = c0 / f_e

# We want the plot against x = r0 / Rd
x_vals = np.linspace(0.6, 5.0, 500)
r0_vals = x_vals * Rd

zeta = 0.5 # r1 / r2 = 0.5
# r0 = 0.5 * (r1 + r2) = 0.5 * (0.5 r2 + r2) = 0.75 r2
# r2 = (4/3) r0
# r1 = (2/3) r0

# Let's parameterize B0 and C such that they affect the Kelvin wave.
# Wait, look at the analytical formula from Section 7 (the LaTeX doc):
# hat_omega_inner = -0.5 * sqrt( m^2 hat_c0sq / hat_r1^2 - 4 hat_omA2 )
# hat_omega_outer = +0.5 * sqrt( m^2 hat_c0sq / hat_r2^2 - 4 hat_omA2 )

# Let's look at the parameters in the swmhd.py snippet the user provided:
B0 = 0.0 # From the prompt: "do it for the case C not equal to zero and B_0 not equal to zero as generalised for my system see previous chats"
# But we need specific values of B0 and C to plot the curves, OR we just use the ones from the previous chat.
# In the base code: B0 = 0, C = 1.5e9. Wait, the user asked for B_0 != 0 and C != 0. Let's set some non-zero B0.
# The user wants "variation ... as generalised for my system".
# Let's look at the mathematical formula.
# hat_omega = omega / f_e
# hat_omA2 = VA2 / (f_e^2 H0^2)
# The Kelvin wave dispersion relation from the doc:
# omega_inner / f_e = -0.5 * sqrt( (m c0 / (f_e r1))^2 - 4 * (VA / (f_e H0))^2 )
# omega_outer / f_e = +0.5 * sqrt( (m c0 / (f_e r2))^2 - 4 * (VA / (f_e H0))^2 )
# Let's map r1, r2 to r0:
# c0 / (f_e r1) = c0 / (f_e (2/3) r0) = (3/2) * (Rd / r0) = (3/2) / x
# c0 / (f_e r2) = c0 / (f_e (4/3) r0) = (3/4) * (Rd / r0) = (3/4) / x

# Note that C (radial gravity) does NOT appear in the frequency equation for Kelvin waves in the dimensionless analytical WKB relation (eq 96 & 97 in LaTeX).
# C only appears in the spatial structure (eq 95).
# So the frequency ONLY depends on B0 (via VA).

# Let's set a value for B0. What was hat_omA2 in previous code?
# B0 was 0 in the snippet, but let's set hat_VA2sq (from the print statement: hat_VA2sq: 0.0000).
# The user says "do it for the case C not equal to zero and B_0 not equal to zero". Let's define a fixed hat_omA.
# Let hat_omA = 0.2 (for example)
hat_omA = 0.2

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for y_val in [-1, 0, 1]:
    ax.axhline(y_val, color='gray', lw=1.0, alpha=0.5)

line_styles = {1: '-', 3: '--', 5: ':'}
color = '#0055ff'

for k in [1, 3, 5]:
    ls = line_styles[k]

    # Inner (minus) branch
    arg_in = (k * 1.5 / x_vals)**2 - 4 * hat_omA**2
    valid_in = arg_in >= 0
    sigma_in = np.full_like(x_vals, np.nan)
    sigma_in[valid_in] = -0.5 * np.sqrt(arg_in[valid_in])

    # Outer (plus) branch
    arg_out = (k * 0.75 / x_vals)**2 - 4 * hat_omA**2
    valid_out = arg_out >= 0
    sigma_out = np.full_like(x_vals, np.nan)
    sigma_out[valid_out] = 0.5 * np.sqrt(arg_out[valid_out])

    ax.plot(x_vals, sigma_out, color=color, ls=ls, lw=2.0)
    ax.plot(x_vals, sigma_in, color=color, ls=ls, lw=2.0)

ax.text(1.2, 0.45, r'$k = 1$', fontsize=18, color='k', ha='center', va='center')
ax.text(2.6, 0.8, r'$k = 3$', fontsize=18, color='k', ha='center', va='center')
ax.text(4.2, 1.15, r'$k = 5$', fontsize=18, color='k', ha='center', va='center')

ax.set_xlim(0.5, 5.0)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
ax.set_ylabel(r'$\frac{\sigma_k^{K\pm}}{f_e}$', rotation=0, labelpad=25, va='center', fontsize=24)
ax.set_xlabel(r'$r_0/R_d$', fontsize=22)
ax.set_yticklabels(['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5'])
ax.set_xticklabels(['1', '2', '3', '4', '5'])
plt.tight_layout()
plt.savefig('kelvin_modes_variation_general.png', dpi=300)
