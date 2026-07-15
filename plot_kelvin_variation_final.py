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

# Parameters mapped perfectly from swmhd.py defaults to mimic the plot
MU0   = 4 * np.pi * 1e-7
Omega = 0.5e-4
H0    = 500.0
g     = 9.81
rho0  = 1000.0
C     = 1.5e9
B0    = 0.05

f_e = 2 * Omega
c0sq = g * H0
c0 = np.sqrt(c0sq)
Rd = c0 / f_e

VA2 = B0**2 / (MU0 * rho0)
omA2 = VA2 / H0**2
hat_omA = np.sqrt(omA2) / f_e

x_vals = np.linspace(0.6, 5.0, 500)

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Plot horizontal lines
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
plt.savefig('variation_kelvin_modes.png', dpi=300)
plt.savefig('variation_kelvin_modes.pdf', dpi=300)
