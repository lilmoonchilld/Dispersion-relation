import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the roots from CSV
    df = pd.read_csv('outputs/dispersion_roots.csv')

    # 1. Matplotlib parameters for publication quality
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

    # Let's filter for Plot 1 (Real parts of roots classified as real vs m)
    # The classification labels are: real_positive, real_negative, real_zero, complex_unstable, complex_damped
    df_real = df[df['root_class'].isin(['real_positive', 'real_negative', 'real_zero'])]

    # Plot 1: Real-frequency dispersion relation
    plt.figure(figsize=(10, 8))
    # We want to distinguish positive and negative frequencies
    df_real_pos = df_real[df_real['real_omega_over_Omega'] >= 0]
    df_real_neg = df_real[df_real['real_omega_over_Omega'] < 0]

    plt.scatter(df_real_pos['m'], df_real_pos['real_omega_over_Omega'],
                color='blue', marker='o', s=40, label=r'$\mathrm{Re}(\omega)/\Omega \geq 0$')
    plt.scatter(df_real_neg['m'], df_real_neg['real_omega_over_Omega'],
                color='red', marker='s', s=40, label=r'$\mathrm{Re}(\omega)/\Omega < 0$')

    plt.xlabel(r'$m$')
    plt.ylabel(r'$\mathrm{Re}(\omega)/\Omega$')
    plt.title('SWMHD WKB Real Dispersion Relation at $r=r_0$')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('outputs/dispersion_real_vs_m.png', dpi=300)
    plt.close()

    # Plot 2: Imaginary-frequency dispersion relation
    # Plot imaginary parts of all roots that are classified as complex (complex_unstable, complex_damped)
    df_complex = df[df['root_class'].isin(['complex_unstable', 'complex_damped'])]

    plt.figure(figsize=(10, 8))
    df_complex_uns = df_complex[df_complex['imag_omega_over_Omega'] > 0]
    df_complex_dmp = df_complex[df_complex['imag_omega_over_Omega'] < 0]

    plt.scatter(df_complex_uns['m'], df_complex_uns['imag_omega_over_Omega'],
                color='forestgreen', marker='^', s=40, label=r'$\mathrm{Im}(\omega)/\Omega > 0$ (Unstable)')
    plt.scatter(df_complex_dmp['m'], df_complex_dmp['imag_omega_over_Omega'],
                color='purple', marker='v', s=40, label=r'$\mathrm{Im}(\omega)/\Omega < 0$ (Damped)')

    plt.xlabel(r'$m$')
    plt.ylabel(r'$\mathrm{Im}(\omega)/\Omega$')
    plt.title('SWMHD WKB Imaginary Dispersion Relation at $r=r_0$')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('outputs/dispersion_imag_vs_m.png', dpi=300)
    plt.close()

    # Plot 3: Full complex-root distribution for selected m = 1, 5, 10, 20, 30
    plt.figure(figsize=(10, 8))
    selected_m = [1, 5, 10, 20, 30]
    colors = ['black', 'orange', 'crimson', 'teal', 'magenta']
    markers = ['o', 's', '^', 'D', 'p']

    for m_val, col, mark in zip(selected_m, colors, markers):
        df_m = df[df['m'] == m_val]
        plt.scatter(df_m['real_omega_over_Omega'], df_m['imag_omega_over_Omega'],
                    color=col, marker=mark, s=80, label=f'$m = {m_val}$')

    plt.xlabel(r'$\mathrm{Re}(\omega)/\Omega$')
    plt.ylabel(r'$\mathrm{Im}(\omega)/\Omega$')
    plt.title('SWMHD WKB Root Distribution in Complex Plane')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('outputs/complex_root_distribution.png', dpi=300)
    plt.close()

    print("Successfully generated all dispersion plots in outputs/ directory!")

if __name__ == "__main__":
    main()
