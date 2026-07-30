with open("swmhd.py", "r") as f:
    text = f.read()

import re

# Fix pbox block
text = re.sub(r'pbox = \([\s\S]*?N_col\}\$"\n\)', '''pbox = (
    f"$\\\\hat{{\\\\omega}}_A = {hat_omA:.4f}\\\\n"
    f"$\\\\hat c_0^2 = {hat_c0sq:.4f}\\\\n"
    f"$\\\\gamma = {gamma:.4f}\\\\n"
    f"$\\\\Omega = {Omega:.1e}$ rad/s,  $H_0 = {H0}$ m\\\\n"
    f"$r_1 = {r1:.1e}$ m,  $r_2 = {r2:.1e}$ m\\\\n"
    f"$N = {N_col}$"
)''', text)

text = re.sub(r"ax\.set_ylabel\(r'\$\\frac\{\\sigma_k\^\{K\\pm\}\}\{f_e\}\$', rotation=0, labelpad=25, va='center', fontsize=24\)", r"ax.set_ylabel(r'$\\frac{\\sigma_k^{K\\pm}}{f_e}$', rotation=0, labelpad=25, va='center', fontsize=24)", text)

with open("swmhd.py", "w") as f:
    f.write(text)
