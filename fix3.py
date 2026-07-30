with open("swmhd.py", "r") as f:
    text = f.read()

import re

# Fix pbox block properly with r-strings
text = re.sub(r'pbox = \([\s\S]*?N_col\}\$"\n\)', '''pbox = (
    rf"$\\hat{{\\omega}}_A = {hat_omA:.4f}\\n"
    rf"$\\hat c_0^2 = {hat_c0sq:.4f}\\n"
    rf"$\\gamma = {gamma:.4f}\\n"
    rf"$\\Omega = {Omega:.1e}$ rad/s,  $H_0 = {H0}$ m\\n"
    rf"$r_1 = {r1:.1e}$ m,  $r_2 = {r2:.1e}$ m\\n"
    rf"$N = {N_col}$"
)''', text)

with open("swmhd.py", "w") as f:
    f.write(text)
