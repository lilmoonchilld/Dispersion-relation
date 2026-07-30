with open("swmhd.py", "r") as f:
    text = f.read()

text = text.replace(r"ax_obj.set_ylabel(r'Normalised frequency $\hat\omega = \omega/(2\Omega)$', fontsize=13)", "ax_obj.set_ylabel(r'Normalised frequency $\\hat\\omega = \\omega/(2\\Omega)$', fontsize=13)")
text = text.replace("    '\\n'", "    '\\n'")
text = text.replace("    r'Magneto-Poincaré (blue), Magneto-Kelvin (green), Rossby (red), Magnetostrophic (orange)',", "    r'Magneto-Poincaré (blue), Magneto-Kelvin (green), Rossby (red), Magnetostrophic (orange)',")

with open("swmhd.py", "w") as f:
    f.write(text)
