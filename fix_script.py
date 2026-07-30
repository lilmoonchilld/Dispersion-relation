with open("plot_kelvin_variation_collocation_standalone.py", "r") as f:
    text = f.read()

text = text.replace(r"ax.set_ylabel(r'$rac{\sigma_k^{K\pm}}{f_e}$'", "ax.set_ylabel(r'$\\frac{\\sigma_k^{K\\pm}}{f_e}$'")

with open("plot_kelvin_variation_collocation_standalone.py", "w") as f:
    f.write(text)
