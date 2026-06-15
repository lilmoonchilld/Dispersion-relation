import sympy as sp

def main():
    # Define variables
    omega = sp.Symbol('omega')

    # We will use intermediate variables A, B, C_val, D, Omega_val for simplicity
    A, B, C_val, D, Omega_val = sp.symbols('A B C_val D Omega_val', real=True, positive=True)

    # Original variables for substitution if needed
    B0, mu0, rho, H, Omega, G, M, r, GN, m, kr = sp.symbols('B_0 mu_0 rho H Omega G M r G_N m k_r', real=True, positive=True)

    # k definitions
    k_sq = kr**2 + m**2 / r**2
    k_abs = sp.sqrt(k_sq)

    # Definition of intermediate variables
    A_def = B0**2 / (mu0 * rho * H**2)
    B_def = H * (G * M * H / r**3 - 2 * sp.pi * GN * rho / k_abs)
    C_def = k_sq - sp.I * 5 * kr / (2 * r)
    D_def = 5 * m * Omega / r**2

    # Coefficients using intermediate variables
    C4 = 1
    C3 = 0
    C2 = -(2*A + 4*Omega_val**2 + B*C_val)
    C1 = -(D * B)
    C0 = A * (A + B * C_val)

    # Polynomial
    poly = C4 * omega**4 + C3 * omega**3 + C2 * omega**2 + C1 * omega + C0

    # Solve for omega
    roots = sp.solve(poly, omega)

    # Write to roots.tex
    with open('roots.tex', 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\usepackage{amsmath}\n')
        f.write('\\usepackage{geometry}\n')
        f.write('\\geometry{margin=1in}\n')
        f.write('\\begin{document}\n\n')

        f.write('\\section*{Definitions of Intermediate Variables}\n')
        f.write('\\begin{align*}\n')
        f.write('A &= ' + sp.latex(A_def) + ' \\\\\n')
        f.write('B &= ' + sp.latex(B_def) + ' \\\\\n')
        f.write('C &= ' + sp.latex(C_def).replace('C_{val}', 'C') + ' \\\\\n')
        f.write('D &= ' + sp.latex(D_def) + ' \\\\\n')
        f.write('\\Omega_{val} &= \\Omega \\\\\n')
        f.write('\\end{align*}\n\n')

        f.write('\\section*{Coefficients}\n')
        f.write('\\begin{align*}\n')
        f.write('C_4 &= ' + sp.latex(C4) + ' \\\\\n')
        f.write('C_3 &= ' + sp.latex(C3) + ' \\\\\n')
        f.write('C_2 &= ' + sp.latex(C2).replace('C_{val}', 'C').replace('\\Omega_{val}', '\\Omega') + ' \\\\\n')
        f.write('C_1 &= ' + sp.latex(C1).replace('C_{val}', 'C').replace('\\Omega_{val}', '\\Omega') + ' \\\\\n')
        f.write('C_0 &= ' + sp.latex(C0).replace('C_{val}', 'C').replace('\\Omega_{val}', '\\Omega') + ' \\\\\n')
        f.write('\\end{align*}\n\n')

        f.write('\\section*{Roots}\n')
        f.write('The roots of the equation $C_4\\omega^4 + C_3\\omega^3 + C_2\\omega^2 + C_1\\omega + C_0 = 0$ are:\n')
        f.write('\\begin{enumerate}\n')
        for i, root in enumerate(roots):
            f.write('\\item $\\omega_{' + str(i+1) + '} = ' + sp.latex(sp.simplify(root)).replace('C_{val}', 'C').replace('\\Omega_{val}', '\\Omega') + '$\n\n')
        f.write('\\end{enumerate}\n')

        f.write('\\end{document}\n')

if __name__ == "__main__":
    main()
