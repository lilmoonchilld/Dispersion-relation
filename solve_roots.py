import sympy as sp

def main():
    # Define variables
    omega = sp.Symbol('omega')
    k_r = sp.Symbol('k_r', real=True)
    k_theta = sp.Symbol('k_theta', real=True)
    c0_sq = sp.Symbol('c_0^2', real=True, positive=True)
    Omega_A_sq = sp.Symbol('Omega_A^2', real=True, positive=True)
    f = sp.Symbol('f', real=True)
    B = sp.Symbol('mathcal{B}', real=True)

    # We define k^2
    k_sq = k_r**2 + k_theta**2

    # A matrix elements from equations A11 to A22
    A11 = sp.I * (k_r**2 * c0_sq - Omega_A_sq - omega**2) + B * k_r
    A12 = sp.I * k_r * k_theta * c0_sq - f * omega + B * k_theta
    A21 = sp.I * k_r * k_theta * c0_sq + f * omega
    A22 = sp.I * (k_theta**2 * c0_sq - Omega_A_sq - omega**2)

    # Compute determinant
    det_M = A11 * A22 - A12 * A21

    # Expand determinant and multiply by -1 to match the derivation structure
    expanded_det = sp.expand(-det_M)

    # Reconstruct the expected quartic equation derived in the LaTeX
    # omega^4 - omega^2(k^2 c0^2 - 2Omega_A^2 + f^2 - i B k_r) + f B k_theta omega - Omega_A^2(k^2 c0^2 - Omega_A^2) + i B k_r Omega_A^2 = 0

    expected_poly = omega**4 \
                  - omega**2 * (k_sq * c0_sq - 2 * Omega_A_sq + f**2 - sp.I * B * k_r) \
                  + f * B * k_theta * omega \
                  - Omega_A_sq * (k_sq * c0_sq - Omega_A_sq) \
                  + sp.I * B * k_r * Omega_A_sq

    expanded_expected = sp.expand(expected_poly)

    # Check if they are exactly equal
    diff = sp.simplify(expanded_det - expanded_expected)

    print("--- Dispensation Relation Verification ---")
    if diff == 0:
        print("SUCCESS! The determinant of the 2x2 system matches the derived quartic dispersion relation exactly.")
    else:
        print("FAILED! The equations do not match.")
        print(f"Difference: {diff}")

if __name__ == "__main__":
    main()
