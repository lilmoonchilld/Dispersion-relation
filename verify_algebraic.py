import numpy as np
import sympy as sp

def main():
    # Let's verify symbolically by substituting random values or checking expanded polynomials rather than full sp.simplify on huge expressions.
    # 1. Symbolic verification
    omega, g_sym, H_sym, Omega_sym, f_sym, B0_sym, rho0_sym, mu0_sym, C_sym, r1_sym, r2_sym, r0_sym, r_sym, n_sym, m_sym, kappa_sq_sym = sp.symbols(
        'omega g H0 Omega f B0 rho0 mu0 C r1 r2 r0 r n m kappa_sq', complex=True
    )

    # Equilibrium quantity expressions
    Heq = H_sym + (Omega_sym**2 / (2*g_sym)) * (r_sym**2 - r0_sym**2) + (C_sym / g_sym) * (1/r_sym - 1/r0_sym)
    Q = Omega_sym**2 * r_sym - C_sym / r_sym**2
    omega_A_sq = B0_sym**2 / (mu0_sym * rho0_sym * Heq**2)
    omega_star = omega + omega_A_sq / omega

    # Direct formula contributions
    W1 = omega * (omega_star**2 - f_sym**2) / (g_sym * Heq * omega_star)
    W2 = m_sym * f_sym * Q / (r_sym * g_sym * Heq * omega_star)
    W3 = 4 * m_sym * f_sym * B0_sym**2 * Q / (r_sym * mu0_sym * rho0_sym * omega * g_sym * Heq**3 * (omega_star**2 - f_sym**2))

    kappa_sq_direct = W1 + W2 + W3

    # Polynomial expressions
    N_expr = omega**4 + (2*omega_A_sq - f_sym**2)*omega**2 + omega_A_sq**2
    D6_expr = (omega**2 + omega_A_sq) * N_expr
    P8_expr = N_expr**2 + (m_sym * f_sym * Q / r_sym) * omega * (N_expr + 4*omega_A_sq*(omega**2 + omega_A_sq))

    kappa_sq_poly = P8_expr / (g_sym * Heq * D6_expr)

    # Instead of full sp.simplify on difference of rational functions (which takes forever), we can put direct on common denominator:
    # W1 + W2 + W3 = ( W1_num + W2_num ) / (g_sym * Heq * omega_star) + W3_num / (g_sym * Heq * omega_star * (omega_star^2 - f^2))
    # which has a common denominator of g_sym * Heq * omega_star * (omega_star^2 - f^2) = g_sym * Heq * D6_expr / omega^2
    # Let's check numerical equivalence at randomly chosen values to confirm algebraic identity.
    # We will also expand and verify that the coefficient definitions match.

    # Let's expand P_8(x) - kappa_sq * g * Heq * D_6(x)
    poly_lhs = P8_expr - kappa_sq_sym * g_sym * Heq * D6_expr
    poly_lhs_expanded = sp.expand(poly_lhs)

    # Collect coefficients in omega (since omega_A_sq contains Heq etc., let's treat omega_A_sq, f, g, Heq, mfQ/r as simple symbols for manual verification to avoid heavy expansion)
    oA_s, f_s, gHeq_s, mfQ_r_s, kap_gHeq_s = sp.symbols('omega_A_sq f gHeq mfQ_r kap_gHeq', complex=True)

    N_simple = omega**4 + (2*oA_s - f_s**2)*omega**2 + oA_s**2
    D6_simple = (omega**2 + oA_s) * N_simple
    P8_simple = N_simple**2 + mfQ_r_s * omega * (N_simple + 4*oA_s*(omega**2 + oA_s))
    poly_simple = P8_simple - kap_gHeq_s * D6_simple
    poly_simple_exp = sp.expand(poly_simple)

    coeffs_simple = [sp.simplify(poly_simple_exp.coeff(omega, i)) for i in range(9)]

    # Manual coefficients with simplified symbols
    A8 = sp.Integer(1)
    A7 = sp.Integer(0)
    A6 = 4*oA_s - 2*f_s**2 - kap_gHeq_s
    A5 = mfQ_r_s
    A4 = 6*oA_s**2 - 4*f_s**2 * oA_s + f_s**4 - kap_gHeq_s * (3*oA_s - f_s**2)
    A3 = mfQ_r_s * (6*oA_s - f_s**2)
    A2 = 4*oA_s**3 - 2*f_s**2 * oA_s**2 - kap_gHeq_s * (3*oA_s**2 - f_s**2 * oA_s)
    A1 = 5 * mfQ_r_s * oA_s**2
    A0 = oA_s**4 - kap_gHeq_s * oA_s**3

    manual_coeffs = [A0, A1, A2, A3, A4, A5, A6, A7, A8]

    print("Verifying manual coefficients against simplified symbolic collected coefficients:")
    matching = True
    for i in range(9):
        diff_coeff = sp.simplify(coeffs_simple[i] - manual_coeffs[i])
        print(f"Coefficient A_{i} difference: {diff_coeff}")
        if diff_coeff != 0:
            matching = False

    if matching:
        print("Verification SUCCESS: Manual coefficients perfectly match the algebraic derivation!")
    else:
        print("Verification FAILURE: Discrepancy found in coefficients.")

    # 2. Numerical verification
    # Define parameters
    Omega = 0.5e-4
    H0 = 500.0
    g = 9.81
    B0 = 5e-4
    rho0 = 1000.0
    C = 0.0
    r1 = 0.5e6
    r2 = 1.0e6
    r0 = 0.5 * (r1 + r2)
    f_val = 2.0 * Omega
    mu0 = 4.0 * np.pi * 1e-7
    n = 2
    m = 10  # use arbitrary m for testing

    # Evaluate quantities at r = r0
    kr = n * np.pi / (r2 - r1)
    kappa_sq_val = kr**2 + m**2 / r0**2

    # We substitute these values into the expressions to check direct vs polynomial kappa^2
    subs_dict = {
        g_sym: g,
        H_sym: H0,
        Omega_sym: Omega,
        f_sym: f_val,
        B0_sym: B0,
        rho0_sym: rho0,
        mu0_sym: mu0,
        C_sym: C,
        r1_sym: r1,
        r2_sym: r2,
        r0_sym: r0,
        r_sym: r0,
        n_sym: n,
        m_sym: m,
        kappa_sq_sym: kappa_sq_val
    }

    print("\nEvaluating numerical verification at r = r0 with m = 10:")
    trial_omegas = [
        (0.5 + 0.2j) * Omega,
        (1.0 + 0.3j) * Omega,
        (2.0 - 0.5j) * Omega
    ]

    for idx, omega_val in enumerate(trial_omegas, 1):
        # We can evaluate direct and polynomial kappa^2 at these omega values
        val_direct = complex(kappa_sq_direct.subs(subs_dict).subs(omega, omega_val))
        val_poly = complex(kappa_sq_poly.subs(subs_dict).subs(omega, omega_val))

        # Relative error
        num = abs(val_direct - val_poly)
        denom = max(abs(val_direct), abs(val_poly), 1e-16)
        rel_err = num / denom

        print(f"Trial omega_{idx} = {omega_val}:")
        print(f"  kappa_direct = {val_direct}")
        print(f"  kappa_poly   = {val_poly}")
        print(f"  Relative error = {rel_err}")

if __name__ == "__main__":
    main()
