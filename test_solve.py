import numpy as np

def solve(hat_r, m_val, n, hat_r1, hat_r2, gamma, hat_c0sq, hat_omA2):
    hat_kr = n * np.pi / (hat_r2 - hat_r1)
    hat_kth = m_val / hat_r
    hat_k2 = hat_kr**2 + hat_kth**2

    R = (1+gamma)*(2*hat_r - 1)/hat_r
    S = m_val*(1+gamma)*(hat_r - 1)/hat_r

    A2 = 2*hat_omA2 - 1 - 0.25*(hat_c0sq*hat_k2 + R)
    A1 = -0.25 * S
    A0 = hat_omA2 * (hat_omA2 - 0.25*(hat_c0sq*hat_k2 + R))

    roots = np.roots([1.0, 0.0, A2, A1, A0])

    # Sort roots: MP_m, MS, R, MP_p?
    # Actually, if omA2=0, A0=0.
    # Roots: X^4 + A2 X^2 + A1 X = 0 -> X(X^3 + A2 X + A1) = 0. One is exactly 0. (Magnetostrophic)
    # The others are Poincare +/-, and Rossby.
    roots = sorted([r.real for r in roots])
    # The smallest negative is MP_m. The largest positive is MP_p.
    # The two in the middle are Rossby and Magnetostrophic.
    # For omA2=0, MS = 0 identically. R is small negative or positive depending on S.
    return roots

print(solve(0.8, 1, 1, 0.666, 1.333, 0.0, 0.0349, 0.0))
print(solve(1.2, 1, 1, 0.666, 1.333, 0.0, 0.0349, 0.0))
