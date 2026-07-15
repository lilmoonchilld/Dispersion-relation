import numpy as np

hat_r1 = 0.5e7 / 0.75e7
hat_r2 = 1.0e7 / 0.75e7
gamma = 0.0
hat_c0sq = 9.81 * 500 / ( (0.5e-4)**2 * (0.75e7)**2 )
hat_omA2 = 0.0

def wkb_roots(m_val, n_radial):
    hat_kr = n_radial * np.pi / (hat_r2 - hat_r1)

    r_grid = np.linspace(hat_r1, hat_r2, 50)
    roots_grid = []

    for r in r_grid:
        hat_kth = m_val / r
        hat_k2 = hat_kr**2 + hat_kth**2

        R = (1+gamma)*(2*r - 1)/r
        S = m_val*(1+gamma)*(r - 1)/r

        A2 = 2*hat_omA2 - 1 - 0.25*(hat_c0sq*hat_k2 + R)
        A1 = -0.25 * S
        A0 = hat_omA2 * (hat_omA2 - 0.25*(hat_c0sq*hat_k2 + R))

        # solve X^4 + A2 X^2 + A1 X + A0 = 0
        coeffs = [1.0, 0.0, A2, A1, A0]
        rts = np.roots(coeffs)
        # sort roots by real part to track branches
        rts = sorted(rts, key=lambda x: x.real)
        roots_grid.append([x.real for x in rts])

    roots_grid = np.array(roots_grid)
    # average over r
    avg_roots = np.mean(roots_grid, axis=0)
    return avg_roots

print("m=1, n=1:", wkb_roots(1, 1))

# Compare to old logic
m_val=1; n_radial=1; r=1.0
hat_kr = n_radial * np.pi / (hat_r2 - hat_r1)
hat_kth = m_val / r
hat_k2 = hat_kr**2 + hat_kth**2
R = (1+gamma)
S = 0.0
A2 = 2*hat_omA2 - 1 - 0.25*(hat_c0sq*hat_k2 + R)
A1 = 0.0
A0 = hat_omA2 * (hat_omA2 - 0.25*(hat_c0sq*hat_k2 + R))
rts = np.roots([1.0, 0.0, A2, A1, A0])
print("Old logic r=1:", sorted([x.real for x in rts]))
