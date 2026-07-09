import numpy as np

hat_r1 = 0.6666667
hat_r2 = 1.3333333
m_val = 1
hat_c0sq = 0.0349
hat_omA2 = 0.0
gamma = 0.0

hat_r_arr = np.linspace(hat_r1, hat_r2, 100)
hat_kr = np.pi / (hat_r2 - hat_r1)

def wkb(hat_r, n):
    hat_kr = n * np.pi / (hat_r2 - hat_r1)
    hat_kth = m_val / hat_r
    hat_k2 = hat_kr**2 + hat_kth**2

    R = (1+gamma)*(2*hat_r - 1)/hat_r
    S = m_val*(1+gamma)*(hat_r - 1)/hat_r

    # Rossby limit:
    wR = - S / (4 + hat_c0sq*hat_k2 + R)
    return wR

wR_avg = np.mean([wkb(r, 1) for r in hat_r_arr])
print("wR average:", wR_avg)

# Old formula
hat_k2_old = hat_kr**2 + (m_val/1.0)**2
wR_old = - (1.0 + gamma) * m_val / (2.0 * (hat_k2_old + 4.0 / hat_c0sq))
print("wR old:", wR_old)
