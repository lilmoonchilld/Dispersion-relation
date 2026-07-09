import numpy as np

def identify_branches(roots, hat_omA2):
    # Sort roots
    roots = sorted(roots)

    # We expect 4 roots: MP_m, MS, R, MP_p
    # MP_m is the most negative. MP_p is the most positive.
    MP_m = roots[0]
    MP_p = roots[3]

    middle = [roots[1], roots[2]]

    # How to distinguish R and MS?
    # If omA2=0, MS is exactly 0.
    # What if omA2 > 0? MS approaches 0 for small k, but R is also slow.
    # In general, if we compute them analytically in the C=0 or B0=0 limit...

    # Since MS arises purely from B (as seen in the old script where oMS_hat = - hat_VA2 * hat_k2 / ...),
    # MS is the one that goes to 0 when omA2 -> 0.
    # Alternatively, we can just track them continuously or just keep them sorted by magnitude.
    # Let's say: R is the one that is non-zero when omA2=0.

    return MP_m, middle[0], middle[1], MP_p
