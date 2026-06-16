import numpy as np
import matplotlib.pyplot as plt

def analyze_modes(vals_real, m):
    """
    Analyzes normalized frequencies (omega / 2*Omega_0) to identify wave types.
    """
    # Magnitudes of normalized frequencies
    mags = np.abs(vals_real)

    # Classification thresholds (approximate)
    # Poincare (gravity) modes: high frequency, |omega| > ~1
    poincare = vals_real[mags > 1.1]

    # Rossby/Kelvin/Rossby-Gravity modes: mid/low frequency
    # Kelvin waves are usually prograde (positive omega in this convention if m>0),
    # Rossby waves are strictly retrograde (negative omega in rotating frame)
    # Rossby typically |omega| < 1
    rossby = vals_real[(vals_real < -0.05) & (vals_real > -1.0)]
    kelvin_or_prograde = vals_real[(vals_real > 0.05) & (vals_real < 1.0)]

    # Magnetostrophic / Slow MHD modes: very low frequency, |omega| ~ 0
    # often scales with Alfven speed
    slow_magnetic = vals_real[mags < 0.05]

    print(f"\n--- Analysis for m = {m} ---")
    print(f"Total physical modes detected: {len(vals_real)}")
    print(f"Poincare (Fast Gravity) Modes (|omega/2Omega| > 1): {len(poincare)}")
    print(f"Rossby (Retrograde) Modes (-1 < omega/2Omega < -0.05): {len(rossby)}")
    print(f"Kelvin/Prograde Mixed Modes (0.05 < omega/2Omega < 1): {len(kelvin_or_prograde)}")
    print(f"Magnetostrophic/Slow MHD Modes (|omega/2Omega| < 0.05): {len(slow_magnetic)}")

def main():
    ms = [1, 2, 3, 4, 5]
    from swmhd_solver import solve_swmhd_fdm

    for m in ms:
        eigvals, Omega_mean = solve_swmhd_fdm(m, N=150)
        eigvals_norm = eigvals / (2 * Omega_mean)

        # Keep mainly real parts for physical oscillating modes
        vals_real = np.real(eigvals_norm[np.abs(np.imag(eigvals_norm)) < 1e-4 * np.abs(np.real(eigvals_norm)) + 1e-8])
        analyze_modes(vals_real, m)

if __name__ == "__main__":
    main()
