import mpmath
import numpy as np
import matplotlib.pyplot as plt
import plot_dispersion as pd
import ast

def trace_roots():
    # Read m=2 roots to start tracing
    initial_roots = []
    current_m = None
    with open('roots_m1_m2.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('m='):
                current_m = int(line.split('=')[1])
            else:
                if current_m == 2:
                    parts = line.split(',')
                    initial_roots.append(complex(float(parts[0]), float(parts[1])))

    # Store trajectories
    trajectories = {1: [], 2: [], 3: []} # store up to 3 root branches

    # Let's read m=1 roots too for plot
    m1_roots = []
    current_m = None
    with open('roots_m1_m2.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('m='):
                current_m = int(line.split('=')[1])
            else:
                if current_m == 1:
                    parts = line.split(',')
                    m1_roots.append(complex(float(parts[0]), float(parts[1])))

    # Fill in m=1
    m_vals = list(range(1, 11))
    all_roots = {1: m1_roots, 2: initial_roots}

    print(f"Starting trace with m=2 roots: {initial_roots}")

    prev_roots = initial_roots
    for m in range(3, 11):
        print(f"\nTracing roots for m={m}...")
        current_roots = []
        for guess in prev_roots:
            try:
                root = mpmath.findroot(lambda w: pd.D_determinant(w, m),
                                       guess, solver='muller', tol=1e-8, maxsteps=50)
                if not mpmath.isnan(root.real) and not mpmath.isnan(root.imag):
                    val = abs(pd.D_determinant(root, m))
                    if val < 1e-2:
                        current_roots.append(complex(root))
                        print(f"Found root for m={m}: {root} (D={val:.2e})")
                    else:
                        print(f"Root found for m={m} but D value large: {val:.2e}")
            except Exception as e:
                print(f"Failed to find root for guess {guess}: {e}")

        # Simple heuristic to sort roots to match branches across m
        # We assume the roots don't jump too far
        if len(current_roots) > 0 and len(prev_roots) > 0:
            sorted_current = []
            for pr in prev_roots:
                # Find closest current root
                if len(current_roots) > 0:
                    dists = [abs(cr - pr) for cr in current_roots]
                    idx = np.argmin(dists)
                    sorted_current.append(current_roots.pop(idx))

            # append remaining if any
            sorted_current.extend(current_roots)
            prev_roots = sorted_current
            all_roots[m] = sorted_current
        else:
            all_roots[m] = []

    # Plot dispersion curve Re(omega)/(2*Omega) vs m
    plt.figure(figsize=(10, 6))

    Omega = pd.Omega

    # We want to plot continuous lines for each branch
    # Let's collect points per branch
    branch_points = {} # index: list of (m, val)

    # Assign branch indices based on m=2 sorting
    for m in range(2, 11):
        if m in all_roots:
            roots = all_roots[m]
            for i, r in enumerate(roots):
                if i not in branch_points:
                    branch_points[i] = []
                # Plot Re(omega) / (2*Omega)
                norm_omega = r.real / (2 * Omega)
                branch_points[i].append((m, norm_omega))

    # Also plot m=1 points
    for r in m1_roots:
        norm_omega = r.real / (2 * Omega)
        plt.plot(1, norm_omega, 'ko', markersize=4)

    for i, points in branch_points.items():
        if len(points) > 0:
            ms, vals = zip(*points)
            plt.plot(ms, vals, 'o-', label=f'Branch {i+1}')

    plt.xlabel('Azimuthal Wavenumber (m)')
    plt.ylabel(r'$\mathrm{Re}(\omega) / 2\Omega$')
    plt.title('Dispersion Relation')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig('dispersion_curve.png', dpi=300)
    print("\nSaved dispersion curve to dispersion_curve.png")

if __name__ == "__main__":
    trace_roots()
