import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def create_fdm_matrix(N, dr):
    """
    Creates a 2nd-order central finite difference matrix for first derivative.
    Forward/backward difference used at boundaries.
    """
    D = np.zeros((N, N))

    # Interior points (central difference)
    for i in range(1, N - 1):
        D[i, i - 1] = -1.0 / (2.0 * dr)
        D[i, i + 1] = 1.0 / (2.0 * dr)

    # Left boundary (forward difference)
    D[0, 0] = -3.0 / (2.0 * dr)
    D[0, 1] = 4.0 / (2.0 * dr)
    D[0, 2] = -1.0 / (2.0 * dr)

    # Right boundary (backward difference)
    D[N-1, N-3] = 1.0 / (2.0 * dr)
    D[N-1, N-2] = -4.0 / (2.0 * dr)
    D[N-1, N-1] = 3.0 / (2.0 * dr)

    return D

def solve_swmhd_fdm(m, N=200):
    # Constants
    G = 6.67430e-11
    M = 5.683e26
    r0 = 100000000.0
    H0 = 10.0
    rho = 100.0
    mu0 = 4 * np.pi * 1e-7
    B0 = 2e-5

    r1 = 80000000.0
    r2 = 120000000.0

    cg2 = G * M * H0**2 / r0**3
    Omega_mean = np.sqrt(G * M / r0**3)

    # Uniform FDM grid
    r = np.linspace(r1, r2, N)
    dr = r[1] - r[0]
    D = create_fdm_matrix(N, dr)

    # Background profiles
    Omega = np.sqrt(G * M / r**3)
    H = H0 * (r / r0)**1.5

    # Create block matrices
    # X = [eta, Vr, Vtheta, Br, Btheta]
    # Matrix A size: 5*N x 5*N
    n_vars = 5
    A = np.zeros((n_vars*N, n_vars*N), dtype=np.complex128)

    I = np.eye(N)

    # Block indices
    idx_eta = slice(0, N)
    idx_Vr = slice(N, 2*N)
    idx_Vtheta = slice(2*N, 3*N)
    idx_Br = slice(3*N, 4*N)
    idx_Btheta = slice(4*N, 5*N)

    # Eq 1: Continuity
    # omega eta = -i dVr/dr - i (5/(2r)) Vr + (m/r) Vtheta
    A[idx_eta, idx_Vr] = -1j * D - 1j * np.diag(5.0 / (2.0 * r))
    A[idx_eta, idx_Vtheta] = np.diag(m / r)

    # Eq 2: r-momentum
    # omega Vr = -i (cg2/H) deta/dr + i 2 Omega Vtheta + i B0/(mu0 rho H) Br
    A[idx_Vr, idx_eta] = -1j * np.diag(cg2 / H) @ D
    A[idx_Vr, idx_Vtheta] = 1j * np.diag(2.0 * Omega)
    A[idx_Vr, idx_Br] = 1j * np.diag(B0 / (mu0 * rho * H))

    # Eq 3: theta-momentum
    # omega Vtheta = (m cg2 / (r H)) eta - i 2 Omega Vr + i B0/(mu0 rho H) Btheta
    A[idx_Vtheta, idx_eta] = np.diag(m * cg2 / (r * H))
    A[idx_Vtheta, idx_Vr] = -1j * np.diag(2.0 * Omega)
    A[idx_Vtheta, idx_Btheta] = 1j * np.diag(B0 / (mu0 * rho * H))

    # Eq 4: r-induction
    # omega Br = -i (B0/H) Vr
    A[idx_Br, idx_Vr] = -1j * np.diag(B0 / H)

    # Eq 5: theta-induction
    # omega Btheta = -i (B0/H) Vtheta
    A[idx_Btheta, idx_Vtheta] = -1j * np.diag(B0 / H)

    # Boundary conditions
    # Vr(r1) = Vr(r2) = 0
    # Modifying the rows corresponding to Vr at boundaries
    # boundary at x=r1 -> index 0
    # boundary at x=r2 -> index N-1

    # Clear equations for Vr at boundaries
    A[N + 0, :] = 0
    A[N + N - 1, :] = 0

    # We actually need a generalized eigenvalue problem A X = lambda C X
    # where C is Identity except at boundary conditions where it is 0
    C = np.eye(n_vars*N, dtype=np.complex128)

    # Set C to 0 for the boundary rows
    C[N + 0, N + 0] = 0
    C[N + N - 1, N + N - 1] = 0

    # Apply BCs in A: A * X = 0 for Vr at boundaries
    A[N + 0, N + 0] = 1.0  # Vr(r=r1) = 0
    A[N + N - 1, N + N - 1] = 1.0  # Vr(r=r2) = 0

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eig(A, C)

    # Filter out infinite or NaN eigenvalues (caused by C having zero rows)
    valid = np.isfinite(eigvals)
    eigvals = eigvals[valid]

    return eigvals, Omega_mean

def main():
    ms = [1, 2, 3, 4, 5]
    all_eigvals = []

    for m in ms:
        print(f"Solving for m={m} using Uniform FDM...")
        # Increase N slightly for uniform grid to maintain accuracy
        eigvals, Omega_mean = solve_swmhd_fdm(m, N=150)
        # normalize
        eigvals_norm = eigvals / (2 * Omega_mean)
        all_eigvals.append(eigvals_norm)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(ms):
        vals = all_eigvals[i]
        # Keep mainly real parts for physical oscillating modes
        vals_real = np.real(vals[np.abs(np.imag(vals)) < 1e-4 * np.abs(np.real(vals)) + 1e-8])
        plt.scatter([m]*len(vals_real), vals_real, color='b', alpha=0.5, s=10)

    plt.xlabel('Azimuthal Mode Number $m$')
    plt.ylabel(r'Normalized Frequency $\omega / (2\Omega_0)$')
    plt.title('Dispersion Relation of Annular SWMHD (Uniform FDM)')
    plt.grid(True)
    plt.savefig('dispersion_relation.png')
    print("Plot saved to dispersion_relation.png")

if __name__ == "__main__":
    main()
