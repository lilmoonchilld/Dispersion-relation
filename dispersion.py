import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def cheb(N):
    """Chebyshev differentiation matrix."""
    x = np.cos(np.pi * np.arange(N) / (N - 1))
    c = np.ones(N)
    c[0] = 2; c[-1] = 2
    c = c * (-1)**np.arange(N)
    X = np.tile(x, (N, 1))
    dX = X - X.T
    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(N))
    D = D - np.diag(np.sum(D, axis=1))
    return x, D

def build_matrices(N, m, C, params):
    """
    Build the GEVP Ax = omega Bx.
    State vector: x = [eta, v_r, v_theta, w_r, w_theta]^T
    where w_r = omega v_r, w_theta = omega v_theta
    """
    r1, r2, r0, H0, Omega, B0, mu0, rho0, g_val = params
    x, D = cheb(N)
    r = (r2 - r1)/2 * x + (r2 + r1)/2
    D1 = (2 / (r2 - r1)) * D

    beta_g = 2 * C / (r0**3)
    beta_eff = Omega**2 + beta_g
    vA2 = B0**2 / (mu0 * rho0)
    omegaA2 = vA2 / H0**2

    A = np.zeros((5*N, 5*N), dtype=complex)
    B = np.zeros((5*N, 5*N), dtype=complex)
    I = np.eye(N)

    # Eq 1: Continuity
    A[0:N, N:2*N] = -H0 * (D1 + np.diag(1/r))
    A[0:N, 2*N:3*N] = -1j * m * H0 * np.diag(1/r)
    B[0:N, 0:N] = I

    # Eq 2: omega v_r = w_r
    A[N:2*N, 3*N:4*N] = I
    B[N:2*N, N:2*N] = I

    # Eq 3: omega v_theta = w_theta
    A[2*N:3*N, 4*N:5*N] = I
    B[2*N:3*N, 2*N:3*N] = I

    # Eq 4: r-mom
    A[3*N:4*N, N:2*N] = -omegaA2 * H0 * I
    A[3*N:4*N, 4*N:5*N] = 2j * Omega * H0 * I
    B[3*N:4*N, 3*N:4*N] = H0 * I
    B[3*N:4*N, 0:N] = 1j * g_val * H0 * D1 - 1j * beta_eff * np.diag(r - r0)

    # Eq 5: theta-mom
    A[4*N:5*N, 2*N:3*N] = -omegaA2 * H0 * I
    A[4*N:5*N, 3*N:4*N] = -2j * Omega * H0 * I
    B[4*N:5*N, 4*N:5*N] = H0 * I
    B[4*N:5*N, 0:N] = np.diag(-m * g_val * H0 / r)

    # Boundaries: v_r(r1) = 0, v_r(r2) = 0
    # Overwrite the rows corresponding to v_r at boundaries (indices N and 2N-1)
    # Actually, we can just replace the corresponding rows in A and B.
    # We want: A * X = 0 (which means v_r = 0) and B * X = 0
    # This yields omega * 0 = 0.

    A[N, :] = 0
    A[N, N] = 1
    B[N, :] = 0

    A[2*N-1, :] = 0
    A[2*N-1, 2*N-1] = 1
    B[2*N-1, :] = 0

    return A, B, r

def match_eigenvalues(vals1, vals2, tol=1e-6):
    """Find eigenvalues in vals1 that also appear in vals2 within tol."""
    matched = []
    for v1 in vals1:
        diffs = np.abs(vals2 - v1)
        if len(vals2) > 0 and np.min(diffs) / max(np.abs(v1), 1e-10) < tol:
            matched.append(v1)
    return np.array(matched)

def compute_spectrum(N, m, C, params, tol=1e-6):
    """Compute drift-filtered spectrum."""
    A1, B1, r1 = build_matrices(N, m, C, params)
    vals1 = linalg.eigvals(A1, B1)
    vals1 = vals1[np.isfinite(vals1)]

    A2, B2, r2 = build_matrices(N+2, m, C, params)
    vals2 = linalg.eigvals(A2, B2)
    vals2 = vals2[np.isfinite(vals2)]

    filtered_vals = match_eigenvalues(vals1, vals2, tol)
    return filtered_vals

def run_dispersion(C_val, label, m_max=10, N=60):
    params = (0.50, 0.65, 0.575, 0.02, 1.0, 0.3, 4*np.pi*1e-7, 6440, 9.81)
    Omega = params[4]

    m_vals = np.arange(1, m_max + 1)
    neutral_modes = []
    unstable_modes = []

    for m in m_vals:
        print(f"[{label}] Computing m = {m}...")
        vals = compute_spectrum(N, m, C_val, params)

        # separate neutral and unstable
        # typically neutral means Im(omega) < 1e-8
        for v in vals:
            if np.abs(np.imag(v)) < 1e-8:
                neutral_modes.append((m, np.real(v) / (2 * Omega)))
            elif np.imag(v) > 1e-8:
                unstable_modes.append((m, v / (2 * Omega)))

    neutral_modes = np.array(neutral_modes)
    unstable_modes = np.array(unstable_modes)

    return neutral_modes, unstable_modes

def plot_dispersion(neutral, unstable, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if len(neutral) > 0:
        ax1.scatter(neutral[:, 0], neutral[:, 1], color='blue', s=10)
    ax1.set_xlabel('Azimuthal wavenumber m')
    ax1.set_ylabel(r'Re($\omega$) / $2\Omega$')
    ax1.set_title('Neutral Dispersion')
    ax1.grid(True)

    if len(unstable) > 0:
        ax2.scatter(unstable[:, 0], np.imag(unstable[:, 1]), color='red', s=10)
    ax2.set_xlabel('Azimuthal wavenumber m')
    ax2.set_ylabel(r'Im($\omega$) / $2\Omega$')
    ax2.set_title('Instability Growth Rates')
    ax2.grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_eigenfunction(N, m, C, omega_target, params, filename):
    A, B, r = build_matrices(N, m, C, params)
    vals, vecs = linalg.eig(A, B)

    # find mode closest to omega_target
    idx = np.argmin(np.abs(vals - omega_target))
    mode = vecs[:, idx]

    eta = mode[0:N]
    vr = mode[N:2*N]
    vtheta = mode[2*N:3*N]

    # Normalize by max of eta
    norm = np.max(np.abs(eta))
    eta = eta / norm
    vr = vr / norm
    vtheta = vtheta / norm

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, np.real(eta), label=r'Re($\tilde{\eta}$)', color='blue')
    ax.plot(r, np.imag(eta), '--', label=r'Im($\tilde{\eta}$)', color='blue')

    ax.plot(r, np.real(vr), label=r'Re($\tilde{v}_r$)', color='red')
    ax.plot(r, np.imag(vr), '--', label=r'Im($\tilde{v}_r$)', color='red')

    ax.plot(r, np.real(vtheta), label=r'Re($\tilde{v}_\theta$)', color='green')
    ax.plot(r, np.imag(vtheta), '--', label=r'Im($\tilde{v}_\theta$)', color='green')

    ax.set_xlabel('Radius r (m)')
    ax.set_title(f"Eigenfunction for m={m}, $\\omega \\approx$ {vals[idx]:.4f}")
    ax.legend()
    ax.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    params = (0.50, 0.65, 0.575, 0.02, 1.0, 0.3, 4*np.pi*1e-7, 6440, 9.81)
    r0 = params[2]
    Omega = params[4]

    print("Running C=0 baseline...")
    n0, u0 = run_dispersion(0.0, "C=0", m_max=5)
    plot_dispersion(n0, u0, "Dispersion Relation (C=0)", "dispersion_C0.png")

    C_test = Omega**2 * r0**3
    print(f"Running C={C_test:.4f} test case...")
    n1, u1 = run_dispersion(C_test, "C=\\Omega^2 r0^3", m_max=5)
    plot_dispersion(n1, u1, "Dispersion Relation (C=$\\Omega^2 r_0^3$)", "dispersion_C_test.png")

    print("Plotting sample eigenfunction for C=0...")
    if len(n0) > 0:
        # Just pick the first matched neutral mode for m=1
        m_sample = int(n0[0, 0])
        omega_sample = n0[0, 1] * 2 * Omega
        plot_eigenfunction(60, m_sample, 0.0, omega_sample, params, "eigenfunction_sample.png")
    else:
        print("No neutral modes found to plot.")
