import numpy as np
import scipy.integrate
import scipy.linalg
import matplotlib.pyplot as plt
import time

# Parameters
G = 6.67430e-11
M = 5.683e26
R_saturn = 60268000.0
r0 = 100000000.0
H0 = 10.0
rho = 100.0
mu0 = 4 * np.pi * 1e-7

r1 = 7.4e7
r2 = 1.37e8
N = 128

def cheb_nodes(N, r1, r2):
    i = np.arange(N)
    x = np.cos(np.pi * i / (N - 1))
    r = 0.5 * (r2 - r1) * x + 0.5 * (r1 + r2)
    return r, x

def cheb_diff_matrix(N, r1, r2):
    x = np.cos(np.pi * np.arange(N) / (N - 1))
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0

    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1)**(i + j)) / (x[i] - x[j])
            elif i == j and i != 0 and i != N - 1:
                D[i, j] = -x[i] / (2.0 * (1.0 - x[i]**2))

    D[0, 0] = (2.0 * (N - 1)**2 + 1.0) / 6.0
    D[-1, -1] = -(2.0 * (N - 1)**2 + 1.0) / 6.0

    D_r = D * (2.0 / (r2 - r1))
    return D_r

def clenshaw_curtis_weights(N):
    n = N - 1
    w = np.zeros(N)
    for j in range(N):
        theta = j * np.pi / n
        s = 0.0
        for k in range(2, n, 2):
            s += 2.0 * np.cos(k * theta) / (1.0 - k**2)
        if n % 2 == 0:
            s += np.cos(n * theta) / (1.0 - n**2)
        w[j] = 2.0 / n * (1.0 + s)
    w[0] /= 2.0
    w[-1] /= 2.0
    return w

def get_base_profiles(r):
    Omega = np.sqrt(G * M / r**3)
    H = H0 * (r / r0)**1.5
    B0 = 2e-5 * (R_saturn / r)**3
    return Omega, H, B0

r, x = cheb_nodes(N, r1, r2)
D_r = cheb_diff_matrix(N, r1, r2)
w_x = clenshaw_curtis_weights(N)
w_r = w_x * 0.5 * (r2 - r1)

Omega, H, B0_prof = get_base_profiles(r)

def build_K_mat(r, m, H, N, G, rho, w_r):
    K_mat = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            rmax = max(r[i], r[j])
            xi = r[i] / rmax
            xj = r[j] / rmax
            e_ij = max(H[i], H[j]) / rmax

            def integrand(psi):
                return np.cos(m * psi) / np.sqrt(xi**2 + xj**2 - 2*xi*xj*np.cos(psi) + e_ij**2)

            # Use smaller epsabs/epsrel for the sharply peaked integrands
            if abs(xi - xj) < 1e-4:
                # We can compute it much faster by splitting the interval at e_ij or just using fewer subdivisions
                # since we only need rough precision for testing
                val1, _ = scipy.integrate.quad(integrand, 0, 10*e_ij, limit=50, epsabs=1e-3, epsrel=1e-3)
                val2, _ = scipy.integrate.quad(integrand, 10*e_ij, np.pi, limit=50, epsabs=1e-3, epsrel=1e-3)
                val = val1 + val2
            else:
                val, _ = scipy.integrate.quad(integrand, 0, np.pi, limit=50, epsabs=1e-4, epsrel=1e-4)

            # The definition of \phi is: - \pi G \int \rho \eta(r') K_m(r,r') r' dr'
            # Wait, the LaTeX says: \tilde{\phi}(r) = -\pi G \int \rho \tilde{\eta}(r') K_m(r,r') r' dr'
            # and K_m = (1/max(r,r')) b_{1/2}^{(m)}(min(r,r')/max(r,r'))
            # b_{1/2}^{(m)} = 2/\pi \int_0^\pi ... d\psi
            # So \tilde{\phi}(r) = -2 G \rho \int ...
            # Wait, the integral integrand(psi) is exactly \int_0^\pi d\psi / sqrt(xi^2 + xj^2 - 2xi xj cos psi + e_ij^2)
            # wait, if we use xi and xj as r_i / rmax and r_j / rmax
            # then xi^2 + xj^2 - 2 xi xj cos psi = (r_i^2 + r_j^2 - 2 r_i r_j cos psi) / rmax^2
            # so \int d\psi / sqrt(...) = rmax * \int d\psi / sqrt(r_i^2 + r_j^2 - 2 r_i r_j cos psi + rmax^2 e_ij^2)
            # Since \tilde{\phi} = -G \int \rho \tilde{\eta} r' d\theta' dr' / sqrt(...)
            # \tilde{\phi} = -2 G \rho \int_0^\pi cos(m psi) dpsi / sqrt(...) r_j dr_j
            # So the integral_val is exactly \int_0^\pi cos(m psi) / sqrt(r_i^2 + r_j^2 - 2 r_i r_j cos psi + eps^2) dpsi
            # which is (1/rmax) \int_0^\pi cos(m psi) / sqrt(xi^2 + xj^2 - 2xi xj cos psi + e_ij^2) dpsi

            # Yes! integral_val = val / rmax
            integral_val = val / rmax

            # And K_mat = -2 G rho r_j w_r_j * integral_val
            # Wait, from LaTeX: \tilde{\phi}(r) = - \pi G \int \rho \tilde{\eta}(r') K_m(r,r') r' dr'
            # K_m = (1/rmax) b_{1/2}^{(m)} = (1/rmax) (2/\pi) \int_0^\pi cos(m psi) / sqrt(...) dpsi
            # So -\pi G \rho r' \tilde{\eta} (1/rmax) (2/\pi) \int ...
            # = -2 G \rho r' \tilde{\eta} (1/rmax) \int ...

            K_mat[i, j] = -2.0 * G * rho * r[j] * w_r[j] * integral_val
            if i != j:
                K_mat[j, i] = -2.0 * G * rho * r[i] * w_r[i] * integral_val
    return K_mat

# Equations:
# 1. Continuity: -i omega eta + H dVr/dr + 5H/(2r) Vr + i m H/r Vtheta = 0
#    => omega eta = -i H dVr/dr - i 5H/(2r) Vr + m H/r Vtheta
# 2. r-Momentum: -i omega Vr + GMH/r^3 deta/dr - 2 Omega Vtheta - B0/(mu0 rho H) Br + dphi/dr = 0
#    => omega Vr = -i GMH/r^3 deta/dr + i 2 Omega Vtheta + i B0/(mu0 rho H) Br - i dphi/dr
# 3. theta-Momentum: -i omega Vtheta + i m GMH/r^4 eta + 2 Omega Vr - B0/(mu0 rho H) Btheta + i m/r phi = 0
#    => omega Vtheta = m GMH/r^4 eta - i 2 Omega Vr + i B0/(mu0 rho H) Btheta + m/r phi
# 4. Induction-r: -i omega Br + B0/H Vr = 0
#    => omega Br = -i B0/H Vr
# 5. Induction-theta: -i omega Btheta + B0/H Vtheta = 0
#    => omega Btheta = -i B0/H Vtheta

def build_system(m, N, r, H, Omega, B0_prof, D_r, K_mat):
    A = np.zeros((5*N, 5*N), dtype=np.complex128)
    B = np.eye(5*N, dtype=np.complex128)

    # State: 0:N is eta, N:2N is Vr, 2N:3N is Vtheta, 3N:4N is Br, 4N:5N is Btheta
    eta_idx = slice(0, N)
    Vr_idx = slice(N, 2*N)
    Vt_idx = slice(2*N, 3*N)
    Br_idx = slice(3*N, 4*N)
    Bt_idx = slice(4*N, 5*N)

    # K_r is D_r @ K_mat
    K_r = D_r @ K_mat

    for i in range(N):
        # 1. Continuity
        # omega eta = -i H dVr/dr - i 5H/(2r) Vr + m H/r Vtheta
        A[eta_idx.start + i, Vr_idx] = -1j * H[i] * D_r[i, :]
        A[eta_idx.start + i, Vr_idx.start + i] += -1j * 5.0 * H[i] / (2.0 * r[i])
        A[eta_idx.start + i, Vt_idx.start + i] += m * H[i] / r[i]

        # 2. r-Momentum
        # omega Vr = -i GMH/r^3 deta/dr + i 2 Omega Vtheta + i B0/(mu0 rho H) Br - i dphi/dr
        A[Vr_idx.start + i, eta_idx] = -1j * G * M * H[i] / (r[i]**3) * D_r[i, :] - 1j * K_r[i, :]
        A[Vr_idx.start + i, Vt_idx.start + i] = 1j * 2.0 * Omega[i]
        A[Vr_idx.start + i, Br_idx.start + i] = 1j * B0_prof[i] / (mu0 * rho * H[i])

        # 3. theta-Momentum
        # omega Vtheta = m GMH/r^4 eta - i 2 Omega Vr + i B0/(mu0 rho H) Btheta + m/r phi
        A[Vt_idx.start + i, eta_idx] = m * G * M * H[i] / (r[i]**4) * np.eye(N)[i, :] + m / r[i] * K_mat[i, :]
        A[Vt_idx.start + i, Vr_idx.start + i] = -1j * 2.0 * Omega[i]
        A[Vt_idx.start + i, Bt_idx.start + i] = 1j * B0_prof[i] / (mu0 * rho * H[i])

        # 4. Induction-r
        # omega Br = -i B0/H Vr
        A[Br_idx.start + i, Vr_idx.start + i] = -1j * B0_prof[i] / H[i]

        # 5. Induction-theta
        # omega Btheta = -i B0/H Vtheta
        A[Bt_idx.start + i, Vt_idx.start + i] = -1j * B0_prof[i] / H[i]

    # Boundary Conditions
    # Vr(r1) = 0 (which is i=0 or i=N-1 depending on cheb nodes, r[0]=r2, r[-1]=r1)

    # r[0] = r2, r[-1] = r1
    # Replace rows for Vr(r[0]) and Vr(r[-1])
    # i=0
    A[Vr_idx.start + 0, :] = 0
    A[Vr_idx.start + 0, Vr_idx.start + 0] = 1.0
    B[Vr_idx.start + 0, :] = 0

    # i=N-1
    A[Vr_idx.start + N - 1, :] = 0
    A[Vr_idx.start + N - 1, Vr_idx.start + N - 1] = 1.0
    B[Vr_idx.start + N - 1, :] = 0

    return A, B

m_vals = np.arange(1, 21)
max_imag_omega = []
all_omegas = []

most_unstable_m = None
most_unstable_omega = -1
most_unstable_vec = None

for m in m_vals:
    print(f"Solving for m = {m}...", flush=True)
    t0 = time.time()
    K_mat = build_K_mat(r, m, H, N, G, rho, w_r)
    A, B = build_system(m, N, r, H, Omega, B0_prof, D_r, K_mat)

    # Solve generalized eigenvalue problem A x = omega B x
    eigvals, eigvecs = scipy.linalg.eig(A, B)

    # Filter out infinite/nan eigenvalues from the algebraic constraints
    valid_mask = np.isfinite(eigvals)
    eigvals = eigvals[valid_mask]
    eigvecs = eigvecs[:, valid_mask]

    all_omegas.append(eigvals)

    # Find most unstable mode
    imag_parts = np.imag(eigvals)
    max_im = np.max(imag_parts)
    max_imag_omega.append(max_im)

    if max_im > most_unstable_omega:
        most_unstable_omega = max_im
        most_unstable_m = m
        most_unstable_vec = eigvecs[:, np.argmax(imag_parts)]

    print(f"  Max Im(omega): {max_im:.4e}, time: {time.time() - t0:.2f} s", flush=True)

# Plot global dispersion
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
for i, m in enumerate(m_vals):
    ax1.plot([m]*len(all_omegas[i]), np.real(all_omegas[i]), 'k.', markersize=2, alpha=0.3)
ax1.set_ylabel('Re($\\\omega$)')
ax1.set_xlabel('Azimuthal wavenumber $m$')
ax1.set_title('Global Dispersion Relation (Real part)')

ax2.plot(m_vals, max_imag_omega, 'r-o')
ax2.set_ylabel('Max Im($\\\omega$) [Growth Rate]')
ax2.set_xlabel('Azimuthal wavenumber $m$')
ax2.set_title('Global Dispersion Relation (Imaginary part)')
ax2.grid(True)
plt.tight_layout()
plt.savefig('global_dispersion.png', dpi=300)
plt.close()

# Plot most unstable mode
if most_unstable_vec is not None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    labels = [r'$\eta$', r'$V_r$', r'$V_\theta$', r'$B_r$', r'$B_\theta$']

    for i in range(5):
        vec_part = most_unstable_vec[i*N:(i+1)*N]
        axes[i].plot(r, np.real(vec_part), label='Real')
        axes[i].plot(r, np.imag(vec_part), label='Imag')
        axes[i].plot(r, np.abs(vec_part), 'k--', label='Magnitude')
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Radius $r$ [m]')
    axes[0].set_title(f'Eigenmode Structure (Most Unstable Mode: m={most_unstable_m}, $\\omega$={most_unstable_omega:.4e})')
    plt.tight_layout()
    plt.savefig('eigenmode_structure.png', dpi=300)
    plt.close()

print("Done!")
