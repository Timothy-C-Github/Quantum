#!/usr/bin/env python3
"""
Hydrogen atom - relativistic solution to the Dirac equation.

The bound electron is described by a four-component spinor built from a large
radial component g(r), a small radial component f(r), and the spin-angular
functions (spinor spherical harmonics). The observable plotted is the spinor
probability density

    |psi|^2 = |psi_1|^2 + |psi_2|^2 + |psi_3|^2 + |psi_4|^2

Because the four components are built from orthogonal angular functions, the
density separates into a large- and small-component part:

    |psi(r, theta, phi)|^2 =
        |g(r)/r|^2 * ( |c1|^2 |Y_k^(m-1/2)|^2 + |c2|^2 |Y_k^(m+1/2)|^2 )
      + |f(r)/r|^2 * ( |c3|^2 |Y_{k'}^(m-1/2)|^2 + |c4|^2 |Y_{k'}^(m+1/2)|^2 )

This is evaluated over a full 3D grid to show the spatial probability cloud and
2D cross-sections, together with the radial density |g(r)|^2 + |f(r)|^2.

State labelling uses the relativistic quantum numbers:
    n   principal quantum number
    l   orbital angular momentum
    j   total angular momentum, j = l +/- 1/2
    m   projection of j, in integer steps from -j to j
    k   Dirac quantum number, k = -(j + 1/2) for j = l + 1/2
                              k = +(j + 1/2) for j = l - 1/2

References
----------
W. Greiner, Relativistic Quantum Mechanics: Wave Equations, 3rd ed., ch. 9.
J. J. Sakurai, Advanced Quantum Mechanics, ch. 3.
"""

import cmath
import math

import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

## Physical constants (SI)
e_0  = 8.854187818814e-12        # vacuum permittivity      [F/m]
planck = 6.62607015e-34          # Planck constant          [J*s]
hbar = planck / (2 * math.pi)    # reduced Planck constant  [J*s]
m_e  = 9.109e-31                 # electron mass            [kg]
m_p  = 1.6726219e-27             # proton mass              [kg]
mu   = m_e * m_p / (m_e + m_p)   # reduced mass             [kg]
q_e  = 1.602176634e-19           # elementary charge        [C]
c    = 299792458                 # speed of light           [m/s]
Z    = 1                         # proton number (hydrogen) [-]
s    = 0.5                       # electron intrinsic spin  [-]

alpha = q_e**2 / (4 * math.pi * e_0 * hbar * c)        # fine-structure constant [-]
a_0   = 4 * math.pi * e_0 * hbar**2 / (mu * q_e**2)    # Bohr radius             [m]


def kappa(l, j):
    """Dirac quantum number k for the state (l, j); 0 means an invalid pairing."""
    if j == round(l + 0.5, 1):
        return int(-j - 0.5)
    if j == round(l - 0.5, 1):
        return int(j + 0.5)
    return 0


def radial_components(r, n, k, j, gamma):
    """
    Large (g) and small (f) Dirac radial components at radius r, plus the
    state energy E. Accepts a scalar or a numpy array for r.

    Two analytic branches are used: the maximum-j case (j = n - 1/2) has a
    nodeless closed form, while all other states carry a generalised Laguerre
    polynomial.
    """
    r = np.asarray(r, dtype=float)

    if j == (n - 0.5):
        E = (gamma / n) * mu * c**2
        C = ((Z * alpha) / n) * ((mu * c**2) / (hbar * c))
        rho = 2 * C * r
        A = (1 / cmath.sqrt(2 * n * (n + gamma))) * cmath.sqrt(C / (gamma * spec.gamma(2 * gamma)))
        g = A * (n + gamma) * rho**gamma * np.exp(-rho / 2)
        f = A * Z * alpha * rho**gamma * np.exp(-rho / 2)
    else:
        E = mu * c**2 * (1 + ((Z * alpha) / (n - abs(k) + gamma))**2)**(-1 / 2)
        C = cmath.sqrt(mu**2 * c**4 - E**2) / (hbar * c)
        rho = 2 * C * r
        A = (1 / np.sqrt(2 * k * (k - gamma))) * np.sqrt(
            (C / (n - abs(k) + gamma))
            * (math.factorial(n - abs(k) - 1) / spec.gamma(n - abs(k) + 2 * gamma + 1))
            * (1 / 2)
            * (((E * k) / (gamma * mu * c**2))**2 + ((E * k) / (gamma * mu * c**2)))
        )
        gen_lag = spec.genlaguerre(n - abs(k) - 1, round((2 * gamma + 1).real, 1))
        beta = ((gamma * mu * c**2) - (k * E)) / (hbar * c * C)
        g = A * rho**gamma * np.exp(-rho / 2) * (
            Z * alpha * rho * gen_lag(rho) + (gamma - k) * beta * gen_lag(rho)
        )
        f = A * rho**gamma * np.exp(-rho / 2) * (
            (gamma - k) * rho * gen_lag(rho) + Z * alpha * beta * gen_lag(rho)
        )

    return g, f, E


def _angular_sq(mq, degree, theta, phi):
    """
    |Y|^2 for the sign-corrected, normalised associated-Legendre angular factor.
    Returns 0 for invalid (mq, degree) pairs (|mq| > degree or degree < 0), which
    are the asymptotic / out-of-range terms the original code zeroed out.
    Accepts arrays for theta and phi.
    """
    if degree < 0 or abs(mq) > degree:
        return np.zeros_like(theta, dtype=float)
    norm = math.sqrt(
        ((2 * degree + 1) / (4 * np.pi))
        * (math.factorial(degree - mq) / math.factorial(degree + mq))
    )
    Y = norm * spec.lpmv(mq, degree, np.cos(theta)) * np.exp(1j * mq * phi)
    return np.abs(Y)**2


def density(r, theta, phi, n, k, j, m, gamma):
    """
    Spinor probability density |psi|^2 over arrays of (r, theta, phi).

    The four spinor components are orthogonal in their angular parts, so the
    total density is the sum of the large- and small-component contributions
    (the -sign(k) and +/- i phase factors drop out under | . |^2).
    """
    r_safe = r + 1e-300
    g, f, _ = radial_components(r, n, k, j, gamma)
    g_r = np.abs(g / r_safe)**2
    f_r = np.abs(f / r_safe)**2

    deg_k    = k if k > 0 else -k - 1
    deg_negk = k - 1 if k > 0 else -k

    c1 = abs((k + 0.5 - m) / (2 * k + 1))
    c2 = abs((k + 0.5 + m) / (2 * k + 1))
    c3 = abs((-k + 0.5 - m) / (2 * -k + 1))
    c4 = abs((-k + 0.5 + m) / (2 * -k + 1))

    ang_large = (c1 * _angular_sq(int(m - 0.5), deg_k, theta, phi)
                 + c2 * _angular_sq(int(m + 0.5), deg_k, theta, phi))
    ang_small = (c3 * _angular_sq(int(m - 0.5), deg_negk, theta, phi)
                 + c4 * _angular_sq(int(m + 0.5), deg_negk, theta, phi))

    return g_r * ang_large + f_r * ang_small


def density_cart(x, y, z, n, k, j, m, gamma):
    """Probability density |psi|^2 from Cartesian coordinates."""
    r     = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-300), -1.0, 1.0))
    phi   = np.arctan2(y, x) % (2 * np.pi)
    return density(r, theta, phi, n, k, j, m, gamma)


def choose(label, options):
    """Print a numbered menu, read a 1-based choice, and return the option."""
    print(f"Please choose a value for {label}:")
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    choice = int(input("Enter the number of your choice: "))
    if not 1 <= choice <= len(options):
        raise ValueError("Choice out of range.")
    selected = options[choice - 1]
    print(f"Selected {label}: {selected}")
    return selected


def prompt_state():
    """Collect a valid (n, l, j, m, k) hydrogen state from the user."""
    n = int(input("Principal quantum number n (1-11): "))
    if not 1 <= n <= 11:
        raise ValueError(f"n must be 1-11, got {n}")

    l = int(input(f"Azimuthal quantum number l (0-{n - 1}): "))
    if not 0 <= l < n:
        raise ValueError(f"l must be 0-{n - 1}, got {l}")

    j = choose("total angular momentum j", list(np.arange(abs(l - s), l + s + 1, 1)))
    m = choose("projection m", list(np.arange(-j, j + 1, 1)))

    spin = "spin-up" if m > 0 else "spin-down"
    print(f"The electron is {spin}.")

    k = kappa(l, j)
    if k == 0:
        raise ValueError("Invalid value for k.")
    print(f"Dirac quantum number k: {k}")

    return n, l, j, m, k, spin


ORBITAL_NAMES = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i', 7: 'k', 8: 'l'}


def build_figure(n, l, j, m, k, gamma, spin):
    """Compute the densities and render the five-panel summary figure."""
    # Radial density |g|^2 + |f|^2, used to size the spatial grid.
    r_scan = np.linspace(1e-13, n * n * 6e-9, 3000)
    g_arr, f_arr, E = radial_components(r_scan, n, k, j, gamma)
    radial_density = np.abs(g_arr)**2 + np.abs(f_arr)**2
    rd_norm = radial_density / radial_density.max()
    significant = np.where(rd_norm > 1e-3)[0]
    r_max = r_scan[significant[-1]] * 1.15 if significant.size else r_scan[-1]

    # 2D cross-sections (high resolution).
    x2 = np.linspace(-r_max, r_max, 400)
    Xxz, Zxz = np.meshgrid(x2, x2)
    d_xz = density_cart(Xxz, np.zeros_like(Xxz), Zxz, n, k, j, m, gamma)
    Xxy, Yxy = np.meshgrid(x2, x2)
    d_xy = density_cart(Xxy, Yxy, np.zeros_like(Xxy), n, k, j, m, gamma)

    # 3D probability cloud on a spherical grid for uniform angular coverage.
    Nr, Nth, Nphi = 55, 70, 80
    r_s   = np.linspace(1e-13, r_max, Nr)
    th_s  = np.linspace(0, np.pi, Nth)
    phi_s = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    Rg, THg, PHIg = np.meshgrid(r_s, th_s, phi_s, indexing='ij')
    d_sph = density(Rg, THg, PHIg, n, k, j, m, gamma)

    Xsph = Rg * np.sin(THg) * np.cos(PHIg)
    Ysph = Rg * np.sin(THg) * np.sin(PHIg)
    Zsph = Rg * np.cos(THg)

    thresh = 0.02 * d_sph.max()
    mask = d_sph > thresh
    xs, ys, zs = Xsph[mask] / a_0, Ysph[mask] / a_0, Zsph[mask] / a_0
    cs = d_sph[mask]

    ## Figure
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#08080f')
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.42, wspace=0.32,
        left=0.04, right=0.97, top=0.90, bottom=0.07,
    )

    j_label = f"{int(2 * j)}/2"
    orbital = f"{n}{ORBITAL_NAMES.get(l, '?')}$_{{{j_label}}}$"
    fig.suptitle(
        f"Hydrogen Atom (Dirac)  |  {orbital}    "
        f"$(n={n},\\ l={l},\\ j={j},\\ m={m},\\ k={k})$",
        fontsize=15, color='white', y=0.965, fontweight='bold',
    )

    # 3D probability cloud.
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    sc3d = ax3d.scatter(
        xs, ys, zs, c=cs, cmap='plasma', s=4, alpha=0.35, linewidths=0, rasterized=True,
    )
    ax3d.set_xlabel('x / a0', fontsize=8, labelpad=-4)
    ax3d.set_ylabel('y / a0', fontsize=8, labelpad=-4)
    ax3d.set_zlabel('z / a0', fontsize=8, labelpad=-4)
    ax3d.tick_params(labelsize=6.5)
    ax3d.set_title('Probability cloud', fontsize=10, pad=6)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#1e1e2e')
    ax3d.grid(True, color='#1e1e2e', alpha=0.6)
    cb3d = fig.colorbar(sc3d, ax=ax3d, fraction=0.028, pad=0.10, shrink=0.55)
    cb3d.set_label('|psi|^2', fontsize=8)
    cb3d.ax.tick_params(labelsize=6.5)

    ext = r_max / a_0

    # xz cross-section.
    ax_xz = fig.add_subplot(gs[0, 1])
    im_xz = ax_xz.imshow(
        d_xz, origin='lower', cmap='inferno', extent=[-ext, ext, -ext, ext], aspect='equal',
    )
    ax_xz.set_xlabel('x / a0', fontsize=9)
    ax_xz.set_ylabel('z / a0', fontsize=9)
    ax_xz.set_title('xz cross-section  (y = 0)', fontsize=10)
    ax_xz.tick_params(labelsize=7)
    cb = fig.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label('|psi|^2', fontsize=8)

    # xy cross-section.
    ax_xy = fig.add_subplot(gs[1, 1])
    im_xy = ax_xy.imshow(
        d_xy, origin='lower', cmap='inferno', extent=[-ext, ext, -ext, ext], aspect='equal',
    )
    ax_xy.set_xlabel('x / a0', fontsize=9)
    ax_xy.set_ylabel('y / a0', fontsize=9)
    ax_xy.set_title('xy cross-section  (z = 0)', fontsize=10)
    ax_xy.tick_params(labelsize=7)
    cb2 = fig.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=6.5)
    cb2.set_label('|psi|^2', fontsize=8)

    # Radial density.
    ax_r = fig.add_subplot(gs[0, 2])
    show = r_scan <= r_max
    ax_r.plot(r_scan[show] / a_0, rd_norm[show], color='#ff6b9d', linewidth=1.8)
    ax_r.fill_between(r_scan[show] / a_0, rd_norm[show], alpha=0.22, color='#ff6b9d')
    ax_r.set_xlabel('r / a0', fontsize=9)
    ax_r.set_ylabel('|g|^2 + |f|^2 (norm.)', fontsize=9)
    ax_r.set_title('Radial density', fontsize=10)
    ax_r.tick_params(labelsize=7)
    ax_r.set_xlim(0, r_max / a_0)
    ax_r.set_ylim(bottom=0)

    # State properties table.
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.set_facecolor('#0e0e1c')
    for sp in ax_info.spines.values():
        sp.set_color('#2a2a4a')
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    binding_eV = (float(E.real) - mu * c**2) / q_e
    r_peak = r_scan[np.argmax(radial_density)]
    props = [
        ("Orbital",      f"{n}{ORBITAL_NAMES.get(l, '?')}_{j_label}"),
        ("Spin",         spin),
        ("Dirac k",      f"{k}"),
        ("gamma",        f"{gamma:.6f}"),
        ("Binding",      f"{binding_eV:.4f} eV"),
        ("r_peak",       f"{r_peak / a_0:.3f} a0"),
    ]
    for i, (key, val) in enumerate(props):
        ax_info.text(0.06, 0.87 - i * 0.145, f"{key}:", fontsize=9.5,
                     color='#7b8ec8', family='monospace', transform=ax_info.transAxes)
        ax_info.text(0.50, 0.87 - i * 0.145, val, fontsize=9.5,
                     color='#d0d8f0', family='monospace', transform=ax_info.transAxes)
    ax_info.set_title('State properties', fontsize=10)

    plt.savefig('hydrogen_dirac.png', dpi=150, bbox_inches='tight', facecolor='#08080f')
    plt.show()


def main():
    n, l, j, m, k, spin = prompt_state()
    gamma = math.sqrt(k**2 - Z**2 * alpha**2)   # real and positive for Z < 137
    build_figure(n, l, j, m, k, gamma, spin)


if __name__ == "__main__":
    main()
