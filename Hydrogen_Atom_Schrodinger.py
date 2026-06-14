#!/usr/bin/env python3
"""
Hydrogen atom - analytical solution to the Schrodinger equation.

Exact wavefunction using associated Laguerre polynomials (radial)
and spherical harmonics (angular). Probability density evaluated on
a spherical coordinate grid for accurate 3D visualization.
"""

import math
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

## Physical constants (SI)
e0   = 8.8541878128e-12          # vacuum permittivity   [F/m]
hbar = 1.054571817e-34           # reduced Planck const  [J*s]
m_e  = 9.1093837015e-31          # electron mass         [kg]
m_p  = 1.67262192369e-27         # proton mass           [kg]
mu   = m_e * m_p / (m_e + m_p)  # reduced mass          [kg]
q_e  = 1.602176634e-19           # elementary charge     [C]

a_0    = 4*np.pi*e0*hbar**2 / (mu*q_e**2)        # Bohr radius [m]
E_1_eV = -mu*q_e**4 / (8*e0**2*hbar**2) / q_e   # ground state energy [eV]

## Quantum number input
n = int(input("Principal quantum number n (1-11): "))
if not 1 <= n <= 11:
    raise ValueError(f"n must be 1-11, got {n}")

l = int(input(f"Azimuthal quantum number l (0-{n-1}): "))
if not 0 <= l < n:
    raise ValueError(f"l must be 0-{n-1}, got {l}")

m = int(input(f"Magnetic quantum number m ({-l}-{l}): "))
if abs(m) > l:
    raise ValueError(f"|m| <= {l} required, got {m}")

## Wavefunction components
norm = math.sqrt(
    (2.0 / (n * a_0))**3
    * math.factorial(n - l - 1)
    / (2 * n * math.factorial(n + l))
)
L = spec.genlaguerre(n - l - 1, 2*l + 1)


def radial(r):
    rho = 2.0 * r / (n * a_0)
    return norm * np.exp(-rho / 2.0) * rho**l * L(rho)


def psi_sq_cart(x, y, z):
    """Probability density |psi|^2 from Cartesian coordinates."""
    r     = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-300), -1.0, 1.0))
    phi   = np.arctan2(y, x) % (2 * np.pi)
    R_nl  = radial(r)
    Y_lm  = spec.sph_harm_y(l, m, theta, phi)
    return np.abs(R_nl * Y_lm)**2


## Grid setup
# r_max captures all radial nodes with margin
r_max = (n**2 + 5*n) * a_0

# 2D cross-sections (high resolution)
x2 = np.linspace(-r_max, r_max, 500)

Xxz, Zzz = np.meshgrid(x2, x2)
d_xz = psi_sq_cart(Xxz, np.zeros_like(Xxz), Zzz)

Xxy, Yxy = np.meshgrid(x2, x2)
d_xy = psi_sq_cart(Xxy, Yxy, np.zeros_like(Xxy))

# Radial probability distribution P(r) = r^2 * |R_nl(r)|^2
r1d   = np.linspace(0, r_max, 3000)
P_r   = r1d**2 * radial(r1d)**2

# 3D probability cloud on a spherical grid for uniform angular coverage
Nr, Nth, Nphi = 55, 70, 80
r_s   = np.linspace(1e-12, r_max, Nr)
th_s  = np.linspace(0, np.pi, Nth)
phi_s = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

Rg, THg, PHIg = np.meshgrid(r_s, th_s, phi_s, indexing='ij')

rho_g = 2.0 * Rg / (n * a_0)
Rnl_g = norm * np.exp(-rho_g / 2.0) * rho_g**l * L(rho_g)
Ylm_g = spec.sph_harm_y(l, m, THg, PHIg)
d_sph = np.abs(Rnl_g * Ylm_g)**2

Xsph = Rg * np.sin(THg) * np.cos(PHIg)
Ysph = Rg * np.sin(THg) * np.sin(PHIg)
Zsph = Rg * np.cos(THg)

thresh = 0.015 * d_sph.max()
mask   = d_sph > thresh
xs = Xsph[mask] / a_0
ys = Ysph[mask] / a_0
zs = Zsph[mask] / a_0
cs = d_sph[mask]

## Figure
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#08080f')

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.42, wspace=0.32,
    left=0.04, right=0.97, top=0.90, bottom=0.07
)

orbital_names = {0:'s', 1:'p', 2:'d', 3:'f', 4:'g', 5:'h', 6:'i', 7:'k', 8:'l'}
lbl = f"{n}{orbital_names.get(l, '?')}"
fig.suptitle(
    f"Hydrogen Atom  |  {lbl} orbital    $(n={n},\\ l={l},\\ m={m})$",
    fontsize=15, color='white', y=0.965, fontweight='bold'
)

# 3D probability cloud
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
sc3d = ax3d.scatter(
    xs, ys, zs, c=cs,
    cmap='plasma', s=4, alpha=0.35, linewidths=0, rasterized=True
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

# xz cross-section
ext = r_max / a_0
ax_xz = fig.add_subplot(gs[0, 1])
im_xz = ax_xz.imshow(
    d_xz, origin='lower', cmap='inferno',
    extent=[-ext, ext, -ext, ext], aspect='equal'
)
ax_xz.set_xlabel('x / a0', fontsize=9)
ax_xz.set_ylabel('z / a0', fontsize=9)
ax_xz.set_title('xz cross-section  (y = 0)', fontsize=10)
ax_xz.tick_params(labelsize=7)
cb = fig.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=6.5)
cb.set_label('|psi|^2', fontsize=8)

# xy cross-section
ax_xy = fig.add_subplot(gs[1, 1])
im_xy = ax_xy.imshow(
    d_xy, origin='lower', cmap='inferno',
    extent=[-ext, ext, -ext, ext], aspect='equal'
)
ax_xy.set_xlabel('x / a0', fontsize=9)
ax_xy.set_ylabel('y / a0', fontsize=9)
ax_xy.set_title('xy cross-section  (z = 0)', fontsize=10)
ax_xy.tick_params(labelsize=7)
cb2 = fig.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04)
cb2.ax.tick_params(labelsize=6.5)
cb2.set_label('|psi|^2', fontsize=8)

# Radial probability distribution
ax_r = fig.add_subplot(gs[0, 2])
ax_r.plot(r1d / a_0, P_r * a_0, color='#ff6b9d', linewidth=1.8)
ax_r.fill_between(r1d / a_0, P_r * a_0, alpha=0.22, color='#ff6b9d')
ax_r.set_xlabel('r / a0', fontsize=9)
ax_r.set_ylabel('P(r) * a0', fontsize=9)
ax_r.set_title(r'Radial distribution  $r^2|R_{nl}|^2$', fontsize=10)
ax_r.tick_params(labelsize=7)
ax_r.set_xlim(0, r_max / a_0)
ax_r.set_ylim(bottom=0)

# State properties
ax_info = fig.add_subplot(gs[1, 2])
ax_info.set_facecolor('#0e0e1c')
for sp in ax_info.spines.values():
    sp.set_color('#2a2a4a')

E_n    = E_1_eV / n**2
r_mean = a_0 / 2 * (3*n**2 - l*(l+1))
r_peak = r1d[np.argmax(P_r)]

props = [
    ("Orbital",       lbl),
    ("Energy",        f"{E_n:.5f} eV"),
    ("Radial nodes",  f"{n - l - 1}"),
    ("Angular nodes", f"{l}"),
    ("<r>",           f"{r_mean/a_0:.3f} a0"),
    ("r_peak",        f"{r_peak/a_0:.3f} a0"),
]

ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)
ax_info.axis('off')
for i, (key, val) in enumerate(props):
    ax_info.text(0.06, 0.87 - i*0.145, f"{key}:", fontsize=9.5,
                 color='#7b8ec8', family='monospace', transform=ax_info.transAxes)
    ax_info.text(0.52, 0.87 - i*0.145, val, fontsize=9.5,
                 color='#d0d8f0', family='monospace', transform=ax_info.transAxes)
ax_info.set_title('State properties', fontsize=10)

plt.savefig('hydrogen_schrodinger.png', dpi=150, bbox_inches='tight',
            facecolor='#08080f')
plt.show()
