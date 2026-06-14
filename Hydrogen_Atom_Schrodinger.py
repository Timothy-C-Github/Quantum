#!/usr/bin/env python3
"""
Hydrogen atom — analytical solution to the Schrödinger equation.

Exact wavefunction via associated Laguerre polynomials (radial part)
and spherical harmonics (angular part). No variational or statistical
approximations; the full 3D probability density is evaluated on a
regular Cartesian grid.
"""

import math
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

# ── Physical constants (SI) ──────────────────────────────────────────────────
_e0   = 8.8541878128e-12          # vacuum permittivity   [F/m]
_hbar = 1.054571817e-34           # reduced Planck const  [J·s]
_m_e  = 9.1093837015e-31          # electron mass         [kg]
_m_p  = 1.67262192369e-27         # proton mass           [kg]
_mu   = _m_e * _m_p / (_m_e + _m_p)   # reduced mass     [kg]
_q_e  = 1.602176634e-19           # elementary charge     [C]

a_0    = 4*np.pi*_e0*_hbar**2 / (_mu*_q_e**2)          # Bohr radius   [m]
E_1_eV = -_mu*_q_e**4 / (8*_e0**2*_hbar**2) / _q_e     # E_1           [eV]

# ── Quantum number input ─────────────────────────────────────────────────────
n = int(input("Principal quantum number n (1–11): "))
if not 1 <= n <= 11:
    raise ValueError(f"n must be 1–11, got {n}")

l = int(input(f"Azimuthal quantum number l (0–{n-1}): "))
if not 0 <= l < n:
    raise ValueError(f"l must be 0–{n-1}, got {l}")

m = int(input(f"Magnetic quantum number m ({-l}–{l}): "))
if abs(m) > l:
    raise ValueError(f"|m| ≤ {l} required, got {m}")

# ── Wavefunction ─────────────────────────────────────────────────────────────
_norm = math.sqrt(
    (2.0 / (n * a_0))**3
    * math.factorial(n - l - 1)
    / (2 * n * math.factorial(n + l))
)
_L = spec.genlaguerre(n - l - 1, 2*l + 1)


def psi_squared(x, y, z):
    """Exact probability density |ψ(r,θ,φ)|² on a Cartesian grid."""
    r     = np.sqrt(x**2 + y**2 + z**2)
    rho   = 2.0 * r / (n * a_0)
    theta = np.arccos(np.clip(z / (r + 1e-300), -1.0, 1.0))  # polar   ∈ [0, π]
    phi   = np.arctan2(y, x) % (2 * np.pi)                    # azimuth ∈ [0, 2π]
    R_nl  = _norm * np.exp(-rho / 2.0) * rho**l * _L(rho)
    # sph_harm_y(l, m, polar_theta, azimuthal_phi) — scipy ≥ 1.15 convention
    Y_lm  = spec.sph_harm_y(l, m, theta, phi)
    return np.abs(R_nl * Y_lm)**2


# ── Grids ────────────────────────────────────────────────────────────────────
# Extent chosen to capture ≥ 99.9 % of probability for any (n, l, m)
r_max = (n**2 + 5*n) * a_0

# High-resolution 2D cross-sections
_x2 = np.linspace(-r_max, r_max, 500)

_Xxz, _Zzz = np.meshgrid(_x2, _x2)
d_xz = psi_squared(_Xxz, np.zeros_like(_Xxz), _Zzz)

_Xxy, _Yxy = np.meshgrid(_x2, _x2)
d_xy = psi_squared(_Xxy, _Yxy, np.zeros_like(_Xxy))

# Radial probability distribution  P(r) = r² |R_{nl}(r)|²
_r1d = np.linspace(0, r_max, 3000)
_rho1d = 2.0 * _r1d / (n * a_0)
_R1d = _norm * np.exp(-_rho1d / 2.0) * _rho1d**l * _L(_rho1d)
P_r  = _r1d**2 * _R1d**2

# 3D probability cloud on a regular Cartesian grid
_N3 = 65
_x3 = np.linspace(-r_max, r_max, _N3)
X3, Y3, Z3 = np.meshgrid(_x3, _x3, _x3)
d3 = psi_squared(X3, Y3, Z3)

_thresh = 0.008 * d3.max()
_mask   = d3 > _thresh
xs = X3[_mask] / a_0
ys = Y3[_mask] / a_0
zs = Z3[_mask] / a_0
cs = d3[_mask]

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#08080f')

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.42, wspace=0.32,
    left=0.04, right=0.97, top=0.90, bottom=0.07
)

_ORBITAL = {0:'s', 1:'p', 2:'d', 3:'f', 4:'g', 5:'h', 6:'i', 7:'k', 8:'l'}
_lbl = f"{n}{_ORBITAL.get(l, '?')}"
fig.suptitle(
    f"Hydrogen atom  ·  {_lbl} orbital    "
    f"$(n={n},\\ l={l},\\ m={m})$",
    fontsize=15, color='white', y=0.965, fontweight='bold'
)

# ── Panel 1: 3D probability cloud ────────────────────────────────────────────
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
sc3d = ax3d.scatter(
    xs, ys, zs, c=cs,
    cmap='plasma', s=1.8, alpha=0.18, linewidths=0, rasterized=True
)
ax3d.set_xlabel('x / a₀', fontsize=8, labelpad=-4)
ax3d.set_ylabel('y / a₀', fontsize=8, labelpad=-4)
ax3d.set_zlabel('z / a₀', fontsize=8, labelpad=-4)
ax3d.tick_params(labelsize=6.5)
ax3d.set_title('Probability cloud', fontsize=10, pad=6)
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor('#1e1e2e')
ax3d.grid(True, color='#1e1e2e', alpha=0.6)
cb3d = fig.colorbar(sc3d, ax=ax3d, fraction=0.028, pad=0.10, shrink=0.55)
cb3d.set_label('|ψ|²', fontsize=8)
cb3d.ax.tick_params(labelsize=6.5)

# ── Panel 2: xz cross-section ────────────────────────────────────────────────
_ext = r_max / a_0
ax_xz = fig.add_subplot(gs[0, 1])
im_xz = ax_xz.imshow(
    d_xz, origin='lower', cmap='inferno',
    extent=[-_ext, _ext, -_ext, _ext], aspect='equal'
)
ax_xz.set_xlabel('x / a₀', fontsize=9)
ax_xz.set_ylabel('z / a₀', fontsize=9)
ax_xz.set_title('xz cross-section  (y = 0)', fontsize=10)
ax_xz.tick_params(labelsize=7)
_cb = fig.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04)
_cb.ax.tick_params(labelsize=6.5)
_cb.set_label('|ψ|²', fontsize=8)

# ── Panel 3: xy cross-section ────────────────────────────────────────────────
ax_xy = fig.add_subplot(gs[1, 1])
im_xy = ax_xy.imshow(
    d_xy, origin='lower', cmap='inferno',
    extent=[-_ext, _ext, -_ext, _ext], aspect='equal'
)
ax_xy.set_xlabel('x / a₀', fontsize=9)
ax_xy.set_ylabel('y / a₀', fontsize=9)
ax_xy.set_title('xy cross-section  (z = 0)', fontsize=10)
ax_xy.tick_params(labelsize=7)
_cb2 = fig.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04)
_cb2.ax.tick_params(labelsize=6.5)
_cb2.set_label('|ψ|²', fontsize=8)

# ── Panel 4: Radial probability distribution ─────────────────────────────────
ax_r = fig.add_subplot(gs[0, 2])
ax_r.plot(_r1d / a_0, P_r * a_0, color='#ff6b9d', linewidth=1.8)
ax_r.fill_between(_r1d / a_0, P_r * a_0, alpha=0.22, color='#ff6b9d')
ax_r.set_xlabel('r / a₀', fontsize=9)
ax_r.set_ylabel('P(r) · a₀', fontsize=9)
ax_r.set_title(r'Radial distribution  $r^2|R_{nl}|^2$', fontsize=10)
ax_r.tick_params(labelsize=7)
ax_r.set_xlim(0, r_max / a_0)
ax_r.set_ylim(bottom=0)

# ── Panel 5: State summary ───────────────────────────────────────────────────
ax_info = fig.add_subplot(gs[1, 2])
ax_info.set_facecolor('#0e0e1c')
for sp in ax_info.spines.values():
    sp.set_color('#2a2a4a')

E_n     = E_1_eV / n**2
r_mean  = a_0 / 2 * (3*n**2 - l*(l+1))          # ⟨r⟩ exact [m]
r_peak1 = max((_rho1d[np.argmax(P_r)] * n * a_0 / 2) if len(_rho1d) else 0, 0)

props = [
    ("Orbital",         f"{_lbl}"),
    ("Energy",          f"{E_n:.5f} eV"),
    ("Radial nodes",    f"{n - l - 1}"),
    ("Angular nodes",   f"{l}"),
    ("⟨r⟩",            f"{r_mean/a_0:.3f} a₀"),
    ("r_peak",          f"{r_peak1/a_0:.3f} a₀"),
]

ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
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
