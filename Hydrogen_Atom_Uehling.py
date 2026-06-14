#!/usr/bin/env python3
"""
Hydrogen atom - Uehling QED vacuum polarization correction.

The Uehling potential is expressed exactly as a single Meijer G-function
following Koegler & Schneider (arXiv:2209.15020, eq. 5), which encodes
the Frolov-Wardlaw (2012) result in a compact closed form:

    V_U(r) = -alpha/(16*pi^2*r) * G^{4,0}_{2,4}( r^2 | 1, 5/2; 0, 0, 1/2, 2 )

All distances are in units of the Compton wavelength lambda_C = hbar/(m_e*c).
Potentials are in units of m_e*c^2 (electron rest energy).

The QED-corrected total potential seen by the bound-state electron is:
    V(r) = V_C(r) - (alpha/pi) * V_U(r)
         = -alpha/(4*pi*r) * [ 1 - alpha/(4*pi^2) * G^{4,0}_{2,4}(r^2 | ...) ]

References
----------
Koegler & Schneider, arXiv:2209.15020v3 (2025), eqs. (5) and (8).
Frolov & Wardlaw, Eur. Phys. J. B 85, 348 (2012) [arXiv:1110.3433].
"""

import numpy as np
import mpmath
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

plt.style.use('dark_background')

## Constants
alpha = 7.2973525693e-3    # fine-structure constant (dimensionless)
# 1 Bohr radius = 1/alpha Compton wavelengths ~ 137.036 lambda_C


## Potentials (Compton wavelength units, energy in m_e*c^2)

def V_coulomb(r_C):
    """Coulomb potential V_C(r) = -alpha/(4*pi*r) in Compton wavelength units."""
    return -alpha / (4.0 * np.pi * r_C)


def _meijer_G(r_C_scalar):
    """G^{4,0}_{2,4}( r^2 | 1, 5/2; 0, 0, 1/2, 2 ) via mpmath."""
    z = mpmath.mpf(r_C_scalar)**2
    return float(mpmath.meijerg(
        [[], [1, mpmath.mpf(5)/2]],
        [[0, 0, mpmath.mpf(1)/2, 2], []],
        z
    ))


def V_uehling(r_C):
    """
    Uehling vacuum polarization potential (Koegler-Schneider eq. 5).
    V_U(r) = -alpha/(16*pi^2*r) * G^{4,0}_{2,4}(r^2 | 1, 5/2; 0, 0, 1/2, 2)
    """
    r_C = np.asarray(r_C, dtype=float)
    G   = np.array([_meijer_G(ri) for ri in r_C])
    return -alpha / (16.0 * np.pi**2 * r_C) * G


def V_total(r_C, VU=None):
    """Full QED-corrected potential V = V_C - (alpha/pi)*V_U."""
    VC = V_coulomb(r_C)
    if VU is None:
        VU = V_uehling(r_C)
    return VC - (alpha / np.pi) * VU


## Compute potentials
# Uehling correction decays as exp(-2r) in Compton units, negligible beyond r~10.
# Proton radius ~3.5e-4 lambda_C sets the lower physical cut-off.
print("Computing Uehling potential (Meijer G-function)...")
r_C  = np.linspace(0.04, 9.0, 300)   # Compton wavelengths
r_a0 = r_C * alpha                    # same points in Bohr radii

VC   = V_coulomb(r_C)
VU   = V_uehling(r_C)
VT   = V_total(r_C, VU)
dV   = VT - VC                        # Uehling shift dV = -(alpha/pi)*V_U
frac = np.abs(dV / VC)                # fractional correction |dV/V_C|

print("Done.")

## Figure
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor('#08080f')

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.45, wspace=0.38,
    left=0.07, right=0.97, top=0.88, bottom=0.08
)

fig.suptitle(
    r"Hydrogen Atom  |  Uehling QED Vacuum Polarization Correction"
    "\n"
    r"$V_U(r) = -\dfrac{\alpha}{16\pi^2\, r}\;"
    r"G^{4,0}_{2,4}\!\left(r^2 \;\middle|\; a\!=\![1,\,5/2],\; b\!=\![0,0,1/2,2]\right)$",
    fontsize=13, color='white', y=0.97
)

SPINE = '#2a2a4a'


def style_ax(ax):
    ax.set_facecolor('#0e0e1c')
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(labelsize=8, colors='#b0b8d8')
    ax.xaxis.label.set_color('#b0b8d8')
    ax.yaxis.label.set_color('#b0b8d8')
    ax.title.set_color('white')


# Coulomb vs QED-corrected potential
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1)

ylim_lo  = max(VC.min(), -0.012)
show     = VC > ylim_lo
ax1.plot(r_a0[show], VC[show], color='#4fc3f7', lw=2.0,
         label=r'Coulomb  $V_C(r) = -\alpha/(4\pi r)$', zorder=3)
ax1.plot(r_a0[show], VT[show], color='#ff7043', lw=1.5, ls='--',
         label=r'QED-corrected  $V(r) = V_C - (\alpha/\pi)\,V_U$', zorder=4)
ax1.set_xlabel(r'$r \;[a_0]$  (Bohr radii)', fontsize=10)
ax1.set_ylabel(r'Potential  $[m_e c^2]$', fontsize=10)
ax1.set_title('Coulomb vs QED-corrected potential', fontsize=11)
ax1.legend(fontsize=9, facecolor='#12122a', edgecolor=SPINE)

# Inset zoomed near origin
ins = ax1.inset_axes([0.60, 0.30, 0.37, 0.55])
ins.set_facecolor('#070712')
near = r_C < 1.5
ins.plot(r_a0[near], VC[near], color='#4fc3f7', lw=1.5)
ins.plot(r_a0[near], VT[near], color='#ff7043', lw=1.2, ls='--')
ins.set_title(r'Zoom: $r < 1.5\,\lambda_C$', fontsize=7.5, color='#b0b8d8')
ins.tick_params(labelsize=6.5, colors='#888')
for sp in ins.spines.values():
    sp.set_color('#2a2a4a')

# Uehling shift dV
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2)
ax2.plot(r_a0, dV, color='#a5d6a7', lw=1.8)
ax2.fill_between(r_a0, dV, alpha=0.25, color='#a5d6a7')
ax2.set_xlabel(r'$r \;[a_0]$', fontsize=10)
ax2.set_ylabel(r'$\delta V = V - V_C \;[m_e c^2]$', fontsize=10)
ax2.set_title(r'Uehling shift  $\delta V(r)$', fontsize=11)
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))

# Fractional correction (log scale)
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3)
ax3.semilogy(r_a0, frac, color='#ce93d8', lw=1.8)
ax3.semilogy(r_a0, np.full_like(r_a0, alpha**2),
             color='#888', lw=0.8, ls=':', label=r'$\alpha^2$')
ax3.set_xlabel(r'$r \;[a_0]$', fontsize=10)
ax3.set_ylabel(r'$|\delta V / V_C|$', fontsize=10)
ax3.set_title('Fractional QED correction', fontsize=11)
ax3.legend(fontsize=8, facecolor='#12122a', edgecolor=SPINE)
ax3.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=10))
ax3.grid(True, which='both', color=SPINE, alpha=0.4)

# Energy shift table
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor('#0e0e1c')
for sp in ax4.spines.values():
    sp.set_color(SPINE)
ax4.axis('off')

# 1st-order Lamb shift contributions from Uehling potential
# dE_nl = (alpha/pi) * <nl|V_U|nl>, values from arXiv:2209.15020 p.3
shifts = [
    ("1s  (n=1, l=0)", "-8.896 x 10^-7 eV"),
    ("2s  (n=2, l=0)", "-1.112 x 10^-7 eV"),
    ("3s  (n=3, l=0)", "-3.295 x 10^-8 eV"),
    ("2p  (n=2, l=1)", "-3.166 x 10^-13 eV"),
]
ax4.text(0.5, 0.97, "Uehling energy shifts", ha='center', va='top',
         fontsize=9.5, color='#7b8ec8', transform=ax4.transAxes, fontweight='bold')
ax4.text(0.5, 0.87, "(Lamb shift contribution)", ha='center', va='top',
         fontsize=8, color='#555577', transform=ax4.transAxes)
for i, (state, shift) in enumerate(shifts):
    ax4.text(0.05, 0.73 - i*0.14, state, fontsize=8.5, color='#a0b0d8',
             family='monospace', transform=ax4.transAxes)
    ax4.text(0.05, 0.65 - i*0.14, shift, fontsize=8.5, color='#d0d8f0',
             family='monospace', transform=ax4.transAxes)
ax4.text(0.5, 0.08, r"Only $s$-states shift at 1st order",
         ha='center', va='bottom', fontsize=7.5, color='#555577',
         transform=ax4.transAxes)
ax4.set_title('Energy shifts', fontsize=10, color='white')

plt.savefig('hydrogen_uehling.png', dpi=150, bbox_inches='tight',
            facecolor='#08080f')
plt.show()
