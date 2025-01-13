# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:06:35 2024

@author: tacco
"""

import numpy as np
import scipy.special as spec
import math
import cmath
import matplotlib.pyplot as plt

e_0 = 8.854187818814 * 10 ** (-12) # F⋅m**(−1) Vacuum Permitivity/Permitivity of Free Space/Electric Constant
planck = 6.62607015 * 10 ** (-34)
reduced_planck = planck / (2 * cmath.pi)
m_e = 9.109 * 10 ** (-31) # kg
m_p = 1.6726219 * 10 ** (-27)
mu = (m_e * m_p)/(m_e + m_p) # Accounts for masses of system
elementary_charge = 1.602176634 * 10 ** (-19) # Coulombs
non_reduced_a_0 = (4 * cmath.pi * e_0 * reduced_planck ** 2)/(elementary_charge ** 2 * m_e)
a_0 = (4 * np.pi * e_0 * reduced_planck ** 2) / (mu * elementary_charge ** 2) # Reduced Bohr Radius

n = int(input("What is the principal quantum number? ")) # Excited State, 4 is max?
l = int(input("What is the azimuthal quantum number? ")) # 0 -> n-1, orbital angular momentum
m = int(input("What is the magnetic quantum number? ")) # 0 -> -l or 0 -> l
# Ground state is 1, 0, 0

# r as dependent variable
def rho(radius):
    return (2 * np.array(radius) / (n * a_0))

GenLaguerre = spec.genlaguerre((n - l - 1), (2 * l + 1))
theta, phi = np.mgrid[0:2 * cmath.pi:100j, 0:2 * cmath.pi:100j]
Y = spec.sph_harm(m, l, phi, theta)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect("equal")

r = 6 * 10 ** (-9)
x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)

psi = math.sqrt((2 / (n * a_0)) ** 3 * (math.factorial(n - l - 1)/(2 * n * math.factorial(n + l)))) * cmath.exp(-rho(r) / 2) * rho(r) ** l * GenLaguerre(rho(r)) * Y

intensities = (np.abs(psi) ** 2) / np.max(np.abs(psi) ** 2)

x_flattened = np.ndarray.flatten(x)
y_flattened = np.ndarray.flatten(y)
z_flattened = np.ndarray.flatten(z)
intensities_flattened = np.ndarray.flatten(intensities)
indices_to_delete = np.where(intensities_flattened < 0.0)[0]

x_filtered = np.delete(x_flattened, indices_to_delete)
y_filtered = np.delete(y_flattened, indices_to_delete)
z_filtered = np.delete(z_flattened, indices_to_delete)
intensities_filtered = np.delete(intensities_flattened, indices_to_delete)

x = np.reshape(x_filtered, [-1, 100])
y = np.reshape(y_filtered, [-1, 100])
z = np.reshape(z_filtered, [-1, 100])
intensities = np.reshape(intensities_filtered, [-1, 100])

scatter = ax.scatter(x, y, z, c=intensities, cmap='viridis')
plt.title("Probability of Finding Electron around Hydrogen Atom\nn = " + str(n) + "\nl = " + str(l) + "\nm = " + str(m))
ax.set_xlabel("          x distance (m)")
ax.xaxis.labelpad = -15
# Add a color bar
colorbar = plt.colorbar(scatter)
colorbar.set_label('Relative Probability')
plt.show()
