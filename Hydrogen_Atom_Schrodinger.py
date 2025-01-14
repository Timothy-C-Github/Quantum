# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:06:35 2024

@author: tacco
"""

## Import statements, scipy.special provides all the special functions needed and visuals are through matplotlib 3D plotting
import cmath
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import math

## Necessary Constants along with Units

# Vacuum Permitivity/Permitivity of Free Space/Electric Constant | Units: F / m 
e_0 = 8.854187818814 * 10 ** (-12)
# Planck's Constant | Units: J / Hz
planck = 6.62607015 * 10 ** (-34)
# Reduced Planck's Constant | Units: J / Hz
reduced_planck = planck / (2 * cmath.pi)
# Electron Mass | Units: kg
m_e = 9.109 * 10 ** (-31)
# Proton Mass | Units: kg
m_p = 1.6726219 * 10 ** (-27)
# Reduced Mass/Inertial Mass | Units: kg
mu = (m_e * m_p) / (m_e + m_p)
# Elementary Charge | Units: Coulombs
elementary_charge = 1.602176634 * 10 ** (-19)
# Non-Reduced Bohr Radius | Units: m
non_reduced_a_0 = (4 * cmath.pi * e_0 * reduced_planck ** 2)/(elementary_charge ** 2 * m_e)
# Bohr Radius | Units: m
a_0 = (4 * np.pi * e_0 * reduced_planck ** 2) / (mu * elementary_charge ** 2) # Reduced Bohr Radius

## User-Inputted Data on Hydrogen State

# Principal Quantum Number/Shell Designator: Theoretically any natural number for n works but I have constrained the values to the highest named spectrum (Humphreys Series)
n = int(input("What is the principal quantum number n? ")) 
if n > 11:
    raise Exception(ValueError("Invalid or Unnatural Excited State"))
# Azimuthal Quantum Number/Orbital Angular Momentum/Subshell Designator: Has to always be within the range of 0 and n - 1
l = int(input("What is the azimuthal quantum number l? ")) 
if l >= n or l < 0:
    raise Exception(ValueError("Invalid Subshell"))
# Magnetic Quantum Number; projection of the angular momentum on the (arbitrarily chosen) z-axis: Can only be an integer at or between -l and l
m = int(input("What is the magnetic quantum number m? "))
if abs(m) > l:
    raise Exception((ValueError("Invalid Magnetic Quantum Number")))

## Setting up solution to Schrodinger equation for Hydrogen atom

# Set r as dependent variable
def rho(radius):
    return (2 * np.array(radius) / (n * a_0))

# Create a grid of theta and phi values for the solution space
theta, phi = np.mgrid[0:2 * cmath.pi:100j, 0:2 * cmath.pi:100j]

# Create spherical harmonic (hypergeometric) functions
GenLaguerre = spec.genlaguerre((n - l - 1), (2 * l + 1))
Y = spec.sph_harm(m, l, phi, theta)

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.set_aspect("equal")

# Radius value to set for to measure Hydrogen atom electron | Units: m
r = 6 * 10 ** (-9)

# Transform coordinates from theta, phi to x, y, z
x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)

# Input solution for each value in theta, phi grid
psi = math.sqrt((2 / (n * a_0)) ** 3 * (math.factorial(n - l - 1)/(2 * n * math.factorial(n + l)))) * cmath.exp(-rho(r) / 2) * rho(r) ** l * GenLaguerre(rho(r)) * Y

# 3D plot is color-coded, so it actually represents 4 parameters instead of 3 with the 4th parameter being the observable probability (psi squared)
intensities = (np.abs(psi) ** 2) / np.max(np.abs(psi) ** 2)

# Correcting data arrays
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

## Visualization of Schrodinger solution for Hydrogen
# Setting axes and plotting for Hydrogen atom Schrodinger solution
scatter = ax.scatter(x, y, z, c = intensities, cmap = 'viridis')
plt.title("Probability of Finding Electron around Hydrogen Atom (Schrodinger)\nn = " + str(n) + "\nl = " + str(l) + "\nm = " + str(m) + "\nr = " + str(r) + " m")
ax.set_xlabel("          x distance (m)")
ax.xaxis.labelpad = -15
# Add a color bar
colorbar = plt.colorbar(scatter)
colorbar.set_label('Relative Probability')
plt.show()
