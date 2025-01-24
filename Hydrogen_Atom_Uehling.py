# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:36:45 2024

@author: tacco
"""

## Import statements, scipy.special provides all the special functions needed and visuals are through matplotlib 3D plotting
import cmath
import numpy as np
import sys
import math
import scipy.special as spec
import matplotlib.pyplot as plt

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
# Proton Number Z | Units: Unitless
proton_number = 1
# Speed of Light | Units: m / s
speed_of_light = 299792458
# Fine Structure Constant | Units: Unitless
fine_structure_constant = elementary_charge ** 2 / (4 * cmath.pi * e_0 * reduced_planck * speed_of_light)
# Intrinsic Spin (Scalar, not a Vector) | Unit: Unitless
s = 0.5

## User-Inputted Data on Hydrogen State

# Principal Quantum Number/Shell Designator: Theoretically any natural number for n works but I have constrained the values to the highest named spectrum (Humphreys Series)
n = int(input("What is the principal quantum number n? ")) 
if n > 11:
    raise Exception(ValueError("Invalid or Unnatural Excited State"))
# Azimuthal Quantum Number/Orbital Angular Momentum/Subshell Designator: Has to always be within the range of 0 and n - 1
l = int(input("What is the azimuthal quantum number l? ")) 
if l >= n or l < 0:
    raise Exception(ValueError("Invalid Subshell"))

# Total Angular Momentum Quantum Number: Has to be at or between l - s to l + s in integer steps
j = np.arange(np.abs(l - s), l + s + 1, 1)
# Display the options to the user
print("Please choose an option from the possible values of total angular momentum quantum number j: ")
for i, option in enumerate(j, start = 1):
    print(f"{i}. {option}")
# Get the user's choice
choice = int(input("Enter the number of your choice: "))
# Validate the choice and display the selected option
if 1 <= choice <= len(j):
    print(f"Selected j: {j[choice - 1]}")
else:
    raise Exception(ValueError("Invalid choice. Please try again."))
j = j[choice - 1]

# Secondary Total Angular Momentum Quantum Number: Has to be at or between -j to j in steps of 1
m = np.arange(-j, j + 1, 1)
# Display the options to the user
print("Please choose an option from the possible values of secondary total angular momentum quantum number m: ")
for i, option in enumerate(m, start = 1):
    print(f"{i}. {option}")
# Get the user's choice
choice = int(input("Enter the number of your choice: "))
# Validate the choice and display the selected option
if 1 <= choice <= len(m):
    print(f"Selected m: {m[choice - 1]}")
else:
    raise(Exception(ValueError("Invalid choice. Please try again.")))
m = m[choice - 1]

# Tell the user if their choices corresponds to a spin-up or spin-down electron
if m > 0.0:
    print("The electron is spin-up")
    spin_vector = "spin-up"
if m < 0.0:
    print("The electron is spin-down")
    spin_vector = "spin-down"

# Determine integer k value based on state
if j == round((l + 0.5), 1): # l + (1 / 2)
    k = int(-j - 0.5) # -j - (1 / 2)
if j == round((l - 0.5), 1): # l - (1 / 2)
    k = int(j + 0.5) # j + (1 / 2)
if j != round((l + 0.5), 1) and j != round((l - 0.5), 1):
    k = 0
if float(k) == 0.0:
    raise(Exception((ValueError("Invalid value for k."))))
    sys.exit()
# Tell the user what the integer k value is, for reference
print("Integer k value: " + str(k))

## Setting up solution to Dirac equation for Hydrogen atom

# Gamma always real positive unless Z is at least 137.
gamma = math.sqrt(k ** 2 - proton_number ** 2 * fine_structure_constant ** 2)

# Radius value to set for to measure Hydrogen atom electron | Units: m
r = 6 * 10 ** (-9)

# Two cases needed, one for the highest j value for a given n and one for most states
if j == (n - 0.5):
    # Rest Energy of State; difference in rest energy between two states is the atomic energy level | Units: J
    E = (gamma / n) * mu * speed_of_light ** 2
    C = ((proton_number * fine_structure_constant) / (n)) * ((mu * speed_of_light ** 2) / (reduced_planck * speed_of_light))
    # Set r as dependent variable
    def rho(radius):
        return (2 * C * np.array(radius))
    A = (1 / cmath.sqrt(2 * n * (n + gamma))) * cmath.sqrt(C / (gamma * spec.gamma(2 * gamma)))
    g = A * (n + gamma) * rho(r) ** gamma * np.exp(-rho(r) / 2)
    f = A * proton_number * fine_structure_constant * rho(r) ** gamma * np.exp(-rho(r) / 2)
if j != (n - 0.5):
    # Rest Energy of State; difference in rest energy between two states is the atomic energy level | Units: J
    E = mu * speed_of_light ** 2 * (1 + ((proton_number * fine_structure_constant) / (n - np.abs(k) + gamma)) ** 2) ** (-1 / 2)
    C = cmath.sqrt(mu ** 2 * speed_of_light ** 4 - E ** 2) / (reduced_planck * speed_of_light)
    # r as dependent variable
    def rho(radius):
        return (2 * C * np.array(radius))
    A = (1 / np.sqrt(2 * k * (k - gamma))) * np.sqrt((C / (n - np.abs(k) + gamma)) * (math.factorial(n - np.abs(k) - 1) / (spec.gamma(n - np.abs(k) + 2 * gamma + 1))) * (1 / 2) * (((E * k) / (gamma * mu * speed_of_light ** 2)) ** 2 + ((E * k) / (gamma * mu * speed_of_light ** 2))))
    genLag = spec.genlaguerre((n - np.abs(k) - 1), round((2 * gamma + 1).real, 1))
    g = A * rho(r) ** gamma * np.exp(-rho(r) / 2) * (proton_number * fine_structure_constant * rho(r) * genLag(rho(r)) + (gamma - k) * (((gamma * mu * speed_of_light ** 2) - (k * E)) / (reduced_planck * speed_of_light * C)) * genLag(rho(r)))
    f = A * rho(r) ** gamma * np.exp(-rho(r) / 2) * ((gamma - k) * rho(r) * genLag(rho(r)) + proton_number * fine_structure_constant * (((gamma * mu * speed_of_light ** 2) - (k * E)) / (reduced_planck * speed_of_light * C)) * genLag(rho(r)))

## Creating solution space

# Create a grid of theta and phi values for the solution space
theta, phi = np.mgrid[0:2 * cmath.pi:100j, 0:2 * cmath.pi:100j]
psi = []
# Input solution for each value in theta, phi grid
# If the solution leads to an asymptotic solution, it is then set to 0 
for i in range(len(np.ndarray.flatten(theta))):
    if k > 0:
        try:
            Y_k_mmin = (-1) ** (m - 0.5) * math.sqrt(((2 * k + 1) / (4 * np.pi)) * (math.factorial(int(k - (m - 0.5))) / math.factorial(int(k + (m - 0.5))))) * spec.lpmv((m - 0.5), k, np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m - 0.5) * np.ndarray.flatten(phi)[i])
        except ValueError:
            Y_k_mmin = 0   
        try:
            Y_k_mplus = (-1) ** (m + 0.5) * math.sqrt(((2 * k + 1) / (4 * np.pi)) * (math.factorial(int(k - (m + 0.5))) / math.factorial(int(k + (m + 0.5))))) * spec.lpmv((m + 0.5), k, np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m + 0.5) * np.ndarray.flatten(phi)[i])
        except:
            Y_k_mplus = 0
        try:
            Y_negk_mmin = (-1) ** (m - 0.5) * math.sqrt(((2 * (k - 1) + 1) / (4 * np.pi)) * (math.factorial(int((k - 1) - (m - 0.5))) / math.factorial(int((k - 1) + (m - 0.5))))) * spec.lpmv((m - 0.5), (k - 1), np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m - 0.5) * np.ndarray.flatten(phi)[i])
        except: 
            Y_negk_mmin = 0
        try:
            Y_negk_mplus = (-1) ** (m + 0.5) * math.sqrt(((2 * (k - 1) + 1) / (4 * np.pi)) * (math.factorial(int((k - 1) - (m + 0.5))) / math.factorial(int((k - 1) + (m + 0.5))))) * spec.lpmv((m + 0.5), (k - 1), np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m + 0.5) * np.ndarray.flatten(phi)[i])
        except:
            Y_negk_mplus = 0
    if k < 0:
        try:
            Y_k_mmin = (-1) ** (m - 0.5) * math.sqrt(((2 * (-k - 1) + 1) / (4 * np.pi)) * (math.factorial(int((-k - 1) - (m - 0.5))) / math.factorial(int((-k - 1) + (m - 0.5))))) * spec.lpmv((m - 0.5), (-k - 1), np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m - 0.5) * np.ndarray.flatten(phi)[i])
        except:
            Y_k_mmin = 0
        try:
            Y_k_mplus = (-1) ** (m + 0.5) * math.sqrt(((2 * (-k - 1) + 1) / (4 * np.pi)) * (math.factorial(int((-k - 1) - (m + 0.5))) / math.factorial(int((-k - 1) + (m + 0.5))))) * spec.lpmv((m + 0.5), (-k - 1), np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m + 0.5) * np.ndarray.flatten(phi)[i])
        except:
            Y_k_mplus = 0
        try:
            Y_negk_mmin = (-1) ** (m - 0.5) * math.sqrt(((2 * -k + 1) / (4 * np.pi)) * (math.factorial(int(-k - (m - 0.5))) / math.factorial(int(-k + (m - 0.5))))) * spec.lpmv((m - 0.5), -k, np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m - 0.5) * np.ndarray.flatten(phi)[i])
        except:
            Y_negk_mmin = 0
        try:
            Y_negk_mplus = (-1) ** (m + 0.5) * math.sqrt(((2 * -k + 1) / (4 * np.pi)) * (math.factorial(int(-k - (m + 0.5))) / math.factorial(int(-k + (m + 0.5))))) * spec.lpmv((m + 0.5), -k, np.cos(np.ndarray.flatten(theta)[i])) * cmath.exp(1j * (m + 0.5) * np.ndarray.flatten(phi)[i])
        except:
            Y_negk_mplus = 0

    # Organize solution for each grid point into a spinor (really it is a vector matrix but will be treated as a spinor)
    psi_unfinished = np.array([[(g / r) * cmath.sqrt((k + 0.5 - m) / (2 * k + 1)) * Y_k_mmin],
                [-(g / r) * np.sign(k) * cmath.sqrt((k + 0.5 + m) / (2 * k + 1)) * Y_k_mplus],
                [((1j * f) / r) * cmath.sqrt((-k + 0.5 - m) / (2 * -k + 1)) * Y_negk_mmin],
                [((-1j * f) / r) * np.sign(k) * cmath.sqrt((-k + 0.5 + m) / (2 * -k + 1)) * Y_negk_mplus]])
    # Taking modulus magnitude of spinor for observable value
    psi.append(np.sqrt(abs(psi_unfinished[0]) ** 2 + abs(psi_unfinished[1]) ** 2 + abs(psi_unfinished[2]) ** 2 + abs(psi_unfinished[3]) ** 2))
    
## Visualization of Dirac solution for Hydrogen

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.set_aspect("equal")

# Transform coordinates from theta, phi to x, y, z
x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)

# 3D plot is color-coded, so it actually represents 4 parameters instead of 3 with the 4th parameter being the observable probability (psi squared)
intensities = (np.abs(psi) / np.max(np.abs(psi))) ** 2

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

# Setting axes and plotting for Hydrogen atom Dirac solution
scatter = ax.scatter(x, y, z, c = intensities, cmap = 'viridis')
plt.title("Probability of Finding Electron around Hydrogen Atom (Dirac)\nn = " + str(n) + "\nl = " + str(l) + "\nj = " + str(j) + "\nm = " + str(m) + "\nk = " + str(k) + "\nr = " + str(r) + " m" + "\nSpin: " + spin_vector)
ax.set_xlabel("          x distance (m)")
ax.xaxis.labelpad = -15
# Add a color bar
colorbar = plt.colorbar(scatter)
colorbar.set_label('Relative Probability')
plt.show()
