# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:49:32 2024

@author: tacco
"""

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
import sys

e_0 = 8.854187818814 * 10 ** (-12) # F⋅m**(−1) Vacuum Permitivity/Permitivity of Free Space/Electric Constant
planck = 6.62607015 * 10 ** (-34)
reduced_planck = planck / (2 * cmath.pi)
m_e = 9.109 * 10 ** (-31) # kg
m_p = 1.6726219 * 10 ** (-27) # kg
mu = (m_e * m_p)/(m_e + m_p) # Accounts for masses of system
elementary_charge = 1.602176634 * 10 ** (-19) # Coulombs
non_reduced_a_0 = (4 * cmath.pi * e_0 * reduced_planck ** 2) / (elementary_charge ** 2 * m_e)
a_0 = (4 * np.pi * e_0 * reduced_planck ** 2)/(mu * elementary_charge ** 2) # Reduced Bohr Radius
proton_number = 1
speed_of_light = 299792458 # m/s
electric_constant = 8.8541878188 * 10 ** (-12) # F / m
fine_structure_constant = elementary_charge ** 2 / (4 * cmath.pi * electric_constant * reduced_planck * speed_of_light)

s = 0.5

n = int(input("What is the principal quantum number? ")) # Excited State, 4 is max?
l = int(input("What is the azimuthal quantum number? ")) # 0 -> n-1, orbital angular momentum
# electron_spin = input("Is the electron spin-up or spin-down? (Answer 'up' or 'down') \n")
# if electron_spin == "up":
#     electron_spin = 0.5
# if electron_spin == "down":
#     electron_spin = -0.5

j = np.arange(np.abs(l - s), l + s + 1, 1)

# Display the options to the user
print("Please choose an option from the possible values of j:")
for i, option in enumerate(j, start=1):
    print(f"{i}. {option}")
# Get the user's choice
choice = int(input("Enter the number of your choice: "))
# Validate the choice and display the selected option
if 1 <= choice <= len(j):
    print(f"j selected: {j[choice - 1]}")
else:
    print("Invalid choice. Please try again.")
j = j[choice - 1]

m = np.arange(-j, j + 1, 1)
# Display the options to the user
print("Please choose an option from the possible values of m:")
for i, option in enumerate(m, start=1):
    print(f"{i}. {option}")
# Get the user's choice
choice = int(input("Enter the number of your choice: "))
# Validate the choice and display the selected option
if 1 <= choice <= len(m):
    print(f"m selected: {m[choice - 1]}")
else:
    print("Invalid choice. Please try again.")
m = m[choice - 1]
if m > 0.0:
    print("The electron is spin-up")
if m < 0.0:
    print("The electron is spin-down")

if j == round((l + 0.5), 1): # l + (1 / 2)
    k = int(-j - 0.5) # -j - (1 / 2)
if j == round((l - 0.5), 1): # l - (1 / 2)
    k = int(j + 0.5) # j + (1 / 2)
if j != round((l + 0.5), 1) and j != round((l - 0.5), 1):
    k = 0
if float(k) == 0.0:
    print("Invalid value for k")
    sys.exit()

print("k: " + str(k))


# Gamma always real positive unless Z is at least 137.
gamma = math.sqrt(k ** 2 - proton_number ** 2 * fine_structure_constant ** 2)


r = 6 * 10 ** (-9)

if j == (n - 0.5):
    E = (gamma / n) * mu * speed_of_light ** 2
    C = ((proton_number * fine_structure_constant) / (n)) * ((mu * speed_of_light ** 2) / (reduced_planck * speed_of_light))
    # r as dependent variable
    def rho(radius):
        return (2 * C * np.array(radius))
    A = (1 / cmath.sqrt(2 * n * (n + gamma))) * cmath.sqrt(C / (gamma * spec.gamma(2 * gamma)))
    g = A * (n + gamma) * rho(r) ** gamma * np.exp(-rho(r) / 2)
    f = A * proton_number * fine_structure_constant * rho(r) ** gamma * np.exp(-rho(r) / 2)

if j != (n - 0.5):
    E = mu * speed_of_light ** 2 * (1 + ((proton_number * fine_structure_constant) / (n - np.abs(k) + gamma)) ** 2) ** (-1 / 2)
    C = cmath.sqrt(mu ** 2 * speed_of_light ** 4 - E ** 2) / (reduced_planck * speed_of_light)
    # r as dependent variable
    def rho(radius):
        return (2 * C * np.array(radius))
    A = (1 / np.sqrt(2 * k * (k - gamma))) * np.sqrt((C / (n - np.abs(k) + gamma)) * (math.factorial(n - np.abs(k) - 1) / (spec.gamma(n - np.abs(k) + 2 * gamma + 1))) * (1 / 2) * (((E * k) / (gamma * mu * speed_of_light ** 2)) ** 2 + ((E * k) / (gamma * mu * speed_of_light ** 2))))
    genLag = spec.genlaguerre((n - np.abs(k) - 1), round((2 * gamma + 1).real, 1))
    g = A * rho(r) ** gamma * np.exp(-rho(r) / 2) * (proton_number * fine_structure_constant * rho(r) * genLag(rho(r)) + (gamma - k) * (((gamma * mu * speed_of_light ** 2) - (k * E)) / (reduced_planck * speed_of_light * C)) * genLag(rho(r)))
    f = A * rho(r) ** gamma * np.exp(-rho(r) / 2) * ((gamma - k) * rho(r) * genLag(rho(r)) + proton_number * fine_structure_constant * (((gamma * mu * speed_of_light ** 2) - (k * E)) / (reduced_planck * speed_of_light * C)) * genLag(rho(r)))



theta, phi = np.mgrid[0:2 * cmath.pi:100j, 0:2 * cmath.pi:100j]
psi = []
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


    psi_unfinished = np.array([[(g / r) * cmath.sqrt((k + 0.5 - m) / (2 * k + 1)) * Y_k_mmin],
                [-(g / r) * np.sign(k) * cmath.sqrt((k + 0.5 + m) / (2 * k + 1)) * Y_k_mplus],
                [((1j * f) / r) * cmath.sqrt((-k + 0.5 - m) / (2 * -k + 1)) * Y_negk_mmin],
                [((-1j * f) / r) * np.sign(k) * cmath.sqrt((-k + 0.5 + m) / (2 * -k + 1)) * Y_negk_mplus]])

    psi.append(cmath.sqrt(abs(psi_unfinished[0]) ** 2 + abs(psi_unfinished[1]) ** 2 + abs(psi_unfinished[2]) ** 2 + abs(psi_unfinished[3]) ** 2))
    




fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect("equal")


x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)


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
