# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:57:30 2025

@author: tacco
"""

import numpy as np
from scipy.special import eval_genlaguerre, sph_harm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generate sample data (r, theta, phi) and target values
num_points = 1000

r_values = np.linspace(0.1, 5.0, num_points)  # Radial values
theta_values = np.linspace(0, np.pi, num_points)  # Polar angle
phi_values = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle

# Target function: you can modify this based on your requirements
y_true = np.cos(r_values) * np.sin(theta_values) * np.cos(phi_values)

# Define the 3F2 hypergeometric model: _3F2(r, theta, phi)
def hypergeometric_3F2(r, theta, phi, laguerre_coeffs, l, m):
    """
    Computes the 3F2 hypergeometric function.
    Parameters:
        r: Radial value
        theta: Polar angle
        phi: Azimuthal angle
        laguerre_coeffs: Coefficients for the generalized Laguerre polynomial
        l, m: Quantum numbers for spherical harmonics
    Returns:
        Value of the 3F2 function
    """
    # _0F_1(r): Generalized Laguerre polynomial
    laguerre_order = len(laguerre_coeffs) - 1
    laguerre_poly = eval_genlaguerre(laguerre_order, 0, r)
    
    # Y(theta, phi): Spherical harmonics
    spherical_harmonic = sph_harm(m, l, phi, theta)
    
    # Combine to compute _3F_2(r, theta, phi)
    return laguerre_poly * np.abs(spherical_harmonic)

# Loss function to optimize the coefficients
def loss_fn(params, r_values, theta_values, phi_values, y_true):
    try:
        # Extract Laguerre coefficients and quantum numbers
        laguerre_coeffs = params[:-2]
        l, m = params[-2], params[-1]

        # Ensure l and m are integers and satisfy the spherical harmonics constraints
        l = int(round(l))
        m = int(round(m))
        if l < 0 or abs(m) > l:
            return np.inf  # Return a large loss for invalid (l, m)

        # Compute predictions
        y_pred = np.array([hypergeometric_3F2(r, theta, phi, laguerre_coeffs, l, m)
            for r, theta, phi in zip(r_values, theta_values, phi_values)
        ])

        # Handle NaNs in predictions
        if np.any(np.isnan(y_pred)):
            return np.inf  # Return a large loss for invalid predictions

        # Return Mean Squared Error
        return np.mean((y_pred - y_true) ** 2)

    except Exception as e:
        # Catch any unexpected errors and return a large loss
        print(f"Error in loss function: {e}")
        return np.inf


# Initial guesses for optimization
initial_laguerre_coeffs = [1.0] * 3  # Starting with a 2nd-order Laguerre polynomial
initial_l, initial_m = 1, 0  # Spherical harmonics quantum numbers
initial_params = np.array(initial_laguerre_coeffs + [initial_l, initial_m])

# Optimize the parameters
result = minimize(
    loss_fn,
    initial_params,
    args=(r_values, theta_values, phi_values, y_true),
    method='Powell'
)

# Extract optimized parameters
optimized_laguerre_coeffs = result.x[:-2]
optimized_l, optimized_m = int(result.x[-2]), int(result.x[-1])

# Compute the fitted values
y_pred = np.array([
    hypergeometric_3F2(r, theta, phi, optimized_laguerre_coeffs, optimized_l, optimized_m)
    for r, theta, phi in zip(r_values, theta_values, phi_values)
])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(r_values, y_true, label='Target Function', color='blue')
plt.plot(r_values, y_pred, label='Fitted 3F2', color='red', linestyle='--')
plt.legend()
plt.xlabel('r')
plt.ylabel('Function Value')
plt.title('Fitting 3F2 to Target Function')
plt.show()

# Print the optimized parameters
print(f"Optimized Laguerre Coefficients: {optimized_laguerre_coeffs}")
print(f"Optimized l: {optimized_l}, m: {optimized_m}")
