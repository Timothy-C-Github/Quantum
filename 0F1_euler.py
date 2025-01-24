# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:39:37 2025

@author: tacco
"""

## Import statements, scipy.special provides all the special functions needed.
import numpy as np
import scipy.optimize as opt
from scipy.special import hyp0f1
import matplotlib.pyplot as plt
import cmath

# Generate training data: x values starting at -1 and ends at 1; and corresponding sine values
x_values = np.linspace(-1, 1, 101)  # Range for x starting at -1
y_true = []
for i in x_values:
    y_true.append(cmath.exp(1j * i))  # Target: sine of x
y_true = np.array(y_true)

# Define the hypergeometric function model using a free parameter z(x) for each x and a fixed b
def hypergeometric_model(x, z, b):
    try:
        # Compute the 0F1 function with parameters (; b, z) where b and z are the free parameters
        result = hyp0f1([b], z) 
        
        # If the result is empty or invalid, return NaN
        if isinstance(result, np.ndarray) and result.size == 0:
            return np.nan
        return result.item()  # Return the scalar value from the result
    except Exception:
        # In case of an error, return NaN
        return np.nan

# Loss function for optimizing the z values (for a fixed b value)
def loss_fn_z(z_values, x_values, y_true, b):
    # Compute predictions using the fixed c value and the current z values
    y_pred = np.array([hypergeometric_model(x, z, b) for x, z in zip(x_values, z_values)])
    
    # Flatten y_pred to ensure it's 1D for compatibility with y_true
    y_pred = y_pred.flatten()

    # Ensure that the size of y_pred matches y_true
    if y_pred.size != y_true.size:
        return np.inf  # Return a large loss in case of mismatch
    
    # Return Mean Squared Error between predicted and true values
    return np.mean(abs(y_pred - y_true) ** 2) # The abs changes it from 2 values to 1 to train with

# Function to optimize the z(x) values for a given b
def optimize_z_for_b(b, x_values, y_true):
    # Initial guess: random initial guesses for z (one per data point)
    initial_z = np.linspace(-100.0, 100.0, len(x_values))  # Random initial guesses for z

    # Optimize the z values for this specific c value
    result = opt.minimize(loss_fn_z, initial_z, args=(x_values, y_true, b), method='Powell')
    
    # Return the optimized z values
    return result.x, result.fun  # Return the optimized z values and the final loss

# Function to find the best c by iterating over different c values
def find_best_b_and_z(x_values, y_true, b_values):
    best_b = None
    best_z = None
    best_loss = np.inf

    for b in b_values:
        # Optimize the z values for the current b
        optimized_z, current_loss = optimize_z_for_b(b, x_values, y_true)
        
        # If this is the best loss so far, store the results
        if current_loss < best_loss:
            best_b = b
            best_z = optimized_z
            best_loss = current_loss

    return best_b, best_z, best_loss

# Define a range of b values to test
b_values = np.linspace(-50, 50, 1001)  # Test different b values from 0.1 to 2.0

# Find the best b and corresponding z values
best_b, best_z, best_loss = find_best_b_and_z(x_values, y_true, b_values)

# After optimization, the fitted model's predictions using the optimized b and z values
y_pred = np.array([hypergeometric_model(x, z, best_b) for x, z in zip(x_values, best_z)])

# Flatten y_pred to ensure it's 1D for compatibility with y_true
y_pred = y_pred.flatten()

# Check if y_pred contains NaNs
if np.any(np.isnan(y_pred)):
    print("Warning: y_pred contains NaNs.")

# Plot the results
plt.plot(x_values, y_true, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.plot(x_values, y_pred, label='Optimized 0F1 Fit', color='red', linestyle='--', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Fitting 0F1 to sin(x)')
plt.show()

# Plot the optimized z values
plt.plot(x_values, best_z, label='Optimized z values', color='green', linestyle='-.', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('Optimized z')
plt.title('Optimized z Values for Each x')
plt.show()

# Print the best b value and z values
print(f"Best b Value: {best_b}")
print(f"Optimized z Values: {best_z}")
print(f"Best Loss of Fits: {best_loss}")

polynomials = []
covariances = []
# Obtain Relevant Polynomials up to Order 9 and Covariances
for i in range(10):
    polynomial, covariance  = np.polyfit(x_values, best_z, i, cov=True)
    polynomial = np.poly1d(polynomial)
    print("Representation of Polynomial Fit of " + str(i) + " Order: ")
    print(polynomial)
    print("Trace of Covariance Matrix for Polynomial Fit of " + str(i) + " Order: " + str(covariance.trace()))
