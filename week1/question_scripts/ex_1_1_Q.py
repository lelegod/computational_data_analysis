#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:43:08 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Learning Objective: Observe coefficient instability and its link to collinearity.

# --- SECTION 1: Parameters ---
np.random.seed(42) 
n_samples = 100
n_simulations = 500
sigma = 1.0
rho = 0.98
beta_true = np.array([2, 0])
x_test = np.array([[1, 1]])
target_val = (x_test @ beta_true)[0]

def generate_data(n, rho, sigma):
    '''
    TASK: Generate synthetic data.
    1. Create X with two features correlated by rho (use multivariate_normal).
    2. Generate y = X @ beta_true + noise.
    '''
    # YOUR CODE HERE
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    X = np.random.multivariate_normal(mean, cov, n)
    y = X @ beta_true + np.random.normal(0, sigma, n)
    return X, y

# --- SECTION 2: Simulation ---
# TASK: Run a loop for n_simulations.
all_betas = []
all_preds = []

print('Running simulations...')
for _ in range(n_simulations):
    X, y = generate_data(n_samples, rho, sigma)
    model = LinearRegression().fit(X, y)

    # Store results
    all_betas.append(model.coef_)
    all_preds.append(model.predict(x_test)[0])

all_betas = np.array(all_betas)
all_preds = np.array(all_preds)

# --- SECTION 3: Calculations ---
# TASK: Calculate the following metrics:
# 1. Mean and Variance of the estimated coefficients (betas).
# 2. The Bias^2 at x_test.
# 3. The Variance at x_test.

mean = np.mean(all_betas, axis=0)
var = np.var(all_betas, axis=0)

bias_sq = (np.mean(all_preds) - target_val)**2
variance = np.var(all_preds)
epe = bias_sq + variance + sigma**2

print(f"True betas: {beta_true}")
print(f"Mean of estimated betas: {mean}")
print(f"variance of estimated betas: {var}")
print(f"bias^2: {bias_sq:4.3f}")
print(f"variance: {variance:4.3f}")
print(f"EPE: {epe:4.3f}")

# --- SECTION 4: Visualization ---
# TASK: Create a histogram of estimated Beta 1 and Beta 2.
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(all_betas[:, 0], bins=30, alpha=0.5, label="Beta1 (True=2)")
ax.hist(all_betas[:, 1], bins=30, alpha=0.5, label="Beta2 (True=0)")
ax.axvline(beta_true[0], linestyle='--', color='blue')
ax.axvline(beta_true[1], linestyle='--', color='orange')
ax.set(title=rf'Instability of OLS with $\rho=${rho}')
ax.legend()
plt.show()