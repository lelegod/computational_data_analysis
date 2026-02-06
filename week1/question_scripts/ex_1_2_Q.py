#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:45:31 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Learning Objective: Quantify the Bias-Variance tradeoff using Ridge lambda.

# --- SECTION 1: Parameters ---
np.random.seed(42)
n_train = 100
n_test = 500
n_simulations = 500
sigma = 1.0          # Noise level
beta_true = np.array([2, 0])
x_eval = np.array([[1, 1]])
target_val = (x_eval @ beta_true)[0]
lambdas = np.logspace(-2, 5, 25)

# Fixed X for consistent theoretical/empirical comparison
mean = [0, 0]
cov = [[1, 0.98], [0.98, 1]] # High correlation
X_train = np.random.multivariate_normal(mean, cov, n_train)
X_test = np.random.multivariate_normal(mean, cov, n_test)

# --- SECTION 2: Simulation ---
# TASK: Iterate through lambdas and simulations.
# 1. Generate y = X_fixed @ beta_true + noise.
# 2. Fit Ridge(alpha=l).
# 3. Predict at x_eval.
# 4. Calculate empirical Bias^2 and Variance.

bias_sq = []
variance = []
mean_train_mse = []
mean_test_mse = []

for l in lambdas:
    all_preds = []
    train_mse = []
    test_mse = []

    for _ in range(n_simulations):
        y_train = X_train @ beta_true + np.random.normal(0, sigma, n_train)
        y_test = X_test @ beta_true + np.random.normal(0, sigma, n_test)

        model = Ridge(alpha=l).fit(X_train, y_train)

        all_preds.append(model.predict(x_eval)[0])
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_mse.append(mean_squared_error(y_train, train_preds))
        test_mse.append(mean_squared_error(y_test, test_preds))

    bias_sq.append((np.mean(all_preds) - target_val)**2)
    variance.append(np.var(all_preds))
    mean_train_mse.append(np.mean(train_mse))
    mean_test_mse.append(np.mean(test_mse))

bias_sq = np.array(bias_sq)
variance = np.array(variance)

print('Simulating Ridge complexity sweep...')
# YOUR CODE HERE


# --- SECTION 3: Visualization ---
# TASK: Plot Bias^2, Variance, and Total (Bias^2 + Var) vs Lambdas.
# Remember to invert the x-axis for complexity.

plt.figure(figsize=(10, 6))
plt.plot(lambdas, bias_sq, label=r'$Bias^2$', color='green', linestyle='--')
plt.plot(lambdas, variance, label='Variance', color='orange', linestyle='--')
plt.axhline(sigma**2, label=r'Noise $\rho^2$', color='gray', linestyle=':')
plt.plot(lambdas, mean_train_mse, label='Mean Train MSE', color='blue')
plt.plot(lambdas, mean_test_mse, label='Mean Test MSE', color='red')
plt.axvline(lambdas[np.argmin(mean_test_mse)], label='Sweet Spot', linestyle='-.', color='black')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.legend()
plt.xlabel(r'lambda $\lambda$')
plt.ylabel('Error')
plt.tight_layout()
plt.show()