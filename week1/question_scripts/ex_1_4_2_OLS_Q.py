#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:05:30 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe OLS breakdown in the Overfitting Regime (n approx m).

# 1. Load and Expand Data
print('Loading Wine Quality data...')
# TASK: Load data and use PolynomialFeatures(degree=2) to expand features to m=77.
X, y = fetch_openml(name='wine-quality-red', as_frame=True, return_X_y=True)
X = StandardScaler().fit_transform(PolynomialFeatures(degree=2, include_bias=False).fit_transform(X))
y = y.values.astype(float)

# 2. Unstable Split
# TASK: Use a very small training fraction (e.g., train_size=0.05 or n=80) 
# to make n approach m.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# 3. Fit OLS
# TASK: Fit LinearRegression on the expanded features.
ols = LinearRegression().fit(X_train, y_train)

# 4. Evaluate
# TASK: Calculate and print Training MSE and Test MSE.
# Note if the test error "explodes" compared to the stable regime.
train_mse = mean_squared_error(y_train, ols.predict(X_train))
test_mse = mean_squared_error(y_test, ols.predict(X_test))

# 5. Visualization
# TASK: Create a bar chart (use log scale for y-axis if error is very high).
plt.figure(figsize=(10, 6))
plt.bar(['Training error', 'Test error'], [train_mse, test_mse], color=['blue', 'red'], alpha=0.7)
plt.grid(axis='y', alpha=0.3)
plt.yscale('log')
plt.ylabel('Mean squared error')
plt.title(r'OLS with instability ($m=77$)')
plt.show()