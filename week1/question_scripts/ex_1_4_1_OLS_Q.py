#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:06:27 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe OLS behavior in the Stable Regime (n >> m).

# 1. Load Data
print('Loading Wine Quality data...')
# TASK: Load wine-quality-red, scale features, and cast target to float.
X, y = fetch_openml(name='wine-quality-red', as_frame=True, return_X_y=True)
X = StandardScaler().fit_transform(X)
y = y.values.astype(float)

# 2. Stable Split
# TASK: Create a split with 80% training data (Large n, Small m).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 3. Fit OLS
# TASK: Use LinearRegression.
ols = LinearRegression().fit(X_train, y_train)

# 4. Evaluate
# TASK: Calculate and print Training MSE and Test MSE.
# Compare the generalization gap.
train_mse = mean_squared_error(y_train, ols.predict(X_train))
test_mse = mean_squared_error(y_test, ols.predict(X_test))
generalization_gap = test_mse - test_mse
print(f'Generalization gap: {generalization_gap:4.3f}')

# 5. Visualization
# TASK: Create a bar chart comparing Train vs Test MSE.
plt.figure(figsize=(10, 6))
plt.bar(['Training error', 'Test error'], [train_mse, test_mse], color=['blue', 'red'], alpha=0.7)
plt.grid(axis='y', alpha=0.3)
plt.ylabel('Mean squared error')
plt.title('OLS')
plt.show()