#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:01:15 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# 1. Setup: Generate a "Big Data" set (200 points)
# We use a linear-separable cluster for clear visualization
X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.5)
y[y == 0] = -1  # Convert to standard SVM labels {-1, 1}

# 2. Train the Original SVM on the FULL dataset
# We use a large C to approximate a 'Hard Margin'
clf_full = SVC(kernel='linear', C=1000)
clf_full.fit(X, y)

# 3. Identify the "Elite" (The Support Vectors)
# These are the ONLY points that define the boundary according to KKT conditions
sv_indices = clf_full.support_
X_sv = X[sv_indices]
y_sv = y[sv_indices]

# 4. Train a NEW model using ONLY the Support Vectors
clf_subset = SVC(kernel='linear', C=1000)
clf_subset.fit(X_sv, y_sv)

# 5. Numerical Proof
# We compare the weights (beta) and the intercept (beta_0)
print("--- SVM SPARSITY PROOF ---")
print(f"Original Data size: {len(X)} points")
print(f"Support Vector count: {len(X_sv)} points")
print(f"Percentage of data irrelevant: {100 * (1 - len(X_sv)/len(X)):.1f}%")

# np.allclose checks if arrays are element-wise equal within a tolerance
betas_match = np.allclose(clf_full.coef_, clf_subset.coef_)
intercepts_match = np.allclose(clf_full.intercept_, clf_subset.intercept_)

print(f"\nDo the Hyperplane weights match? {'YES' if betas_match else 'NO'}")
print(f"Do the Intercepts match?        {'YES' if intercepts_match else 'NO'}")

# 6. Visual Proof
def plot_svc_decision_function(model, ax=None, plot_sv=True):
    """Plot the decision boundary and margins for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # Plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # Highlight support vectors
    if plot_sv:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=150, linewidth=1, facecolors='none', edgecolors='k',
                   label='Support Vectors')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left Plot: Full Data
ax1.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='winter')
plot_svc_decision_function(clf_full, ax1)
ax1.set_title(f"Model 1: Full Dataset (N={len(X)})")
ax1.legend()

# Right Plot: Only Support Vectors
ax2.scatter(X_sv[:, 0], X_sv[:, 1], c=y_sv, s=30, cmap='winter')
plot_svc_decision_function(clf_subset, ax2, plot_sv=False)
ax2.set_title(f"Model 2: Support Vectors ONLY (N={len(X_sv)})")

plt.show()