#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:21:40 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D

# 1. Generate Non-Linear Data (Circles)
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)

# 2. Visualize in 2D - Impossible to separate with a straight line
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', edgecolors='k')
ax1.set_title("Original 2D Space: Non-Linear")

# 3. Manual Kernel Mapping (Radial-like expansion)
# We add a 3rd dimension: Z = exp(-(X^2 + Y^2))
# This is a manual version of what an RBF kernel does
z = np.exp(-(X[:, 0]**2 + X[:, 1]**2))

# 4. Visualize in 3D - Now it is linearly separable!
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:, 0], X[:, 1], z, c=y, cmap='winter', edgecolors='k')

# Draw a conceptual separating plane
xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
zz = np.zeros_like(xx) + 0.7  # A flat plane at Z=0.7
ax2.plot_surface(xx, yy, zz, alpha=0.2, color='red')

ax2.set_title("Mapped 3D Space: Linearly Separable")
ax2.set_zlabel("Exp-Distance (New Feature)")
plt.tight_layout()
plt.show()