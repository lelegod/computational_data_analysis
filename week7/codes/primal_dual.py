#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:53:13 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- PROBLEM DEFINITION ---
# Primal: Minimize f(x) = x^2 subject to x >= 2
# Lagrangian: L(x, lambda) = x^2 - lambda(x - 2)
# p* = 4 (at x=2), d* = 4 (at lambda=4)

def lagrangian(x, lam):
    return x**2 - lam * (x - 2)

# Generate data
x_range = np.linspace(0.5, 4.5, 50)
lam_range = np.linspace(0, 8, 50)
X, LAM = np.meshgrid(x_range, lam_range)
Z = lagrangian(X, LAM)

# Optimal points
x_star, lam_star, p_star = 2, 4, 4

# Create Figure
fig = plt.figure(figsize=(16, 8))

# --- PLOT 1: 3D SADDLE POINT VISUALIZATION ---
ax1 = fig.add_subplot(121, projection='3d')

# Plot the Lagrangian Surface
surf = ax1.plot_surface(X, LAM, Z, cmap='viridis', alpha=0.8, antialiased=True)

# Highlight the Primal Path (Fix lambda=lam_star, vary x)
ax1.plot(x_range, [lam_star]*50, lagrangian(x_range, lam_star), color='red', linewidth=4, label='Primal Path (Minimizing x)')

# Highlight the Dual Path (Fix x=x_star, vary lambda)
ax1.plot([x_star]*50, lam_range, lagrangian(x_star, lam_range), color='blue', linewidth=4, label='Dual Path (Maximizing λ)')

# Mark the Saddle Point (The Optimum)
ax1.scatter(x_star, lam_star, p_star, color='white', s=200, edgecolors='black', depthshade=False, label='Saddle Point (Optimum)')

ax1.set_title("The Lagrangian Saddle Point", fontsize=16, pad=20)
ax1.set_xlabel("x (Primal Variable)")
ax1.set_ylabel("lambda (Dual Variable)")
ax1.set_zlabel("L(x, lambda)")
ax1.view_init(elev=20, azim=-135)
ax1.legend()

# --- PLOT 2: 2D CROSS-SECTION (LOWER BOUNDS) ---
ax2 = fig.add_subplot(122)

# Plot f(x)
ax2.plot(x_range, x_range**2, 'k--', alpha=0.5, label='Original f(x) = x^2')

# Plot L(x, lambda) for different lambda values
lambdas_to_show = [0, 2, 4, 6]
colors = ['#cbd5e1', '#94a3b8', '#3b82f6', '#1e40af']

for l, c in zip(lambdas_to_show, colors):
    label = f"L(x, λ={l})"
    if l == 4: label += " (Optimal Dual)"
    ax2.plot(x_range, lagrangian(x_range, l), color=c, linewidth=2 if l!=4 else 4, label=label)

# Feasible region highlight
ax2.axvspan(2, 4.5, color='green', alpha=0.1, label='Feasible Region (x >= 2)')
ax2.scatter(x_star, p_star, color='red', s=100, zorder=5, label='Primal Minimum')

ax2.set_title("Dual Functions as 'Supports'", fontsize=16)
ax2.set_xlabel("x")
ax2.set_ylabel("Objective Value")
ax2.set_ylim(-2, 15)
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()