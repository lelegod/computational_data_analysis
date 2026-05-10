#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:53:25 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 1. Generate synthetic source signals (Non-Gaussian)
# In ICA, sources must be non-Gaussian for recovery to be possible.
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : Sine wave
s2 = np.sign(np.sin(3 * time))  # Signal 2 : Square wave (highly non-Gaussian)

S = np.c_[s1, s2]
S += 0.1 * np.random.normal(size=S.shape)  # Add minor noise

S /= S.std(axis=0)  # Standardize sources

# 2. Mix data
# Simulate two microphones picking up a linear mixture of the two sources
A = np.array([[1, 1], [0.5, 2]])  # The mixing matrix
X = np.dot(S, A.T)  # Observed mixtures

# 3. Apply ICA
ica = FastICA(n_components=2, random_state=42)
S_recovered = ica.fit_transform(X)  # Estimate the original sources
A_estimated = ica.mixing_  # Estimated mixing matrix

# 4. Visualization
plt.figure(figsize=(10, 8))

models = [S, X, S_recovered]
names = ['Original Sources (True)', 
         'Mixed Signals (Observed/Microphones)', 
         'ICA Recovered Signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color, alpha=0.7)

plt.tight_layout()
plt.show()

print("ICA Recovery complete.")
print("Original Mixing Matrix:\n", A)
print("Estimated Mixing Matrix:\n", A_estimated)