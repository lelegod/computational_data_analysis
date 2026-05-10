#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:14:20 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Number of data points per source
num_samples = 100000

# We will test summing this many independent sources
# 1 source (pure), 2 sources, 5 sources, and 20 sources
num_sources_to_mix = [1, 2, 5, 20]

# Setup the matplotlib figure with a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, n_sources in enumerate(num_sources_to_mix):
    
    # 1. GENERATE INDEPENDENT NON-GAUSSIAN SOURCES
    # We use a Uniform distribution because it is completely flat 
    # (very non-Gaussian / negative kurtosis).
    # Shape: (n_sources, num_samples)
    sources = np.random.uniform(low=-1.0, high=1.0, size=(n_sources, num_samples))
    
    # 2. MIX THE SOURCES
    # Summing the sources together (mimicking the mixing matrix A)
    mixed_signal = np.sum(sources, axis=0)
    
    # 3. STANDARDIZE THE MIXTURE
    # To easily compare it to a standard Normal distribution (mean 0, std 1),
    # we center and scale the mixed signal.
    mixed_signal_standardized = (mixed_signal - np.mean(mixed_signal)) / np.std(mixed_signal)
    
    ax = axes[i]
    
    # Plot the histogram of the standardized mixed signal
    ax.hist(mixed_signal_standardized, bins=100, density=True, alpha=0.6, 
            color='royalblue', label='Mixed Signal')
    
    # Plot a perfect Standard Normal (Gaussian) bell curve for reference
    x_axis = np.linspace(-4, 4, 1000)
    ax.plot(x_axis, norm.pdf(x_axis), color='darkred', linewidth=2, 
            label='Standard Gaussian')
    
    # Formatting the plot
    ax.set_title(f'Sum of {n_sources} Non-Gaussian Source(s)')
    ax.set_xlim([-4, 4])
    ax.set_ylim([0, 0.5])
    if i == 0:
        ax.legend()

plt.suptitle('Demonstrating the Central Limit Theorem for ICA\n(Why mixtures are "more Gaussian" than pure sources)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.88) # Adjust top to make room for suptitle
plt.show()