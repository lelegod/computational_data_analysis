#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:35:09 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

def generate_audio_data():
    """
    Generates a synthetic 'spectrogram' matrix X.
    We create two distinct spectral 'atoms' (e.g., a low-freq and high-freq instrument).
    """
    time = np.linspace(0, 1, 100)
    freqs = np.linspace(0, 50, 50)
    
    # Basis W: Two spectral signatures (Normal distributions at different freqs)
    w1 = np.exp(-(freqs - 10)**2 / 10)  # Low frequency signature
    w2 = np.exp(-(freqs - 40)**2 / 10)  # High frequency signature
    W_true = np.vstack([w1, w2]).T      # Shape: (50, 2)
    
    # Activations H: When these signatures occur
    h1 = np.where(time < 0.5, 1.0, 0.1) # First half dominant
    h2 = np.where(time > 0.4, 1.0, 0.1) # Overlapping in middle
    H_true = np.vstack([h1, h2])        # Shape: (2, 100)
    
    # Construct X = W*H + noise
    X = np.dot(W_true, H_true)
    X += np.random.uniform(0, 0.05, X.shape) # Add non-negative noise
    
    return X, W_true, H_true

def manual_nmf_multiplicative_update(X, k, max_iter=500):
    """
    Foundational implementation of Lee & Seung (1999) Multiplicative Updates.
    Minimizes Frobenius norm: ||X - WH||^2
    """
    I, J = X.shape
    W = np.random.rand(I, k)
    H = np.random.rand(k, J)
    
    eps = 1e-9 # Prevent division by zero
    
    for i in range(max_iter):
        # Update H
        H = H * (np.dot(W.T, X) / (np.dot(np.dot(W.T, W), H) + eps))
        # Update W
        W = W * (np.dot(X, H.T) / (np.dot(W, np.dot(H, H.T)) + eps))
        
    return W, H

# --- Main Execution ---
X, W_true, H_true = generate_audio_data()

# 1. Use Sklearn for robust implementation
model = NMF(n_components=2, init='random', random_state=0, max_iter=1000)
W_sk = model.fit_transform(X)
H_sk = model.components_

# 2. Use our manual implementation
W_man, H_man = manual_nmf_multiplicative_update(X, k=2)

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].imshow(X, aspect='auto', cmap='viridis')
axes[0].set_title("Input Spectrogram (X)")


axes[1].plot(W_sk[:, 0], label="Extracted Basis 1", color='red')
axes[1].plot(W_sk[:, 1], label="Extracted Basis 2", color='blue')
axes[1].set_title("Learned Spectral Signatures - auto (W)")
axes[1].legend()

axes[2].plot(H_sk[0, :], label="Activation 1", color='red', alpha=0.6)
axes[2].plot(H_sk[1, :], label="Activation 2", color='blue', alpha=0.6)
axes[2].set_title("Time Activations (H)")
axes[2].legend()

plt.tight_layout()
plt.show()


# Visualization
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].imshow(X, aspect='auto', cmap='viridis')
axes[0].set_title("Input Spectrogram (X)")


axes[1].plot(W_man[:, 0], label="Extracted Basis 1", color='red')
axes[1].plot(W_man[:, 1], label="Extracted Basis 2", color='blue')
axes[1].set_title("Learned Spectral Signatures - man (W)")
axes[1].legend()

axes[2].plot(H_man[0, :], label="Activation 1", color='red', alpha=0.6)
axes[2].plot(H_man[1, :], label="Activation 2", color='blue', alpha=0.6)
axes[2].set_title("Time Activations (H)")
axes[2].legend()

plt.tight_layout()
plt.show()
print("NMF Demo Completed. Reconstructed error (Sklearn):", model.reconstruction_err_)