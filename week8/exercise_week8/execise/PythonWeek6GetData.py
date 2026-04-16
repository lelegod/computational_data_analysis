# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:21:21 2018
Gets all variables, used in the gui
@author: dnor
"""
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from pathlib import Path

def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)

def getVarsWeek6(control = 0):
    
    datapath = Path().cwd().parent / "Data"
        
    X = np.loadtxt(datapath / "faces.csv", delimiter =",")
    conlist = np.loadtxt(datapath / "conlist.csv", delimiter = ",").astype(int)
    # Conlist is stupid and made with "only" matlab in mind, retract 1 to start from 0
    conlist[:,0:2] -= 1
    
    n, p = np.shape(X)
    
    mu = np.mean(X, axis = 0)
    
    # center the data
    Xc = X - np.ones((n,1))*mu
    
    # Compute PCA as an eigenvalue analysis of the covariance matrix
    Eval, Evec = np.linalg.eig(np.cov(X.T))
    
    # Sort while still keeping the imaginary part, therefor lexicographic sorting
    # Also, descending sort
    sortIndex = np.argsort(Eval)[::-1] 
    
    Eval = Eval[sortIndex].astype(np.float64) # Away with imaginary part (~0) and order
    Evec = Evec[:, sortIndex].astype(np.float64)
    
    # Discard strictly none-positive eigenvalues modes
    Eval = Eval[np.where(Eval > 1e-9)]
    Evec = Evec[:, 0: len(Eval)]
    
    u, d, v = np.linalg.svd(Xc)
    
    # calculate the variances
    # keep only modes correspoding to strictly positive singular values
    d = d[np.where(d > 1e-9)]
    k = len(d)
    u = u[:, :k]
    v = v[:k, :].T # Matrix that is returned from svd is ordered differently, therefor different slicing
    
         # Assign PCA
    L = v # the loading
    S = np.matmul(u, np.diag(d)) # The scores
    sigma2 = d ** 2 / n
    
    if control == 1: # Threshold it
        L[np.where(np.abs(L) < 0.15)] = 0
        S = np.matmul(Xc, L)
        sigma2  = np.var(S, axis = 0)        
        
    elif control == 2: #Verimax
        L = varimax(L[:, :12])
        S = np.matmul(Xc, L)
        sigma2 = np.var(S, axis = 0)
        
    elif control == 3: # Elastic net
        k = 12 # estimate first 12 components
        L = np.zeros((p,k))
        ElasticNet = linear_model.ElasticNet(alpha = 0.0001, l1_ratio = 0.1, fit_intercept = False) # Ratio is 0 for only l2 penalty
        for i in range(k):
            reg_elastic = ElasticNet.fit(Xc, S[:,i]).coef_
            L[:,i] = preprocessing.normalize(reg_elastic, norm = "l2")
            
        S = np.matmul(Xc, L)
        sigma2 = np.var(S, axis = 0)
        
    return n, p, L, X, conlist, Xc, mu, sigma2, S