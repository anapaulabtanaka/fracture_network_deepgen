#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Miscalleneous tools.
"""

# -------------------------------------------------------------------------
#   Author: Julien Straubhaar
#   Year: 2024
#   Company: University of Neuchâtel
#
#   Copyright (c) 2024 Julien Straubhaar
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# -------------------------------------------------------------------------

import numpy as np
import scipy
import torch
# -----------------------------------------------------------------------------
def multidimensional_scaling(dmat, dim=None):
    """
    Do Multi-Dimensional Scaling (MDS). Given a dissimilarity matrix `dmat` of order M
    (square matrix with positive entries and zeros on the diagonal), this function 
    retrieves M points in dimension dim such that the distance matrix of those points is 
    as close as possible as `dmat`; if the dimension `dim` is not given, it is 
    automatically computed (< M) such that the distance matrix of the points is `dmat`.

    Parameters
    ----------
    dmat : 1d array or 2d array
        dissimilarity matrix, symmetric, of order M, with positive entries;
        if `dmat` is a 1d array: its length should be M*(M-1)/2 and corresponds 
        to the strictly upper (or lower) part of the matrix
    
    dim : int, optional
        dimension of output points;
        by default (`dim=None`): `dim` is computed automatically computed (< M) 
        such that the distance matrix of the points is `dmat`
    
    Returns
    -------
    X : 2d array of shape (M, dim)
        each line correspond to a point
    
    f_ssq : 1d array of shape (dim,)
        fraction of the sum of squared "explained" by each MDS axis

    Examples
    --------
    ::
    
        x = np.random.random((12, 35))           # 12 random points in dimension 35
        dmat = scipy.spatial.distance.pdist(x)   # distance (dissimilarity) matrix of these points
        x2, _ = multidimensional_scaling(dmat)   # 12 "MDS points" in dimension dim < 12
        dmat2 = scipy.spatial.distance.pdist(x2) # distance matrix of the MDS points
        np.allclose(dmat, dmat2)                 # is True
    """
    if dmat.ndim == 1:
        dmat = scipy.spatial.distance.squareform(dmat)
    
    dmat = dmat**2

    M = dmat.shape[0]
    J = np.eye(M) - 1.0/M * np.ones((M, M))
    
    B = -0.5 * J.dot(dmat).dot(J)
    v, Q = np.linalg.eig(B)
    ind = np.argsort(v)[::-1]
    v = v[ind]
    Q = Q[:, ind]
    # np.allclose(Q.dot(np.diag(v)).dot(Q.T), B) # True

    if dim is None:
        dim = np.where(np.isclose(v, 0))[0]
        if len(dim) == 0:
            dim = M
        else:
            dim = dim[0]

    v = np.maximum(v.real, 0) # take real part (truncated at zero)
    f_ssq = v[:dim]/v.sum()
    X = Q[:, :dim].dot(np.diag(np.sqrt(v[:dim]))).real # take the real part
    return X, f_ssq
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def nw_kernel_estimate(tx, x, y, bw=None, extrapolate=False):
    """
    Nadayara-Watson kernel estimator.
    
    Ref: Demir, S. and Toktamiş, Ö. (2010)
    ON THE ADAPTIVE NADARAYA-WATSON KERNEL REGRESSION ESTIMATORS,
    Hacettepe Journal of Mathematics and Statistics, 39, pp. 429-437

    For a point cloud `(x, y)`, this function computes the kernel estimates `ty`
    (ordinates) at `tx` (abscissa), using the formula

    .. math::

        ty_i = \sum_{j}{w_{i,j} \cdot ty_i}/\sum_{j} w_{i,j}
        
    with

    .. math::
        
        w_{i,j} = \exp(-1/2\cdot((tx_i - x_j)/bw)^2)
    
    The bandwidth (`bw`) is set by default to
    
    .. math:: 
        
        bw = 0.9\min(\sigma(x), IQR(x)/1.34)\cdot N^{-1/5}
            
    where N is the number of point in the points cloud.

    Parameters
    ----------
    tx : 1d array
        abscissa of the points where the estimate of the ordinate has to 
        be done
    
    x : 1d array
        abscissa of the known points
    
    y : 1d array
        ordinates of the known points
    
    bw : float, optional
        bandwidth (i.e. standard deviation of the Gaussian kernel) used in the
        kernel estimator (see above for default)
    
    extrapolate : bool, default: `False`
        the estimates for abscissa (from `tx`) out of the interval [min(x), max(x)]
        are computed if `extrapolate=True`, otherwise (`extrapolate=False`) those 
        estimates are replaced by `numpy.nan`

    Returns
    -------
    ty : 1d array
        kernel estimates (ordinates) at `tx` (abscissa)    
    """
    if bw is None:
        bw = 0.9*min(np.std(x), np.diff(np.quantile(x, q=(0.25, 0.75)))/1.34)*x.size**(-0.2)
    
    xmin, xmax = min(x), max(x)

    ty = np.full(tx.shape, np.nan)
    for i in range(len(tx)):
        if tx[i] < xmin or tx[i] > xmax:
            continue
        w = scipy.stats.norm.pdf(tx[i], loc=x, scale=bw)
        ty[i] = np.sum(w*y)/np.sum(w)

    return ty
# ----------------------------------------------------------------------------

def default_device():
# Autoset device to MPS (Apple Silicon) if available, 
# otherwise to CUDA (NVIDIA GPU) if available, 
# otherwise to CPU

    if torch.backends.mps.is_available():
        #print("mps")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        #print("cuda")
        device = torch.device("cuda")
    else:
        #print("cpu")
        device = torch.device("cpu")

    return device