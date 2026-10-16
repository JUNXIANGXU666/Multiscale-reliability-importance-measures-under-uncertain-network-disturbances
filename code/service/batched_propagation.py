"""Batched matrix products for the unchanged eigendecomposition feature formula."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from service_validation import generators


def batched_features(network, alphas, horizons, kernel, order=32, window='uniform', batch=64):
    values = np.atleast_2d(alphas)
    out = np.empty((len(values), len(horizons)))
    points, quadrature = leggauss(order)
    s, weights = (points+1)/2, quadrature/2
    if window == 'late_triangular':
        weights = weights*2*s
    elif window == 'instantaneous':
        s, weights = np.array([1.]), np.array([1.])
    elif window != 'uniform':
        raise ValueError(window)
    for start in range(0, len(values), batch):
        h = generators(network, values[start:start+batch], kernel != 'adjacency_quantum')
        eigenvalues, vectors = np.linalg.eigh(h)
        transpose = vectors.transpose(0, 2, 1)[:, None, :, :]
        for j, horizon in enumerate(horizons):
            factors = np.exp((-1 if kernel == 'classical_diffusion' else -1j)*
                             eigenvalues[:, None, :]*(s*horizon)[None, :, None])
            propagators = np.matmul(vectors[:, None, :, :]*factors[:, :, None, :], transpose)
            if kernel == 'classical_diffusion':
                assert np.min(propagators) > -1e-10
                probability = np.maximum(propagators.real, 0)
            else:
                probability = np.abs(propagators)**2
            assert np.max(np.abs(probability.sum(axis=2)-1)) < 1e-10
            participation = 1/np.sum(probability**2, axis=2)
            out[start:start+len(eigenvalues), j] = np.einsum(
                'btr,t,r->b', participation, weights, network.source_weights)
    return out
