"""Small independent checks of the DPC implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from dpc_core import motif_networks, dpc, rank_aware_coefficients, finite_difference_coefficients

with threadpool_limits(limits=1):
    networks = motif_networks()
    if isinstance(networks, dict):
        networks = list(networks.values())
    network = networks[0]
    matrix = network.h0
    propagator = expm(-1j*matrix*.7)
    assert np.max(abs(propagator.conj().T@propagator-np.eye(network.n))) < 1e-12
    probability = abs(propagator)**2
    assert np.max(abs(probability.sum(axis=0)-1)) < 1e-12
    index = 1/(probability**2).sum(axis=0)
    assert np.all(index >= 1-1e-12) and np.all(index <= network.n+1e-12)
    analytical = rank_aware_coefficients(network, 1.2, 18, 5)
    finite = finite_difference_coefficients(network, 1.2, 48, 1e-3,
        fourth_order_edges=True, richardson_mixed=True)
    assert np.max(abs(analytical[1]-finite[1])) < 1e-5
    assert np.max(abs(analytical[2]-finite[2])) < 1e-4
    assert np.max(abs(analytical[2]-analytical[2].T)) < 1e-12
print('DPC unitarity, probability, participation and derivative checks passed.')
