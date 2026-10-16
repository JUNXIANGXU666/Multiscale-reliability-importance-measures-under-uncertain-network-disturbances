"""Independent checks of the service bridge and exact directional example."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits

from service_validation import ROOT, ServiceNetwork, service_cost, generators, write_json


def dpc_scalar(h, horizon, source_weights):
    points, weights = leggauss(64)
    value = 0.
    for t, weight in zip((points+1)*horizon/2, weights/2):
        p = np.abs(expm(-1j*h*t))**2
        value += weight*np.dot(1/np.sum(p*p, axis=0), source_weights)
    return value


def main():
    rng = np.random.default_rng(2026101610)
    homogeneous = []
    for n in (3, 4, 6):
        a = rng.uniform(.1, 1, (n, n))
        h = np.triu(a, 1)
        h += h.T
        source_weights = rng.dirichlet(np.ones(n))
        kappa = np.dot(np.sum(h*h, axis=0), source_weights)
        for horizon in (.01, .02, .04, .2):
            baseline = dpc_scalar(h, horizon, source_weights)
            alpha = .37
            left = dpc_scalar((1-alpha)*h, horizon, source_weights)
            right = dpc_scalar(h, (1-alpha)*horizon, source_weights)
            assert abs(left-right) < 1e-12
            leading = 1+2*kappa*horizon**2/3
            homogeneous.append({'n': n, 'horizon': horizon,
                                'scale_identity_error': abs(left-right),
                                'expansion_error_over_L4': (baseline-leading)/horizon**4})
        for cost_limit in (1.05, 1.1, 1.2):
            horizon = .04
            threshold = dpc_scalar(h, horizon/cost_limit, source_weights)
            for alpha in (0, .01, .04, .08, .15, .3, .6):
                service = 1/(1-alpha) <= cost_limit
                proxy = dpc_scalar((1-alpha)*h, horizon, source_weights) >= threshold
                assert service == proxy
    network = ServiceNetwork(np.array([[0, 1], [1, 0]]), np.ones(2), np.array([[0, 1]]),
        np.array([0, 0]), np.array([[0., 1.], [0., 0.]]), np.array([1., 0.]), np.array([[0., 0.], [1., 0.]]))
    states = np.array([[.8, 0], [0, .8]])
    directed = service_cost(network, states)
    symmetric = service_cost(network, states, True)
    h = generators(network, states, False)
    assert np.array_equal(h[0], h[1])
    assert np.allclose(directed, [5., 1.])
    assert np.allclose(symmetric, [5/3, 5/3])
    bits = np.array(list(itertools.product([0, 1], repeat=6)))
    worst_slack = 10.
    for _ in range(1000):
        probability = rng.dirichlet(np.ones(len(bits)))
        r_y0, r_z0, r_ys, r_zs, r_yt, r_zt = probability @ bits
        e0 = probability @ (bits[:, 0] != bits[:, 1])
        es = probability @ (bits[:, 2] != bits[:, 3])
        et = probability @ (bits[:, 4] != bits[:, 5])
        importance_error = abs((r_y0-r_ys)-(r_z0-r_zs))
        ordering_error = abs((r_yt-r_ys)-(r_zt-r_zs))
        assert importance_error <= e0+es+1e-14
        assert ordering_error <= es+et+1e-14
        worst_slack = min(worst_slack, e0+es-importance_error, es+et-ordering_error)
    result = {'status': 'pass', 'homogeneous_tests': homogeneous,
              'direction_counterexample': {'directed_costs': directed.tolist(), 'symmetrised_costs': symmetric.tolist(),
                  'identical_generator': True, 'service_success_at_cost_2': (directed <= 2).tolist()},
              'transfer_random_finite_laws': 1000, 'minimum_bound_slack': worst_slack,
              'role': 'Numerical checks supplement the analytical proof; they are not a proof of universality.'}
    write_json(ROOT/'audit/service_theory_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
