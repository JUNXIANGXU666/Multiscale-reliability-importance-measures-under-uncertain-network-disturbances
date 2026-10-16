"""Check numerical equivalence of the batched and original propagation evaluators."""
import hashlib
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from service_validation import ROOT, load_sioux, sample_law, propagation_features, write_json
from batched_propagation import batched_features


def main():
    network = load_sioux()
    states = np.concatenate([np.zeros((1, network.m)), sample_law(network, 31, 9317, .55, .35)])
    horizons = np.array([.3, 1.08, 1.2, 1.32, 2.4, 4.8])
    rows = []
    for kernel in ('adjacency_quantum', 'laplacian_quantum', 'classical_diffusion'):
        for window in ('uniform', 'late_triangular', 'instantaneous'):
            started = perf_counter()
            expected = propagation_features(network, states, horizons, kernel, 64, window)
            reference_seconds = perf_counter()-started
            started = perf_counter()
            actual = batched_features(network, states, horizons, kernel, 64, window)
            elapsed = perf_counter()-started
            error = float(np.max(np.abs(actual-expected)))
            assert error < 1e-11
            rows.append({'kernel': kernel, 'window': window, 'maximum_absolute_difference': error,
                         'reference_seconds': reference_seconds, 'batched_seconds': elapsed})
    write_json(ROOT/'audit/batched_propagation_equivalence.json',
        {'status': 'pass', 'state_count': len(states), 'horizons': horizons.tolist(), 'checks': rows,
         'implementation_sha256': hashlib.sha256((ROOT/'experiments/batched_propagation.py').read_bytes()).hexdigest(),
         'reference_sha256': hashlib.sha256((ROOT/'experiments/service_validation.py').read_bytes()).hexdigest()})
    print(f'All {len(rows)} batched propagation equivalence checks passed.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
