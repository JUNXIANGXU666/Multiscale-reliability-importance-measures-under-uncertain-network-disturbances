"""Directed OD travel-service evaluation and CTQW feature diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm
from scipy.sparse.csgraph import shortest_path
from scipy.stats import beta, norm, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'experiments/inputs'
OUTPUTS = ROOT / 'experiments/results'


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


@dataclass
class ServiceNetwork:
    arcs: np.ndarray
    free_time: np.ndarray
    edges: np.ndarray
    arc_edge: np.ndarray
    od: np.ndarray
    source_weights: np.ndarray
    positions: np.ndarray

    @property
    def n(self) -> int:
        return len(self.od)

    @property
    def m(self) -> int:
        return len(self.edges)


def load_sioux() -> ServiceNetwork:
    rows = []
    for line in (INPUTS / 'SiouxFalls_net.tntp').read_text(encoding='utf-8').splitlines():
        fields = line.replace(';', '').split()
        if len(fields) >= 10 and fields[0].isdigit() and fields[1].isdigit():
            rows.append([float(x) for x in fields[:10]])
    data = np.asarray(rows)
    arcs = data[:, :2].astype(int) - 1
    edges = np.array(sorted({tuple(sorted(a)) for a in arcs}), dtype=int)
    lookup = {tuple(e): i for i, e in enumerate(edges)}
    arc_edge = np.array([lookup[tuple(sorted(a))] for a in arcs])
    od = np.zeros((24, 24))
    origin = None
    for line in (INPUTS / 'SiouxFalls_trips.tntp').read_text(encoding='utf-8').splitlines():
        match = re.match(r'\s*Origin\s+(\d+)', line)
        if match:
            origin = int(match.group(1)) - 1
        elif origin is not None:
            for destination, flow in re.findall(r'(\d+)\s*:\s*([\d.]+)', line):
                od[origin, int(destination)-1] = float(flow)
    positions = np.zeros((24, 2))
    for line in (INPUTS / 'SiouxFalls_node.tntp').read_text(encoding='utf-8').splitlines():
        fields = line.replace(';', '').split()
        if len(fields) >= 3 and fields[0].isdigit():
            positions[int(fields[0])-1] = [float(fields[1]), float(fields[2])]
    assert len(arcs) == 76 and len(edges) == 38 and np.isclose(od.sum(), 360600)
    return ServiceNetwork(arcs, data[:, 4], edges, arc_edge, od, od.sum(axis=1)/od.sum(), positions)


def arc_disturbances(network: ServiceNetwork, alphas: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(alphas, dtype=float))
    if values.shape[1] == network.m:
        return values[:, network.arc_edge]
    if values.shape[1] == len(network.arcs):
        return values
    raise ValueError('Expected physical-edge or directed-arc disturbances.')


def service_cost(network: ServiceNetwork, alphas: np.ndarray, symmetrise: bool = False) -> np.ndarray:
    values = arc_disturbances(network, alphas)
    if np.any(values < 0) or np.any(values >= 1):
        raise ValueError('Service travel-time model requires 0 <= alpha < 1.')
    costs = np.empty(len(values))
    for i, alpha in enumerate(values):
        times = network.free_time / (1-alpha)
        matrix = np.zeros((network.n, network.n))
        matrix[network.arcs[:, 0], network.arcs[:, 1]] = times
        if symmetrise:
            rates = 1/times
            sums = np.bincount(network.arc_edge, weights=rates, minlength=network.m)
            counts = np.bincount(network.arc_edge, minlength=network.m)
            symmetric_times = counts/sums
            matrix.fill(0)
            matrix[network.edges[:, 0], network.edges[:, 1]] = symmetric_times
            matrix[network.edges[:, 1], network.edges[:, 0]] = symmetric_times
        distances = shortest_path(matrix, method='FW', directed=True)
        if not np.isfinite(distances[network.od > 0]).all():
            costs[i] = np.inf
        else:
            costs[i] = np.sum(network.od * distances) / network.od.sum()
    return costs


def generators(network: ServiceNetwork, alphas: np.ndarray, laplacian: bool) -> np.ndarray:
    alpha_arc = arc_disturbances(network, alphas)
    raw = (1-alpha_arc)/network.free_time[None, :]
    rates = np.zeros((len(raw), network.m))
    for arc, edge in enumerate(network.arc_edge):
        rates[:, edge] += raw[:, arc]
    rates /= np.bincount(network.arc_edge, minlength=network.m)[None, :]
    # Fixed undisturbed scale, never renormalise a disturbed network.
    normalisation = max(np.mean(1/network.free_time[network.arc_edge == e]) for e in range(network.m))
    rates /= normalisation
    matrices = np.zeros((len(raw), network.n, network.n))
    matrices[:, network.edges[:, 0], network.edges[:, 1]] = rates
    matrices[:, network.edges[:, 1], network.edges[:, 0]] = rates
    if laplacian:
        degree = matrices.sum(axis=2)
        matrices *= -1
        index = np.arange(network.n)
        matrices[:, index, index] = degree
    return matrices


def propagation_features(network: ServiceNetwork, alphas: np.ndarray, horizons: np.ndarray,
                         kernel: str, order: int = 48, window: str = 'uniform', batch: int = 16) -> np.ndarray:
    values = np.atleast_2d(alphas)
    out = np.empty((len(values), len(horizons)))
    points, quadrature = leggauss(order)
    s, weights = (points+1)/2, quadrature/2
    if window == 'late_triangular':
        weights = weights * 2*s
    elif window == 'instantaneous':
        s, weights = np.array([1.]), np.array([1.])
    elif window != 'uniform':
        raise ValueError(window)
    for start in range(0, len(values), batch):
        matrices = generators(network, values[start:start+batch], kernel != 'adjacency_quantum')
        eig, vectors = np.linalg.eigh(matrices)
        for j, horizon in enumerate(horizons):
            times = s * horizon
            factors = np.exp((-1 if kernel == 'classical_diffusion' else -1j) * eig[:, None, :] * times[None, :, None])
            propagators = np.einsum('bik,btk,bjk->btij', vectors, factors, vectors, optimize=True)
            if kernel == 'classical_diffusion':
                assert np.min(propagators) > -1e-10
                probability = np.maximum(propagators.real, 0)
            else:
                probability = np.abs(propagators)**2
            assert np.max(np.abs(probability.sum(axis=2)-1)) < 1e-10
            participation = 1/np.sum(probability**2, axis=2)
            out[start:start+len(eig), j] = np.einsum('btr,t,r->b', participation, weights, network.source_weights)
    return out


def sample_law(network: ServiceNetwork, count: int, seed: int, scale: float, rho: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.sqrt(rho)*rng.standard_normal((count, 1)) + np.sqrt(1-rho)*rng.standard_normal((count, network.m))
    return scale * beta.ppf(np.clip(norm.cdf(z), 1e-12, 1-1e-12), 2, 8)


def apply_stress(values: np.ndarray, indices: np.ndarray, cap: float, amount: float, mode: str) -> np.ndarray:
    stressed = values.copy()
    headroom = cap-values[:, indices]
    if mode == 'fractional':
        increment = amount*headroom
    elif mode == 'headroom':
        increment = amount*headroom/headroom.sum(axis=1)[:, None]
    elif mode == 'uniform':
        increment = np.full_like(headroom, amount/len(indices))
    else:
        raise ValueError(mode)
    if np.any(increment > headroom+1e-13) or np.any(increment < 0):
        raise ValueError('Infeasible stress, clipping is forbidden.')
    stressed[:, indices] += increment
    return stressed


def best_threshold(values: np.ndarray, success: np.ndarray) -> tuple[float, float]:
    if success.all() or not success.any():
        return float('nan'), float('nan')
    order = np.argsort(-values, kind='stable')
    y, x = success[order], values[order]
    true_positive = np.r_[0, np.cumsum(y)]
    false_positive = np.r_[0, np.cumsum(~y)]
    false_negative = success.sum()-true_positive
    balanced_error = (false_negative/success.sum()+false_positive/(~success).sum())/2
    eligible = np.r_[True, x[:-1] != x[1:], True]
    balanced_error[~eligible] = np.inf
    index = int(np.argmin(balanced_error))
    threshold = np.nextafter(x[0], np.inf) if index == 0 else (x[-1] if index == len(x) else (x[index-1]+x[index])/2)
    return float(threshold), float(balanced_error[index])


def tests() -> dict[str, object]:
    network = load_sioux()
    zero = np.zeros((1, network.m))
    c0 = service_cost(network, zero)[0]
    states = sample_law(network, 5, 381991, .3, .35)
    assert np.all(service_cost(network, states) >= c0-1e-12)
    reference_errors = []
    for a, c in zip(states, service_cost(network, states), strict=True):
        graph = nx.DiGraph()
        for (u, v), t, e in zip(network.arcs, network.free_time, network.arc_edge, strict=True):
            graph.add_edge(int(u), int(v), weight=float(t/(1-a[e])))
        distances = dict(nx.all_pairs_dijkstra_path_length(graph))
        reference = sum(network.od[r, s]*distances[r][s] for r in range(24) for s in range(24))/network.od.sum()
        reference_errors.append(abs(reference-c))
    assert max(reference_errors) < 1e-12
    feature = propagation_features(network, zero, np.array([1.8]), 'adjacency_quantum', 64)[0, 0]
    assert abs(feature-3.6426) < 5e-5
    h = generators(network, zero, False)[0]
    t, w = leggauss(48)
    scalar = 0.
    for point, weight in zip((t+1)*.9, w/2, strict=True):
        p = np.abs(expm(-1j*h*point))**2
        scalar += weight*np.dot(1/np.sum(p*p, axis=0), network.source_weights)
    assert abs(scalar-feature) < 1e-10
    for mode in ('headroom', 'uniform'):
        changed = apply_stress(states, np.array([1, 2, 3]), .6, .15, mode)
        assert np.max(np.abs((changed-states).sum(axis=1)-.15)) < 1e-13
    symmetric_error = np.max(np.abs(service_cost(network, states)-service_cost(network, states, True)))
    assert symmetric_error < 1e-12
    return {'status': 'pass', 'n': network.n, 'arcs': len(network.arcs), 'physical_edges': network.m,
            'od_sum': network.od.sum(), 'baseline_mean_travel_time_data_units': c0,
            'baseline_mean_travel_time_minutes': .6*c0, 'baseline_dpc': feature,
            'independent_shortest_path_max_error': max(reference_errors),
            'scalar_expm_feature_error': abs(scalar-feature), 'symmetric_control_error': symmetric_error}


def pilot() -> None:
    config = json.loads((ROOT / 'plans/diagnostic_pilot.json').read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256((ROOT / 'plans/diagnostic_pilot.json').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    done = OUTPUTS / 'pilot_completion.json'
    if done.exists():
        prior = json.loads(done.read_text(encoding='utf-8'))
        if prior['fingerprint'] == fingerprint:
            print('Verified matching completed pilot; no rerun.', flush=True)
            return
        raise RuntimeError('Pilot code/design changed; record a new run reason before replacing evidence.')
    OUTPUTS.mkdir(exist_ok=True)
    write_json(OUTPUTS / 'pilot_progress.json', {'status': 'running', 'pid': os.getpid(),
               'fingerprint': fingerprint, 'completed_kernels': [], 'total_kernels': len(config['kernels'])})
    write_json(OUTPUTS / 'service_unit_tests.json', tests())
    network = load_sioux()
    baseline = sample_law(network, config['scenario_count'], config['seed'], config['baseline_alpha_scale'], config['gaussian_copula_correlation'])
    candidate_frame = pd.read_csv(INPUTS / 'sioux_candidates.csv')
    selected = pd.concat([candidate_frame[candidate_frame.form == 'edge'],
                          candidate_frame[candidate_frame.form == 'corridor'].iloc[::64],
                          candidate_frame[candidate_frame.form == 'region']], ignore_index=True)
    states = [baseline]
    for candidate in selected.itertuples(index=False):
        indices = np.array([int(e) for e in str(candidate.edge_ids).split('|')])
        states.append(apply_stress(baseline, indices, config['physical_alpha_cap'], config['common_stress'], 'fractional'))
    all_states = np.concatenate(states)
    costs = service_cost(network, all_states)
    c0 = service_cost(network, np.zeros((1, network.m)))[0]
    ratios = costs/c0
    save_npz(OUTPUTS / 'pilot_states.npz', alphas=all_states, cost_ratios=ratios)
    selected.to_csv(OUTPUTS / 'pilot_candidates.csv', index=False)
    horizons = np.asarray(config['horizons'])
    rows = []
    start = perf_counter()
    for kernel in config['kernels']:
        checkpoint = OUTPUTS / f'pilot_{kernel}.npz'
        if checkpoint.exists():
            saved = np.load(checkpoint)
            assert str(saved['fingerprint']) == fingerprint
            features = saved['features']
        else:
            features = propagation_features(network, all_states, horizons, kernel, config['quadrature_order'])
            save_npz(checkpoint, features=features, fingerprint=np.array(fingerprint))
        for j, horizon in enumerate(horizons):
            for limit in config['service_ratio_limits']:
                success = ratios <= limit
                threshold, error = best_threshold(features[:, j], success)
                predicted = features[:, j] >= threshold
                M = config['scenario_count']
                actual_importance = success[:M].mean()-success[M:].reshape(-1, M).mean(axis=1)
                proxy_importance = predicted[:M].mean()-predicted[M:].reshape(-1, M).mean(axis=1)
                for form in ('edge', 'corridor', 'region', 'all'):
                    mask = np.ones(len(selected), dtype=bool) if form == 'all' else (selected.form.to_numpy() == form)
                    x, y = actual_importance[mask], proxy_importance[mask]
                    correlation = float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else float('nan')
                    rows.append({'kernel': kernel, 'horizon': horizon, 'service_ratio_limit': limit, 'form': form,
                                 'fitted_dpc_threshold': threshold, 'in_sample_balanced_error': error,
                                 'service_success_fraction': success.mean(), 'scenario_spearman': float(spearmanr(-ratios, features[:, j]).statistic),
                                 'candidate_count': int(mask.sum()), 'importance_spearman': correlation,
                                 'importance_mae': float(np.mean(abs(x-y)))})
        pd.DataFrame(rows).to_csv(OUTPUTS / 'pilot_diagnostics.csv', index=False)
        write_json(OUTPUTS / 'pilot_progress.json', {'status': 'running', 'pid': os.getpid(),
                   'fingerprint': fingerprint, 'completed_kernels': config['kernels'][:config['kernels'].index(kernel)+1],
                   'total_kernels': len(config['kernels']), 'elapsed_seconds': perf_counter()-start})
        print(f'Pilot finished {kernel}; {len(all_states)} states at {len(horizons)} horizons; elapsed {perf_counter()-start:.1f}s', flush=True)
    write_json(done, {'fingerprint': fingerprint, 'status': 'complete', 'role': config['role'],
                      'states': len(all_states), 'candidates': len(selected), 'elapsed_seconds': perf_counter()-start})
    write_json(OUTPUTS / 'pilot_progress.json', {'status': 'complete', 'pid': os.getpid(),
               'fingerprint': fingerprint, 'completed_kernels': config['kernels'], 'total_kernels': len(config['kernels'])})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['test', 'pilot'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.mode == 'test':
            OUTPUTS.mkdir(exist_ok=True)
            result = tests()
            write_json(OUTPUTS / 'service_unit_tests.json', result)
            print(json.dumps(result, indent=2), flush=True)
        else:
            pilot()


if __name__ == '__main__':
    main()
