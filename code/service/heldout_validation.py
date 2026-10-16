"""Full-family held-out experiments with bounded-memory rolling checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import beta, norm
from threadpoolctl import threadpool_limits

from service_validation import (ROOT, OUTPUTS, apply_stress, load_sioux,
    propagation_features, sample_law, service_cost, write_json)
from confirmatory_validation import (candidates, config_and_fingerprint, local_retention,
    normalised_features, quadrature_order)
from batched_propagation import batched_features


def load_design():
    config, calibration_fingerprint = config_and_fingerprint()
    path = OUTPUTS / 'calibration_completion.json'
    calibration = json.loads(path.read_text(encoding='utf-8'))
    assert calibration['status'] == 'complete'
    assert calibration['fingerprint'] == calibration_fingerprint
    assert calibration['selection_locked_before_test']
    coefficient_path = OUTPUTS/'service_coefficients.npz'
    coefficient_audit = json.loads((OUTPUTS/'service_coefficients_completion.json').read_text(encoding='utf-8'))
    assert coefficient_audit['status'] == 'complete'
    assert hashlib.sha256(coefficient_path.read_bytes()).hexdigest() == coefficient_audit['coefficient_sha256']
    assert coefficient_audit['horizon'] == calibration['selected']['adjacency_quantum']['horizon']
    implementation = ROOT/'experiments/batched_propagation.py'
    equivalence = json.loads((ROOT/'audit/batched_propagation_equivalence.json').read_text(encoding='utf-8'))
    assert equivalence['status'] == 'pass'
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() == equivalence['implementation_sha256']
    digest = hashlib.sha256(path.read_bytes()+Path(__file__).read_bytes()+coefficient_path.read_bytes()+
                            (ROOT/'plans/coefficient_validation.json').read_bytes()+implementation.read_bytes()).hexdigest()
    return config, calibration, digest


@lru_cache(maxsize=1)
def loss_coefficients():
    with np.load(OUTPUTS/'service_coefficients.npz') as values:
        return float(values['D0']), values['g'].copy(), values['Q'].copy()


def feature_specs(config, calibration, scope):
    selected = calibration['selected']
    primary_l = selected['adjacency_quantum']['horizon']
    if scope != 'primary':
        return [{'name': 'adjacency_quantum', 'kernel': 'adjacency_quantum',
                 'horizon': primary_l, 'window': 'uniform', 'factor': 1.0}]
    specs = [{'name': f'adjacency_{factor:.2f}', 'kernel': 'adjacency_quantum',
              'horizon': primary_l*factor, 'window': 'uniform', 'factor': factor}
             for factor in config['local_horizon_factors']]
    for kernel in ('laplacian_quantum', 'classical_diffusion'):
        specs.append({'name': kernel, 'kernel': kernel, 'horizon': selected[kernel]['horizon'],
                      'window': 'uniform', 'factor': 1.0})
    for window in ('late_triangular', 'instantaneous'):
        specs.append({'name': window, 'kernel': 'adjacency_quantum', 'horizon': primary_l,
                      'window': window, 'factor': 1.0})
    specs.append({'name': 'local_retention', 'kernel': 'local_retention', 'horizon': 0.,
                  'window': 'none', 'factor': 1.0})
    for order in ('first_order', 'second_order'):
        specs.append({'name': order, 'kernel': 'loss_approximation', 'horizon': primary_l,
                      'window': order, 'factor': 1.0})
    return specs


def compute_features(network, values, specs, orders):
    result = np.empty((len(values), len(specs)))
    groups = sorted({(s['kernel'], s['window']) for s in specs})
    for kernel, window in groups:
        ids = [i for i, s in enumerate(specs) if s['kernel'] == kernel and s['window'] == window]
        if kernel == 'local_retention':
            result[:, ids[0]] = local_retention(network, values)
        elif kernel == 'loss_approximation':
            d0, g, q = loss_coefficients()
            loss = values @ g
            if window == 'second_order':
                loss = loss + .5*np.einsum('bi,ij,bj->b', values, q, values, optimize=True)
            result[:, ids[0]] = (d0-loss-1)/(d0-1)
        else:
            horizons = np.array([specs[i]['horizon'] for i in ids])
            order = orders[kernel+'_'+window]
            healthy = batched_features(network, np.zeros((1, network.m)), horizons, kernel, order, window)[0]
            raw = batched_features(network, values, horizons, kernel, order, window)
            result[:, ids] = (raw-1)/(healthy-1)
    return result


def scope_baseline(network, config, scope):
    count = config['test_scenarios'] if scope == 'primary' else config[scope+'_scenarios']
    seed = config['seeds']['test' if scope == 'primary' else scope]
    if scope != 'direction':
        return sample_law(network, count, seed, config['baseline_alpha_scale'],
                          config['gaussian_copula_correlation'])
    rng = np.random.default_rng(seed)
    rho = config['gaussian_copula_correlation']
    z = np.sqrt(rho)*rng.standard_normal((count, 1)) + np.sqrt(1-rho)*rng.standard_normal((count, len(network.arcs)))
    return config['baseline_alpha_scale']*beta.ppf(np.clip(norm.cdf(z), 1e-12, 1-1e-12), *config['beta_shapes'])


def initialise_arrays(paths, shapes, resume):
    arrays = {}
    for name, shape in shapes.items():
        path = paths[name]
        if resume:
            value = np.load(path, mmap_mode='r+')
            assert value.shape == shape and value.dtype == np.float64
        else:
            value = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=shape)
            value.fill(np.nan)
            value.flush()
        arrays[name] = value
    return arrays


def run_scope(scope):
    config, calibration, fingerprint = load_design()
    completion_path = OUTPUTS / f'{scope}_completion.json'
    if completion_path.exists():
        prior = json.loads(completion_path.read_text())
        if prior['fingerprint'] != fingerprint:
            raise RuntimeError(f'Changed dependencies for completed {scope}; a version decision is required.')
        for artifact in prior['artifacts']:
            assert hashlib.sha256((ROOT/artifact['path']).read_bytes()).hexdigest() == artifact['sha256']
        print(f'{scope}: complete outputs verified; no rerun.', flush=True)
        return
    network = load_sioux()
    frame, members = candidates()
    baseline = scope_baseline(network, config, scope)
    count = len(baseline)
    specs = feature_specs(config, calibration, scope)
    total_rows = 1+len(frame)*(2 if scope == 'allocation' else 1)
    costs_count = 2 if scope == 'direction' else 1
    progress_path = OUTPUTS/f'{scope}_progress.json'
    resume = progress_path.exists()
    progress = json.loads(progress_path.read_text()) if resume else {}
    if resume and progress['fingerprint'] != fingerprint:
        raise RuntimeError(f'Checkpoint fingerprint differs for {scope}.')
    start = perf_counter()
    elapsed_prior = progress.get('elapsed_seconds', 0.)
    next_row = progress.get('next_row', 0)
    if 'orders' in progress:
        orders, integration_checks = progress['orders'], progress['integration_checks']
    else:
        orders, integration_checks = {}, []
        for kernel, window in sorted({(s['kernel'], s['window']) for s in specs}):
            if kernel in ('local_retention', 'loss_approximation'):
                continue
            horizons = [s['horizon'] for s in specs if s['kernel'] == kernel and s['window'] == window]
            q, checks = quadrature_order(network, config, horizons, [kernel], window)
            orders[kernel+'_'+window] = q
            integration_checks += checks
    paths = {'features': OUTPUTS/f'{scope}_features.npy', 'costs': OUTPUTS/f'{scope}_costs.npy'}
    shapes = {'features': (total_rows, count, len(specs)), 'costs': (total_rows, count, costs_count)}
    arrays = initialise_arrays(paths, shapes, resume)
    healthy = []
    for spec in specs:
        if spec['kernel'] == 'local_retention':
            healthy.append(None)
        elif spec['kernel'] == 'loss_approximation':
            healthy.append(loss_coefficients()[0])
        else:
            healthy.append(float(propagation_features(network, np.zeros((1, network.m)),
                np.array([spec['horizon']]), spec['kernel'], orders[spec['kernel']+'_'+spec['window']], spec['window'])[0, 0]))
    metadata = {'status': 'running', 'scope': scope, 'pid': os.getpid(), 'fingerprint': fingerprint,
                'next_row': next_row, 'total_rows': total_rows, 'scenarios_per_row': count,
                'orders': orders, 'integration_checks': integration_checks, 'feature_specs': specs,
                'healthy_dpc': healthy, 'baseline_cost': float(service_cost(network, np.zeros((1, network.m)))[0]),
                'elapsed_seconds': elapsed_prior}
    write_json(progress_path, metadata)
    print(f'{scope}: starting at row {next_row}/{total_rows}, M={count}, {len(specs)} scores, quadrature={orders}', flush=True)
    for row in range(next_row, total_rows):
        if row == 0:
            states = baseline
        else:
            candidate = (row-1) % len(frame)
            indices = members[candidate]
            if scope == 'direction':
                indices = np.array([int(np.flatnonzero((network.arc_edge == edge) &
                    (network.arcs[:, 0] < network.arcs[:, 1]))[0]) for edge in indices])
                mode, amount = 'fractional', config['common_fractional_stress']
            elif scope == 'allocation':
                mode = 'headroom' if row <= len(frame) else 'uniform'
                amount = config['equal_total_stress']
            else:
                mode, amount = 'fractional', config['common_fractional_stress']
            states = apply_stress(baseline, indices, config['physical_alpha_cap'], amount, mode)
        arrays['features'][row] = compute_features(network, states, specs, orders)
        arrays['costs'][row, :, 0] = service_cost(network, states)
        if scope == 'direction':
            arrays['costs'][row, :, 1] = service_cost(network, states, symmetrise=True)
        for array in arrays.values():
            array.flush()
        metadata.update(next_row=row+1, elapsed_seconds=elapsed_prior+perf_counter()-start)
        write_json(progress_path, metadata)
        if row == 0 or (row+1) % 50 == 0 or row == total_rows-1:
            print(f'{scope}: {row+1}/{total_rows}, elapsed {metadata["elapsed_seconds"]:.1f}s', flush=True)
    assert all(np.isfinite(a).all() for a in arrays.values())
    if scope != 'direction':
        assert np.min(arrays['costs'][1:, :, 0]-arrays['costs'][0, :, 0]) >= -1e-10
    artifacts = [{'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                 for path in paths.values()]
    metadata.update(status='complete', artifacts=artifacts)
    write_json(completion_path, metadata)
    write_json(progress_path, metadata)
    print(f'{scope}: completed and hashed all {total_rows*count} evaluations.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scope', choices=['primary', 'allocation', 'direction', 'all'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for scope in (('primary', 'allocation', 'direction') if args.scope == 'all' else (args.scope,)):
            run_scope(scope)


if __name__ == '__main__':
    main()
