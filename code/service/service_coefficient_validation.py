"""Verify the ordered-integral coefficients at the locked service-calibration scale."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from confirmatory_validation import config_and_fingerprint
from service_validation import ROOT, OUTPUTS, generators, load_sioux, propagation_features, save_npz, write_json
from reused_coefficient_kernel import WeightedNetwork, finite_difference_coefficients, rank_aware_coefficients


def main():
    _, calibration_fingerprint = config_and_fingerprint()
    calibration_path = OUTPUTS/'calibration_completion.json'
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    assert calibration['status'] == 'complete' and calibration['fingerprint'] == calibration_fingerprint
    plan_path = ROOT/'plans/coefficient_validation.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(calibration_path.read_bytes()+plan_path.read_bytes()+
        Path(__file__).read_bytes()+(ROOT/'experiments/reused_coefficient_kernel.py').read_bytes()).hexdigest()
    completion_path = OUTPUTS/'service_coefficients_completion.json'
    output_path = OUTPUTS/'service_coefficients.npz'
    if completion_path.exists():
        prior = json.loads(completion_path.read_text(encoding='utf-8'))
        assert prior['fingerprint'] == fingerprint
        assert hashlib.sha256(output_path.read_bytes()).hexdigest() == prior['coefficient_sha256']
        print('Calibrated coefficients already complete and verified.', flush=True)
        return
    network = load_sioux()
    h = generators(network, np.zeros((1, network.m)), False)[0]
    adapter = WeightedNetwork(network.n, network.edges, h[network.edges[:, 0], network.edges[:, 1]])
    assert np.max(np.abs(adapter.h0-h)) == 0
    horizon = calibration['selected']['adjacency_quantum']['horizon']
    started = perf_counter()
    previous = None
    records = []
    selected = None
    for outer, inner in plan['quadrature_pairs']:
        result = rank_aware_coefficients(adapter, horizon, outer, inner, network.source_weights)
        record = {'n_time': outer, 'n_inner': inner, 'timing': result[3]}
        if previous is not None:
            errors = [float(np.max(np.abs(result[i]-previous[i]))) for i in range(3)]
            record['successive_absolute_errors_D_g_Q'] = errors
            record['pass'] = max(errors) <= plan['successive_absolute_tolerance']
        records.append(record)
        write_json(OUTPUTS/'service_coefficients_progress.json',
            {'status': 'running', 'fingerprint': fingerprint, 'horizon': horizon, 'quadrature': records})
        print(f'Coefficient quadrature {outer}/{inner}: {record}', flush=True)
        if previous is not None and record['pass']:
            selected = result
            break
        previous = result
    if selected is None:
        raise RuntimeError('Coefficient quadrature did not converge under the frozen rule.')
    reference = finite_difference_coefficients(adapter, horizon,
        plan['finite_difference_time_order'], plan['finite_difference_step'],
        fourth_order_edges=True, richardson_mixed=True, source_weights=network.source_weights)
    independent = float(propagation_features(network, np.zeros((1, network.m)),
        np.array([horizon]), 'adjacency_quantum', 64)[0, 0])
    errors = {'g_max_absolute': float(np.max(np.abs(selected[1]-reference[1]))),
              'Q_max_absolute': float(np.max(np.abs(selected[2]-reference[2]))),
              'D0_independent_absolute': abs(selected[0]-independent),
              'Q_asymmetry': float(np.max(np.abs(selected[2]-selected[2].T)))}
    passed = (errors['g_max_absolute'] <= plan['finite_difference_g_absolute_tolerance'] and
              errors['Q_max_absolute'] <= plan['finite_difference_Q_absolute_tolerance'] and
              errors['D0_independent_absolute'] <= plan['independent_feature_tolerance'])
    audit = {'status': 'pass' if passed else 'fail', 'fingerprint': fingerprint,
             'horizon': horizon, 'quadrature': records, 'independent_errors': errors,
             'finite_difference_timing': reference[3], 'elapsed_seconds': perf_counter()-started}
    write_json(OUTPUTS/'service_coefficients_progress.json', audit)
    if not passed:
        raise RuntimeError(f'Independent coefficient audit failed: {errors}')
    save_npz(output_path, D0=np.array(selected[0]), g=selected[1], Q=selected[2],
             horizon=np.array(horizon), source_weights=network.source_weights)
    audit.update(status='complete', coefficient_path=str(output_path.relative_to(ROOT)).replace('\\', '/'),
                 coefficient_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest())
    write_json(completion_path, audit)
    print(f'Coefficient audit complete: {errors}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
