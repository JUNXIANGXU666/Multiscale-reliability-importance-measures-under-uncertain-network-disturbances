"""Independent calibration and held-out transport-service validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import beta, norm, spearmanr
from threadpoolctl import threadpool_limits

from service_validation import (ROOT, INPUTS, OUTPUTS, apply_stress, best_threshold,
    generators, load_sioux, propagation_features, sample_law, save_npz, service_cost,
    write_json)

CONFIG_PATH = ROOT / 'plans/confirmatory_protocol.json'
FORMS = ('baseline', 'edge', 'corridor', 'region')


def config_and_fingerprint():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    digest = hashlib.sha256()
    for path in [CONFIG_PATH, Path(__file__), Path(__file__).with_name('service_validation.py')]:
        digest.update(path.read_bytes())
    for path in sorted(INPUTS.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return config, digest.hexdigest()


def candidates():
    frame = pd.read_csv(INPUTS / 'sioux_candidates.csv')
    assert len(frame) == 797
    return frame, [np.array([int(x) for x in str(v).split('|')]) for v in frame.edge_ids]


def error_metrics(prediction, success):
    n_positive, n_negative = int(success.sum()), int((~success).sum())
    false_negative = float(np.mean(~prediction[success])) if n_positive else None
    false_positive = float(np.mean(prediction[~success])) if n_negative else None
    balanced = (false_negative+false_positive)/2 if n_positive and n_negative else None
    return {'balanced_error': balanced, 'false_negative_rate': false_negative,
            'false_positive_rate': false_positive, 'disagreement': float(np.mean(prediction != success)),
            'successes': n_positive, 'failures': n_negative}


def mixed_states(network, config, split):
    count = config[f'{split}_states']
    values = sample_law(network, count, config['seeds'][split], config['baseline_alpha_scale'],
                        config['gaussian_copula_correlation'])
    frame, members = candidates()
    rng = np.random.default_rng(config['seeds'][split]+1000)
    forms = np.repeat(np.arange(4), count//4)
    selected = np.full(count, -1, dtype=int)
    for form in range(1, 4):
        ids = np.flatnonzero(frame.form.to_numpy() == FORMS[form])
        for row in np.flatnonzero(forms == form):
            candidate = int(rng.choice(ids))
            values[row] = apply_stress(values[row:row+1], members[candidate],
                config['physical_alpha_cap'], config['common_fractional_stress'], 'fractional')[0]
            selected[row] = candidate
    return values, forms, selected


def normalised_features(network, values, horizons, kernel, order, window='uniform'):
    healthy = propagation_features(network, np.zeros((1, network.m)), horizons, kernel, order, window)[0]
    assert np.all(healthy > 1+1e-10)
    raw = propagation_features(network, values, horizons, kernel, order, window)
    return (raw-1)/(healthy-1), healthy


def local_retention(network, values):
    base = generators(network, np.zeros((1, network.m)), False)[0]
    denominator = np.dot(np.sum(base*base, axis=0), network.source_weights)
    out = np.empty(len(values))
    for start in range(0, len(values), 128):
        h = generators(network, values[start:start+128], False)
        out[start:start+len(h)] = np.sum(h*h, axis=1) @ network.source_weights / denominator
    return out


def quadrature_order(network, config, horizons, kernels, window='uniform'):
    audit = sample_law(network, 64, config['seeds']['audit'], .55, .35)
    audit = np.concatenate([np.zeros((1, network.m)), audit])
    checks = []
    if window == 'instantaneous':
        return 1, []
    for order in (32, 64, 128):
        maximum = 0.
        for kernel in kernels:
            low, _ = normalised_features(network, audit, np.array(horizons), kernel, order, window)
            high, _ = normalised_features(network, audit, np.array(horizons), kernel, order*2, window)
            error = float(np.max(abs(high-low)))
            maximum = max(maximum, error)
            checks.append({'kernel': kernel, 'window': window, 'order': order, 'double_order_error': error})
        if maximum <= config['quadrature_tolerance']:
            return order, checks
    raise ArithmeticError(f'Quadrature convergence failed: {checks}')


def calibrate():
    config, fingerprint = config_and_fingerprint()
    completion = OUTPUTS / 'calibration_completion.json'
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved['fingerprint'] != fingerprint:
            raise RuntimeError('Changed calibration dependencies require a recorded version decision.')
        print('Matching completed calibration verified; no rerun.', flush=True)
        return
    start = perf_counter()
    network = load_sioux()
    horizons = np.array(config['horizons'])
    order, integration_checks = quadrature_order(network, config, horizons, config['kernels'])
    train, train_forms, train_candidates = mixed_states(network, config, 'training')
    validation, validation_forms, validation_candidates = mixed_states(network, config, 'calibration')
    c0 = service_cost(network, np.zeros((1, network.m)))[0]
    train_ratios = service_cost(network, train)/c0
    validation_ratios = service_cost(network, validation)/c0
    save_npz(OUTPUTS / 'calibration_states.npz', training=train, calibration=validation,
             training_forms=train_forms, calibration_forms=validation_forms,
             training_candidates=train_candidates, calibration_candidates=validation_candidates,
             training_ratios=train_ratios, calibration_ratios=validation_ratios,
             fingerprint=np.array(fingerprint))
    rows, selected, thresholds_by_method = [], {}, {}
    write_json(OUTPUTS / 'calibration_progress.json', {'status': 'running', 'pid': os.getpid(),
               'fingerprint': fingerprint, 'completed_kernels': [], 'quadrature_order': order})
    for k, kernel in enumerate(config['kernels']):
        path = OUTPUTS / f'calibration_{kernel}.npz'
        if path.exists():
            saved = np.load(path)
            assert str(saved['fingerprint']) == fingerprint
            train_score, validation_score = saved['training'], saved['calibration']
            local_low, local_high, healthy = saved['local_low'], saved['local_high'], saved['healthy']
        else:
            train_score, healthy = normalised_features(network, train, horizons, kernel, order)
            validation_score, _ = normalised_features(network, validation, horizons, kernel, order)
            local_low, _ = normalised_features(network, validation, horizons*.95, kernel, order)
            local_high, _ = normalised_features(network, validation, horizons*1.05, kernel, order)
            save_npz(path, training=train_score, calibration=validation_score,
                     local_low=local_low, local_high=local_high, healthy=healthy, fingerprint=np.array(fingerprint))
        method_rows, fitted = [], []
        ytrain = train_ratios <= config['primary_service_ratio']
        yvalidation = validation_ratios <= config['primary_service_ratio']
        for j, horizon in enumerate(horizons):
            threshold, training_error = best_threshold(train_score[:, j], ytrain)
            fitted.append({str(limit): best_threshold(train_score[:, j], train_ratios <= limit)[0]
                           for limit in config['service_ratio_limits']})
            prediction = validation_score[:, j] >= threshold
            metrics = error_metrics(prediction, yvalidation)
            strata = [error_metrics(prediction[validation_forms == form], yvalidation[validation_forms == form])
                      for form in range(4)]
            local_metrics = [error_metrics(v[:, j] >= threshold, yvalidation) for v in (local_low, local_high)]
            local_flips = [float(np.mean((v[:, j] >= threshold) != prediction)) for v in (local_low, local_high)]
            adequate = all(min(s['successes'], s['failures']) >= config['eligibility']['per_stratum_min_class_count'] for s in strata)
            stratum_max = max(s['balanced_error'] for s in strata if s['balanced_error'] is not None)
            eligible = (adequate and metrics['balanced_error'] <= config['eligibility']['calibration_balanced_error_max']
                        and stratum_max <= config['eligibility']['per_stratum_balanced_error_max']
                        and max(s['balanced_error'] for s in local_metrics) <= config['eligibility']['local_balanced_error_max']
                        and max(local_flips) <= config['eligibility']['local_label_disagreement_max'])
            row = {'kernel': kernel, 'horizon': float(horizon), 'threshold': threshold,
                   'training_balanced_error': training_error, **metrics,
                   'max_stratum_balanced_error': stratum_max, 'adequate_class_counts': adequate,
                   'max_local_balanced_error': max(s['balanced_error'] for s in local_metrics),
                   'max_local_label_disagreement': max(local_flips), 'eligible': bool(eligible)}
            for form, metric in zip(FORMS, strata):
                row[form+'_balanced_error'] = metric['balanced_error']
            rows.append(row)
            method_rows.append(row)
        eligible_rows = [row for row in method_rows if row['eligible']]
        chosen = min(eligible_rows or method_rows, key=lambda x: (x['balanced_error'], x['horizon']))
        j = config['horizons'].index(chosen['horizon'])
        selected[kernel] = {**chosen, 'index': j, 'thresholds': fitted[j], 'healthy_dpc': float(healthy[j])}
        thresholds_by_method[kernel] = fitted
        pd.DataFrame(rows).to_csv(OUTPUTS / 'calibration_horizons.csv', index=False)
        write_json(OUTPUTS / 'calibration_progress.json', {'status': 'running', 'pid': os.getpid(),
                   'fingerprint': fingerprint, 'completed_kernels': config['kernels'][:k+1],
                   'elapsed_seconds': perf_counter()-start})
        print(f'Calibration {kernel}: L={chosen["horizon"]}, eligible={chosen["eligible"]}, balanced error={chosen["balanced_error"]:.6f}', flush=True)
    comparison = {}
    primary_l = selected['adjacency_quantum']['horizon']
    for window in ('late_triangular', 'instantaneous'):
        q, check = quadrature_order(network, config, [primary_l], ['adjacency_quantum'], window)
        integration_checks += check
        train_score, healthy = normalised_features(network, train, np.array([primary_l]), 'adjacency_quantum', q, window)
        validation_score, _ = normalised_features(network, validation, np.array([primary_l]), 'adjacency_quantum', q, window)
        thresholds = {str(limit): best_threshold(train_score[:, 0], train_ratios <= limit)[0] for limit in config['service_ratio_limits']}
        comparison[window] = {'horizon': primary_l, 'order': q, 'healthy_dpc': float(healthy[0]),
                             'thresholds': thresholds, **error_metrics(validation_score[:, 0] >= thresholds[str(config['primary_service_ratio'])],
                                                                     validation_ratios <= config['primary_service_ratio'])}
    training_local, validation_local = local_retention(network, train), local_retention(network, validation)
    local_thresholds = {str(limit): best_threshold(training_local, train_ratios <= limit)[0] for limit in config['service_ratio_limits']}
    comparison['local_retention'] = {'thresholds': local_thresholds,
        **error_metrics(validation_local >= local_thresholds[str(config['primary_service_ratio'])], validation_ratios <= config['primary_service_ratio'])}
    result = {'status': 'complete', 'fingerprint': fingerprint, 'selection_locked_before_test': True,
              'selected': selected, 'comparators': comparison, 'thresholds_all_horizons': thresholds_by_method,
              'quadrature_order': order, 'integration_checks': integration_checks,
              'training_states': len(train), 'calibration_states': len(validation), 'elapsed_seconds': perf_counter()-start}
    write_json(completion, result)
    write_json(OUTPUTS / 'calibration_progress.json', {'status': 'complete', 'fingerprint': fingerprint,
               'completed_kernels': config['kernels'], 'elapsed_seconds': perf_counter()-start})
    print(f'Calibration complete in {perf_counter()-start:.1f}s. Test scenarios have not been generated.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['calibrate'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        calibrate()


if __name__ == '__main__':
    main()
