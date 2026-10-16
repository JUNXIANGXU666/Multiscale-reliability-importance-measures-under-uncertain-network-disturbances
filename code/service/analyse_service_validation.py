"""Analyse the complete frozen service experiments without selecting models on test data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata
from threadpoolctl import threadpool_limits

from service_validation import ROOT, OUTPUTS, write_json
from confirmatory_validation import candidates, error_metrics
from heldout_validation import load_design


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def rank_correlation(x, y):
    a, b = rankdata(x), rankdata(y)
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator > 0 else None


def top_weights(values, k=5):
    k = min(k, len(values))
    cut = np.sort(values)[-k]
    above = values > cut+1e-12
    equal = np.abs(values-cut) <= 1e-12
    weights = above.astype(float)
    weights[equal] = (k-int(above.sum()))/int(equal.sum())
    assert abs(weights.sum()-k) < 1e-10
    return weights, int(equal.sum())


def ordering_metrics(reference, estimate):
    i, j = np.triu_indices(len(reference), 1)
    dx, dy = reference[i]-reference[j], estimate[i]-estimate[j]
    strict = abs(dx) > 1e-12
    opposite = strict & (abs(dy) > 1e-12) & (np.sign(dx) != np.sign(dy))
    tied = strict & (abs(dy) <= 1e-12)
    a, ties_a = top_weights(reference)
    b, ties_b = top_weights(estimate)
    return {'spearman': rank_correlation(reference, estimate),
            'strict_reference_pairs': int(strict.sum()), 'reversed_pairs': int(opposite.sum()),
            'proxy_tied_pairs': int(tied.sum()),
            'reversal_fraction': float(opposite.sum()/strict.sum()) if strict.any() else None,
            'top_five_overlap': float(np.minimum(a, b).sum()/a.sum()),
            'reference_boundary_ties': ties_a, 'proxy_boundary_ties': ties_b}


def importance(success):
    changes = success[0:1].astype(float)-success[1:].astype(float)
    return changes.mean(axis=1), changes.std(axis=1, ddof=1)/np.sqrt(changes.shape[1])


def threshold_for(name, calibration, limit):
    key = str(limit)
    if name.startswith('adjacency_') or name in ('first_order', 'second_order', 'adjacency_quantum'):
        return calibration['selected']['adjacency_quantum']['thresholds'][key]
    if name in calibration['selected']:
        return calibration['selected'][name]['thresholds'][key]
    return calibration['comparators'][name]['thresholds'][key]


def verified_scope(scope, fingerprint):
    path = OUTPUTS/f'{scope}_completion.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    assert meta['status'] == 'complete' and meta['fingerprint'] == fingerprint
    assert meta['next_row'] == meta['total_rows']
    for artifact in meta['artifacts']:
        target = ROOT/artifact['path']
        assert hashlib.sha256(target.read_bytes()).hexdigest() == artifact['sha256']
    return meta, np.load(OUTPUTS/f'{scope}_features.npy', mmap_mode='r'), np.load(OUTPUTS/f'{scope}_costs.npy', mmap_mode='r')


def bootstrap_metrics(reference, prediction, forms, config):
    count = reference.shape[1]
    rng = np.random.default_rng(config['bootstrap_seed'])
    weights = rng.multinomial(count, np.full(count, 1/count), config['bootstrap_replications'])/count
    truth_i = (reference[0:1].astype(float)-reference[1:])@weights.T
    estimate_i = (prediction[0:1].astype(float)-prediction[1:])@weights.T
    rows = []
    for form in ('all', 'edge', 'corridor', 'region'):
        mask = np.ones(len(forms), dtype=bool) if form == 'all' else forms == form
        errors = np.mean(abs(truth_i[mask]-estimate_i[mask]), axis=0)
        correlations, overlaps = [], []
        for b in range(config['bootstrap_replications']):
            x, y = truth_i[mask, b], estimate_i[mask, b]
            correlation = rank_correlation(x, y)
            if correlation is not None:
                correlations.append(correlation)
            wx, _ = top_weights(x)
            wy, _ = top_weights(y)
            overlaps.append(np.minimum(wx, wy).sum()/wx.sum())
        row = {'form': form, 'replications': config['bootstrap_replications'],
               'mae_low': float(np.quantile(errors, .025)), 'mae_high': float(np.quantile(errors, .975)),
               'top_five_low': float(np.quantile(overlaps, .025)), 'top_five_high': float(np.quantile(overlaps, .975)),
               'valid_correlation_replications': len(correlations)}
        if correlations:
            row.update(spearman_low=float(np.quantile(correlations, .025)), spearman_high=float(np.quantile(correlations, .975)))
        rows.append(row)
    return rows


def primary_analysis(config, calibration, statistical, fingerprint):
    frame, _ = candidates()
    forms = frame.form.to_numpy()
    meta, features, costs = verified_scope('primary', fingerprint)
    names = [s['name'] for s in meta['feature_specs']]
    selected_names = ['adjacency_1.00', 'laplacian_quantum', 'classical_diffusion', 'local_retention',
                      'late_triangular', 'instantaneous', 'first_order', 'second_order']
    rows, candidate_rows, interval_rows, bootstrap_rows = [], [], [], []
    ratio = costs[:, :, 0]/meta['baseline_cost']
    assert np.min(costs[1:, :, 0]-costs[0, :, 0]) >= -1e-10
    for limit in statistical['service_limits']:
        reference = ratio <= limit
        true_i, true_se = importance(reference)
        exact_prediction = features[:, :, names.index('adjacency_1.00')] >= threshold_for('adjacency_1.00', calibration, limit)
        exact_i, _ = importance(exact_prediction)
        for name in selected_names:
            score = features[:, :, names.index(name)]
            prediction = score >= threshold_for(name, calibration, limit)
            estimated_i, estimated_se = importance(prediction)
            disagreement = np.mean(prediction != reference, axis=1)
            assert np.max(abs(estimated_i-true_i)-disagreement[0]-disagreement[1:]) < 1e-12
            for form in ('all', 'edge', 'corridor', 'region'):
                mask = np.ones(len(frame), dtype=bool) if form == 'all' else forms == form
                metrics = error_metrics(prediction[1:][mask].ravel(), reference[1:][mask].ravel())
                row = {'method': name, 'service_limit': limit, 'form': form, 'candidates': int(mask.sum()),
                       'importance_mae': float(np.mean(abs(estimated_i[mask]-true_i[mask]))),
                       'importance_rmse': float(np.sqrt(np.mean((estimated_i[mask]-true_i[mask])**2))),
                       'maximum_importance_error': float(np.max(abs(estimated_i[mask]-true_i[mask]))),
                       'negative_importances': int(np.sum(estimated_i[mask] < -1e-12)),
                       'exact_dpc_importance_mae': float(np.mean(abs(estimated_i[mask]-exact_i[mask]))),
                       **metrics, **ordering_metrics(true_i[mask], estimated_i[mask])}
                rows.append(row)
            rows.append({'method': name, 'service_limit': limit, 'form': 'baseline', 'candidates': 1,
                         **error_metrics(prediction[0], reference[0])})
            for j in range(len(frame)):
                candidate_rows.append({'method': name, 'service_limit': limit, 'candidate': frame.iloc[j].candidate_id,
                    'form': forms[j], 'service_importance': true_i[j], 'service_se': true_se[j],
                    'estimated_importance': estimated_i[j], 'estimate_se': estimated_se[j],
                    'event_disagreement': disagreement[j+1], 'empirical_transfer_bound': disagreement[0]+disagreement[j+1]})
            if limit == statistical['primary_service_limit']:
                for row in bootstrap_metrics(reference, prediction, forms, statistical):
                    bootstrap_rows.append({'method': name, **row})
                if name == 'adjacency_1.00':
                    count = reference.shape[1]
                    errors = np.sum(prediction != reference, axis=1)
                    alpha = (1-statistical['confidence_level'])/len(errors)
                    upper = np.ones(len(errors))
                    valid = errors < count
                    upper[valid] = beta.ppf(1-alpha, errors[valid]+1, count-errors[valid])
                    for j in range(len(frame)):
                        interval_rows.append({'candidate': frame.iloc[j].candidate_id, 'form': forms[j],
                            'disagreement_count': int(errors[j+1]), 'scenario_count': count,
                            'disagreement_upper': upper[j+1], 'baseline_disagreement_upper': upper[0],
                            'service_importance_error_upper': upper[0]+upper[j+1],
                            'observed_importance_error': abs(estimated_i[j]-true_i[j])})
    pd.DataFrame(rows).to_csv(OUTPUTS/'service_validation_summary.csv', index=False)
    pd.DataFrame(candidate_rows).to_csv(OUTPUTS/'service_validation_candidates.csv', index=False)
    pd.DataFrame(interval_rows).to_csv(OUTPUTS/'service_transfer_bounds.csv', index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUTPUTS/'service_validation_intervals.csv', index=False)
    return {'scope': 'primary', 'candidate_count': len(frame), 'scenarios': reference.shape[1],
            'model_count': len(selected_names), 'service_limits': statistical['service_limits']}


def sensitivity_analysis(calibration, statistical, fingerprint):
    frame, _ = candidates()
    forms = frame.form.to_numpy()
    meta, features, costs = verified_scope('primary', fingerprint)
    specs = meta['feature_specs']
    names = [s['name'] for s in specs]
    reference_id = names.index('adjacency_1.00')
    zeta = threshold_for('adjacency_1.00', calibration, statistical['primary_service_limit'])
    base_prediction = features[:, :, reference_id] >= zeta
    base_i, _ = importance(base_prediction)
    service = costs[:, :, 0]/meta['baseline_cost'] <= statistical['primary_service_limit']
    raw_threshold = 1+zeta*(meta['healthy_dpc'][reference_id]-1)
    rows = []
    for j, spec in enumerate(specs):
        if not spec['name'].startswith('adjacency_'):
            continue
        for rule in ('relative', 'raw'):
            threshold = zeta if rule == 'relative' else (raw_threshold-1)/(meta['healthy_dpc'][j]-1)
            prediction = features[:, :, j] >= threshold
            current_i, _ = importance(prediction)
            for form in ('all', 'edge', 'corridor', 'region'):
                mask = np.ones(len(frame), dtype=bool) if form == 'all' else forms == form
                rows.append({'rule': rule, 'factor': spec['factor'], 'form': form,
                    'mean_importance_change': float(np.mean(abs(current_i[mask]-base_i[mask]))),
                    'maximum_importance_change': float(np.max(abs(current_i[mask]-base_i[mask]))),
                    'sign_changes': int(np.sum(np.sign(current_i[mask]) != np.sign(base_i[mask]))),
                    'label_disagreement': float(np.mean(prediction[1:][mask] != base_prediction[1:][mask])),
                    **error_metrics(prediction[1:][mask].ravel(), service[1:][mask].ravel()),
                    **ordering_metrics(base_i[mask], current_i[mask])})
    pd.DataFrame(rows).to_csv(OUTPUTS/'horizon_sensitivity_summary.csv', index=False)


def allocation_analysis(calibration, statistical, fingerprint):
    frame, _ = candidates()
    forms = frame.form.to_numpy()
    meta, features, costs = verified_scope('allocation', fingerprint)
    n = len(frame)
    rows, candidate_rows = [], []
    for limit in statistical['service_limits']:
        reference = costs[:, :, 0]/meta['baseline_cost'] <= limit
        prediction = features[:, :, 0] >= threshold_for('adjacency_quantum', calibration, limit)
        service_i, _ = importance(reference)
        proxy_i, _ = importance(prediction)
        for metric, values in (('service', service_i), ('DPC', proxy_i)):
            a, b = values[:n], values[n:]
            for form in ('all', 'edge', 'corridor', 'region'):
                mask = np.ones(n, dtype=bool) if form == 'all' else forms == form
                rows.append({'metric': metric, 'service_limit': limit, 'form': form,
                    'mean_absolute_allocation_difference': float(np.mean(abs(a[mask]-b[mask]))),
                    'maximum_absolute_allocation_difference': float(np.max(abs(a[mask]-b[mask]))),
                    'mean_headroom_importance': float(np.mean(a[mask])), 'mean_uniform_importance': float(np.mean(b[mask])),
                    **ordering_metrics(a[mask], b[mask])})
            for j in range(n):
                candidate_rows.append({'candidate': frame.iloc[j].candidate_id, 'form': forms[j], 'metric': metric,
                    'service_limit': limit, 'headroom_importance': a[j], 'uniform_importance': b[j]})
        edge_mask = forms == 'edge'
        assert np.max(abs(service_i[:n][edge_mask]-service_i[n:][edge_mask])) < 1e-12
        assert np.max(abs(proxy_i[:n][edge_mask]-proxy_i[n:][edge_mask])) < 1e-12
    pd.DataFrame(rows).to_csv(OUTPUTS/'allocation_comparison_summary.csv', index=False)
    pd.DataFrame(candidate_rows).to_csv(OUTPUTS/'allocation_comparison_candidates.csv', index=False)


def direction_analysis(calibration, statistical, fingerprint):
    frame, _ = candidates()
    forms = frame.form.to_numpy()
    meta, features, costs = verified_scope('direction', fingerprint)
    ratios = costs/meta['baseline_cost']
    rows, candidate_rows = [], []
    for limit in statistical['service_limits']:
        directed = ratios[:, :, 0] <= limit
        symmetric = ratios[:, :, 1] <= limit
        proxy = features[:, :, 0] >= threshold_for('adjacency_quantum', calibration, limit)
        directed_i, directed_se = importance(directed)
        symmetric_i, _ = importance(symmetric)
        proxy_i, _ = importance(proxy)
        for form in ('all', 'edge', 'corridor', 'region'):
            mask = np.ones(len(frame), dtype=bool) if form == 'all' else forms == form
            rows.append({'service_limit': limit, 'form': form,
                'symmetrisation_importance_mae': float(np.mean(abs(directed_i[mask]-symmetric_i[mask]))),
                'maximum_symmetrisation_importance_error': float(np.max(abs(directed_i[mask]-symmetric_i[mask]))),
                'DPC_directed_importance_mae': float(np.mean(abs(directed_i[mask]-proxy_i[mask]))),
                'DPC_symmetric_importance_mae': float(np.mean(abs(symmetric_i[mask]-proxy_i[mask]))),
                'mean_relative_cost_bias': float(np.mean((costs[1:, :, 1][mask]-costs[1:, :, 0][mask])/costs[1:, :, 0][mask])),
                'event_disagreement': float(np.mean(directed[1:][mask] != symmetric[1:][mask])),
                **ordering_metrics(directed_i[mask], symmetric_i[mask])})
        for j in range(len(frame)):
            candidate_rows.append({'candidate': frame.iloc[j].candidate_id, 'form': forms[j], 'service_limit': limit,
                'directed_importance': directed_i[j], 'directed_se': directed_se[j],
                'symmetric_importance': symmetric_i[j], 'DPC_importance': proxy_i[j]})
    pd.DataFrame(rows).to_csv(OUTPUTS/'direction_bias_summary.csv', index=False)
    pd.DataFrame(candidate_rows).to_csv(OUTPUTS/'direction_bias_candidates.csv', index=False)


def main():
    config, calibration, fingerprint = load_design()
    plan = ROOT/'plans/statistical_analysis.json'
    statistical = json.loads(plan.read_text(encoding='utf-8'))
    analysis_fingerprint = hashlib.sha256(fingerprint.encode()+plan.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    completion_path = OUTPUTS/'service_analysis_completion.json'
    if completion_path.exists():
        prior = json.loads(completion_path.read_text(encoding='utf-8'))
        assert prior['fingerprint'] == analysis_fingerprint
        for artifact in prior['artifacts']:
            assert hashlib.sha256((ROOT/artifact['path']).read_bytes()).hexdigest() == artifact['sha256']
        print('Matching completed analysis verified, no regeneration.', flush=True)
        return
    for scope in ('primary', 'allocation', 'direction'):
        verified_scope(scope, fingerprint)
    primary = primary_analysis(config, calibration, statistical, fingerprint)
    sensitivity_analysis(calibration, statistical, fingerprint)
    allocation_analysis(calibration, statistical, fingerprint)
    direction_analysis(calibration, statistical, fingerprint)
    paths = [OUTPUTS/name for name in ('service_validation_summary.csv', 'service_validation_candidates.csv',
        'service_transfer_bounds.csv', 'service_validation_intervals.csv', 'horizon_sensitivity_summary.csv',
        'allocation_comparison_summary.csv', 'allocation_comparison_candidates.csv', 'direction_bias_summary.csv',
        'direction_bias_candidates.csv')]
    write_json(completion_path, finite_json({'status': 'complete', 'fingerprint': analysis_fingerprint,
        'experiment_fingerprint': fingerprint, 'primary': primary,
        'artifacts': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'),
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}))
    print('All declared service analyses completed and hashed.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
