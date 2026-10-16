"""Independent small examples for the frozen statistical analysis definitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from analyse_service_validation import (bootstrap_metrics, importance,
    ordering_metrics, rank_correlation, threshold_for, top_weights)
from service_validation import ROOT, OUTPUTS, write_json


def main():
    checks = []
    weights, ties = top_weights(np.array([5., 4., 3., 2., 1., 1.]))
    assert np.array_equal(weights, [1., 1., 1., 1., .5, .5]) and ties == 2
    checks.append('top-five tie mass is shared without arbitrary tie-breaking')
    result = ordering_metrics(np.array([3., 2., 1.]), np.array([-3., -2., -1.]))
    assert result['reversed_pairs'] == result['strict_reference_pairs'] == 3
    assert abs(result['spearman']+1) < 1e-12
    checks.append('all strict pair reversals and reverse Spearman order')
    result = ordering_metrics(np.array([3., 2., 1.]), np.zeros(3))
    assert result['proxy_tied_pairs'] == 3 and result['reversed_pairs'] == 0
    assert result['spearman'] is None and rank_correlation(np.ones(3), np.ones(3)) is None
    checks.append('ties are not reversals and constant rank correlations are undefined')
    success = np.array([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]], dtype=bool)
    means, standard_errors = importance(success)
    assert np.array_equal(means, [.5, 0.])
    assert np.allclose(standard_errors, np.sqrt([1/12, 1/6]))
    checks.append('importance uses paired scenario differences, including negative changes')
    rng = np.random.default_rng(2026101612)
    truth, prediction = rng.random((2, 17, 41)) < .6
    ti, _ = importance(truth)
    pi, _ = importance(prediction)
    e = np.mean(truth != prediction, axis=1)
    assert np.all(abs(ti-pi) <= e[0]+e[1:]+1e-12)
    i, j = np.triu_indices(len(ti), 1)
    assert np.all(abs((ti[i]-ti[j])-(pi[i]-pi[j])) <= e[1:][i]+e[1:][j]+1e-12)
    checks.append('sample analogues of baseline/intervention and pair-order transfer bounds')
    alpha, count = .05/798, 1024
    assert np.isclose(beta.ppf(1-alpha, 1, count), 1-alpha**(1/count))
    checks.append('zero-disagreement Clopper-Pearson bound matches its analytic expression')
    forms = np.array(['edge', 'corridor', 'region']*2)
    reference = rng.random((7, 32)) < .6
    rows = bootstrap_metrics(reference, reference, forms,
        {'bootstrap_seed': 11, 'bootstrap_replications': 50})
    assert all(row['mae_low'] == row['mae_high'] == 0 for row in rows)
    assert all(np.isclose(row['top_five_low'], 1) and np.isclose(row['top_five_high'], 1) for row in rows)
    checks.append('scenario-block bootstrap preserves identical-method equality')
    calibration = json.loads((OUTPUTS/'calibration_completion.json').read_text(encoding='utf-8'))
    selected = calibration['selected']['adjacency_quantum']['thresholds']['1.1']
    assert all(threshold_for(name, calibration, 1.1) == selected
        for name in ('adjacency_0.90', 'adjacency_1.00', 'first_order', 'second_order'))
    checks.append('horizon and approximation checks retain the locked exact-DPC threshold')
    script = Path(__file__).with_name('analyse_service_validation.py')
    write_json(ROOT/'audit/service_analysis_unit_tests.json', {'status': 'pass',
        'analysis_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks': checks, 'test_count': len(checks), 'heldout_outcomes_accessed': False})
    print(f'Passed {len(checks)} statistical checks without reading held-out outcomes.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
