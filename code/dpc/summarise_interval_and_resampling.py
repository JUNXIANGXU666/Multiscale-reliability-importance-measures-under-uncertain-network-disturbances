"""Summarise interval and resampling."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from kernel_dependence_experiments import (
    RESULTS, SEED, equal_family_resampling, grid_network, kernel_matrix,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inputs = [
        RESULTS / "certificate_candidate_intervals.csv",
        RESULTS / "certificate_quantity_summary.csv",
        RESULTS / "pathwise_equal_total_importance.csv",
        RESULTS / "sioux_equal_count_maximum_sample.csv",
        RESULTS / "sioux_equal_family_count_resampling.csv",
    ]
    hashes = {p.name: sha256(p) for p in inputs}
    intervals = pd.read_csv(inputs[0])
    previous = pd.read_csv(inputs[1])
    previous = previous[previous.scope == "all"].set_index(["case", "condition_id", "allowance_type"])
    condition_rows = []
    for key, group in intervals.groupby(["case", "condition_id", "allowance_type"], sort=True):
        values = group.exact_importance.to_numpy(float)
        low, high = group.lower.to_numpy(float), group.upper.to_numpy(float)
        n = len(values)
        first, second = np.triu_indices(n, k=1)
        difference = values[first] - values[second]
        strict = np.abs(difference) > 1e-12
        higher = np.where(difference > 0, first, second)
        lower = np.where(difference > 0, second, first)
        pair_resolved = int(np.sum(strict & (low[higher] > high[lower])))
        false_pairs = int(np.sum(strict & (low[lower] > high[higher])))
        resolved_sign = (low > 0) | (high < 0)
        order = np.argsort(-values, kind="stable")
        unique_top = n == 1 or values[order[0]] > values[order[1]] + 1e-12
        top1 = bool(unique_top and (n == 1 or low[order[0]] > np.max(high[order[1:]])))
        applicable = n > 5
        unique_top5_set = bool(applicable and values[order[4]] > values[order[5]] + 1e-12)
        top5_set = bool(unique_top5_set and np.min(low[order[:5]]) > np.max(high[order[5:]]))
        unique_top5_order = bool(unique_top5_set and np.all(np.diff(values[order[:5]]) < -1e-12))
        top5_order = bool(top5_set and np.all(low[order[:4]] > high[order[1:5]]))
        row = {
            "case": key[0], "condition_id": key[1], "allowance_type": key[2],
            "candidate_count": n, "sign_resolved_count": int(np.sum(resolved_sign)),
            "strict_pair_count": int(np.sum(strict)), "pair_resolved_count": pair_resolved,
            "top1_unique_truth": unique_top, "top1_resolved": top1,
            "top5_applicable": applicable, "top5_unique_set_truth": unique_top5_set,
            "top5_unique_order_truth": unique_top5_order, "top5_resolved": top5_set,
            "top5_order_resolved": top5_order,
            "interval_width_mean": float(np.mean(high-low)),
            "interval_width_median": float(np.median(high-low)),
        }
        old = previous.loc[key]
        for field in ("candidate_count", "sign_resolved_count", "strict_pair_count", "pair_resolved_count", "top1_unique_truth", "top1_resolved", "top5_applicable", "top5_resolved"):
            assert row[field] == old[field], (key, field, row[field], old[field])
        assert np.isclose(row["interval_width_mean"], old.interval_width_mean, atol=1e-14)
        assert false_pairs == 0
        assert np.all((low <= values+1e-12) & (high >= values-1e-12))
        assert np.all(group.sign_status.to_numpy()[resolved_sign] == group.true_sign.to_numpy()[resolved_sign])
        condition_rows.append(row)
    conditions = pd.DataFrame(condition_rows)
    conditions.to_csv(RESULTS / "certificate_topk_condition_summary.csv", index=False)
    assert set(conditions["case"]) == {"cycle5", "diamond4", "grid5x5", "sioux_falls"}
    conditions["network_group"] = conditions["case"].where(~conditions["case"].isin(["cycle5", "diamond4"]), "motifs")
    intervals["network_group"] = intervals["case"].where(~intervals["case"].isin(["cycle5", "diamond4"]), "motifs")
    aggregates = []
    for (network, allowance), group in conditions.groupby(["network_group", "allowance_type"], sort=True):
        details = intervals[(intervals.network_group == network) & (intervals.allowance_type == allowance)]
        signs = int(group.sign_resolved_count.sum())
        candidates = int(group.candidate_count.sum())
        pairs = int(group.pair_resolved_count.sum())
        strict_pairs = int(group.strict_pair_count.sum())
        aggregates.append({
            "network_group": network, "allowance_type": allowance,
            "condition_count": len(group), "sign_resolved_count": signs,
            "candidate_evaluations": candidates, "sign_resolved_rate": signs/candidates,
            "pair_resolved_count": pairs, "strict_pair_evaluations": strict_pairs,
            "pair_resolved_rate": pairs/strict_pairs if strict_pairs else 0.,
            "top1_resolved_count": int(group.top1_resolved.sum()),
            "top1_condition_count": len(group),
            "top1_unique_truth_count": int(group.top1_unique_truth.sum()),
            "top5_set_resolved_count": int(group.top5_resolved.sum()),
            "top5_condition_count": int(group.top5_applicable.sum()),
            "top5_unique_set_truth_count": int(group.top5_unique_set_truth.sum()),
            "top5_order_resolved_count": int(group.top5_order_resolved.sum()),
            "top5_unique_order_truth_count": int(group.top5_unique_order_truth.sum()),
            "interval_width_mean": float(details.width.mean()),
            "interval_width_median": float(details.width.median()),
        })
    aggregate = pd.DataFrame(aggregates)
    assert set(aggregate.loc[aggregate.network_group == "motifs", "condition_count"]) == {420}
    aggregate.to_csv(RESULTS / "certificate_quantity_aggregate.csv", index=False)

    pathwise = pd.read_csv(inputs[2])
    sioux = pathwise[(pathwise["case"] == "sioux_falls") & (pathwise.stress_mode == "pathwise_equal_total")]
    forms = ("edge", "corridor", "region")
    values = {form: sioux.loc[sioux.form == form, "direct_empirical_importance"].to_numpy(float) for form in forms}
    summary, maxima, scores = equal_family_resampling(values)
    old_maxima = pd.read_csv(inputs[3])
    pd.testing.assert_frame_equal(maxima.iloc[:2000].reset_index(drop=True), old_maxima, check_exact=True)
    old_summary = pd.read_csv(inputs[4])
    assert np.allclose(summary.fractional_win_probability, old_summary.fractional_win_probability, atol=1e-14)
    assert np.allclose(scores[list(forms)].sum(axis=1), 1)
    summary.to_csv(inputs[4], index=False)
    scores.to_csv(RESULTS / "sioux_equal_count_fractional_scores.csv", index=False)
    maxima.to_csv(RESULTS / "sioux_equal_count_all_maxima.csv", index=False)

    network = grid_network(5, 5, 20260716)
    alpha = np.linspace(0.01, 0.18, len(network.edges))
    generator = kernel_matrix(network, alpha, "classical_diffusion")
    unperturbed = kernel_matrix(network, np.zeros_like(alpha), "classical_diffusion")
    probabilities = expm(-generator * 1.8)
    reference = expm(-unperturbed * 1.8)
    assert np.max(np.abs(probabilities.sum(axis=0)-1)) < 1e-12
    assert np.min(probabilities) > -1e-12
    participation = 1/np.sum(probabilities**2, axis=0)
    baseline_participation = 1/np.sum(reference**2, axis=0)
    assert np.max(np.abs(participation-baseline_participation)) > 1e-3
    audit = {
        "input_sha256": hashes,
        "python": platform.python_version(), "resampling_seed": SEED+31,
        "all_saved_quantity_summaries_reproduced": True,
        "top_five_membership_and_internal_order_reported_separately": True,
        "pooled_widths_weight_each_candidate_condition_once": True,
        "first_2000_resampling_maxima_unchanged_exactly": True,
        "win_probabilities_unchanged": True,
        "fractional_score_standard_errors": dict(zip(summary.form, summary.monte_carlo_standard_error)),
        "diffusion_column_sum_maximum_error": float(np.max(np.abs(probabilities.sum(axis=0)-1))),
        "diffusion_perturbation_changes_participation": float(np.max(np.abs(participation-baseline_participation))),
        "pass": True,
    }
    (RESULTS / "interval_resampling_correction_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
