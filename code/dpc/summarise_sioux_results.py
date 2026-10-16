from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "sioux_outputs" / "results"


def scalar(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def record(row: pd.Series, columns: list[str]) -> dict[str, object]:
    return {column: scalar(row[column]) for column in columns}


def main() -> None:
    sioux_dpc = pd.read_csv(RESULTS / "sioux_weighted_dpc_importance.csv")
    sioux_uniform = pd.read_csv(RESULTS / "sioux_uniform_source_edge_importance.csv")
    rank_comparisons = pd.read_csv(RESULTS / "sioux_rank_comparisons.csv")
    quadratic = pd.read_csv(RESULTS / "sioux_quadratic_accuracy.csv")
    metadata = json.loads(
        (RESULTS / "sioux_benchmark_metadata.json").read_text(encoding="utf-8")
    )
    sensitivity = pd.read_csv(RESULTS / "full_parameter_sensitivity.csv")
    sensitivity_metrics = pd.read_csv(
        RESULTS / "full_parameter_stability_metrics.csv"
    )
    dependence_summary = pd.read_csv(RESULTS / "dependence_family_summary.csv")
    dependence_stability = pd.read_csv(
        RESULTS / "dependence_family_stability.csv"
    )
    decomposition = pd.read_csv(RESULTS / "stress_moment_decomposition.csv")
    fair = pd.read_csv(RESULTS / "equal_total_expected_stress_importance.csv")
    certificates = pd.read_csv(RESULTS / "certificate_rate_map.csv")
    matrix_free = pd.read_csv(RESULTS / "sioux_matrix_free_runtime.csv")
    storage = pd.read_csv(RESULTS / "interaction_matrix_storage_projection.csv")
    scoring = json.loads(
        (RESULTS / "sioux_scoring_runtime.json").read_text(encoding="utf-8")
    )

    benchmark_signs: dict[str, dict[str, object]] = {}
    for form, group in sioux_dpc.groupby("form", sort=True):
        values = group["direct_empirical_importance"]
        benchmark_signs[form] = {
            "candidate_count": int(len(group)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "positive_count": int((values > 0.0).sum()),
            "negative_count": int((values < 0.0).sum()),
            "zero_count": int((values == 0.0).sum()),
        }

    sensitivity_by_form: dict[str, dict[str, object]] = {}
    for form, group in sensitivity_metrics.groupby("form", sort=True):
        defined = group["rank_spearman"].dropna()
        sensitivity_by_form[form] = {
            "configuration_count": int(len(group)),
            "defined_rank_count": int(len(defined)),
            "undefined_rank_count": int(group["rank_spearman"].isna().sum()),
            "rank_spearman_minimum": float(defined.min()),
            "rank_spearman_median": float(defined.median()),
            "rank_spearman_maximum": float(defined.max()),
            "sign_agreement_minimum": float(group["sign_agreement"].min()),
            "sign_agreement_median": float(group["sign_agreement"].median()),
            "top5_overlap_minimum": float(group["top5_overlap"].min()),
            "top5_overlap_median": float(group["top5_overlap"].median()),
        }

    configurations = sensitivity[
        ["horizon", "stress", "gamma", "baseline_reliability"]
    ].drop_duplicates()
    degenerate = configurations[
        (configurations["baseline_reliability"] <= 0.05)
        | (configurations["baseline_reliability"] >= 0.95)
    ]

    reference = sensitivity[
        np.isclose(sensitivity["horizon"], 1.8)
        & np.isclose(sensitivity["stress"], 0.35)
        & np.isclose(sensitivity["gamma"], 0.9625)
    ][["candidate_id", "form", "direct_empirical_importance"]]
    joined = reference.merge(decomposition, on=["candidate_id", "form"])
    negative = joined[joined["direct_empirical_importance"] < 0.0].sort_values(
        "direct_empirical_importance"
    )
    positive = joined[joined["direct_empirical_importance"] > 0.0].sort_values(
        "direct_empirical_importance", ascending=False
    )
    contribution_columns = [
        "candidate_id",
        "form",
        "size",
        "direct_empirical_importance",
        "mean_shift_contribution",
        "marginal_variance_contribution",
        "off_diagonal_covariance_contribution",
        "total_expected_quadratic_loss_change",
    ]

    fair_by_form: dict[str, dict[str, object]] = {}
    for form, group in fair.groupby("form", sort=True):
        best = group.sort_values(
            "direct_empirical_importance", ascending=False
        ).iloc[0]
        fair_by_form[form] = record(
            best,
            [
                "candidate_id",
                "size",
                "stress",
                "direct_empirical_importance",
                "total_expected_added_disturbance",
            ],
        )

    analytical = certificates[
        certificates["allowance_type"] == "analytical_global"
    ]
    diagnostic = certificates[
        certificates["allowance_type"] == "finite_support_diagnostic"
    ]

    summary = {
        "sioux_falls": {
            "network": metadata["network"],
            "od_total": 360600.0,
            "settings": metadata["settings"],
            "coupling_example_first_edge": metadata[
                "coupling_example_first_edge"
            ],
            "importance_signs": benchmark_signs,
            "rank_comparisons": rank_comparisons.to_dict(orient="records"),
            "od_weighted_edge_importance_range": [
                float(
                    sioux_dpc.loc[
                        sioux_dpc["form"] == "edge",
                        "direct_empirical_importance",
                    ].min()
                ),
                float(
                    sioux_dpc.loc[
                        sioux_dpc["form"] == "edge",
                        "direct_empirical_importance",
                    ].max()
                ),
            ],
            "equal_source_edge_importance_range": [
                float(sioux_uniform["direct_empirical_importance"].min()),
                float(sioux_uniform["direct_empirical_importance"].max()),
            ],
            "quadratic_mean_absolute_importance_error": float(
                quadratic["importance_error"].abs().mean()
            ),
            "quadratic_maximum_absolute_importance_error": float(
                quadratic["importance_error"].abs().max()
            ),
            "quadratic_exact_match_count": int(
                np.isclose(quadratic["importance_error"], 0.0).sum()
            ),
            "quadrature_maximum_absolute_difference": float(
                metadata["quadrature_max_absolute_difference"]
            ),
        },
        "full_parameter_sensitivity": {
            "candidate_count": int(sensitivity["candidate_id"].nunique()),
            "configuration_count": int(len(configurations)),
            "result_count": int(len(sensitivity)),
            "by_form": sensitivity_by_form,
            "threshold_degenerate_configuration_count": int(len(degenerate)),
            "threshold_degenerate_configurations": degenerate.to_dict(
                orient="records"
            ),
        },
        "dependence": {
            "summary": dependence_summary.to_dict(orient="records"),
            "minimum_rank_spearman": float(
                dependence_stability["rank_spearman"].min()
            ),
            "minimum_top5_overlap": float(
                dependence_stability["top5_overlap"].min()
            ),
            "minimum_sign_agreement": float(
                dependence_stability["sign_agreement"].min()
            ),
            "marginal_mean_spread": float(
                dependence_summary["marginal_mean"].max()
                - dependence_summary["marginal_mean"].min()
            ),
            "marginal_standard_deviation_spread": float(
                dependence_summary["marginal_standard_deviation"].max()
                - dependence_summary["marginal_standard_deviation"].min()
            ),
        },
        "stress_intervention": {
            "decomposition_maximum_absolute_closure_error": float(
                decomposition["closure_error"].abs().max()
            ),
            "reference_negative_importance_count": int(len(negative)),
            "representative_negative_candidate": (
                record(negative.iloc[0], contribution_columns)
                if len(negative)
                else None
            ),
            "representative_positive_candidate": (
                record(positive.iloc[0], contribution_columns)
                if len(positive)
                else None
            ),
            "equal_total_expected_disturbance": float(
                fair["total_expected_added_disturbance"].iloc[0]
            ),
            "equal_total_budget_spread": float(
                fair["total_expected_added_disturbance"].max()
                - fair["total_expected_added_disturbance"].min()
            ),
            "equal_total_stress_range": [
                float(fair["stress"].min()),
                float(fair["stress"].max()),
            ],
            "best_equal_total_candidate_by_form": fair_by_form,
        },
        "certificates": {
            "condition_count_per_allowance": int(len(analytical)),
            "gamma_range": [
                float(certificates["gamma"].min()),
                float(certificates["gamma"].max()),
            ],
            "analytical_nonzero_condition_count": int(
                (
                    (analytical["sign_certificate_rate"] > 0.0)
                    | (analytical["ranking_certificate_rate"] > 0.0)
                ).sum()
            ),
            "analytical_maximum_sign_certificate_rate": float(
                analytical["sign_certificate_rate"].max()
            ),
            "analytical_maximum_ranking_certificate_rate": float(
                analytical["ranking_certificate_rate"].max()
            ),
            "diagnostic_maximum_sign_resolution_rate": float(
                diagnostic["sign_certificate_rate"].max()
            ),
            "diagnostic_maximum_ranking_resolution_rate": float(
                diagnostic["ranking_certificate_rate"].max()
            ),
            "minimum_analytical_coverage": float(
                analytical["interval_coverage"].min()
            ),
            "false_analytical_sign_count": int(
                analytical["false_sign_certificate_count"].sum()
            ),
            "false_analytical_ranking_count": int(
                analytical["false_ranking_certificate_count"].sum()
            ),
        },
        "computation": {
            "coefficient_construction_seconds": float(
                metadata["coefficient_wall_seconds"]
            ),
            "direct_candidate_seconds": float(metadata["direct_candidate_seconds"]),
            "conventional_candidate_seconds": float(
                metadata["efficiency_candidate_seconds"]
            ),
            "quadratic_candidate_scoring_seconds": float(
                scoring["quadratic_candidate_scoring_seconds"]
            ),
            "matrix_free_maximum_relative_error": float(
                matrix_free["relative_directional_error"].max()
            ),
            "matrix_free_query_seconds_at_20": float(
                matrix_free.loc[
                    matrix_free["query_count"] == 20, "matrix_free_seconds"
                ].iloc[0]
            ),
            "stored_matrix_query_seconds_at_20": float(
                matrix_free.loc[
                    matrix_free["query_count"] == 20,
                    "stored_q_query_seconds",
                ].iloc[0]
            ),
            "storage_projection": storage.to_dict(orient="records"),
        },
    }
    (RESULTS / "r1_verified_result_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
