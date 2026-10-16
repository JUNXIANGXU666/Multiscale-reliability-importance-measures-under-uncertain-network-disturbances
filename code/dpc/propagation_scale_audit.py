"""Propagation scale audit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

from dpc_core import (
    dpc,
    dpc_batch,
    gauss_rule,
    grid_candidates,
    grid_network,
    motif_networks,
    normalise_source_weights,
    propagators_from_eigendecomposition,
    synthetic_topology_network,
)
from sioux_and_sensitivity import (
    RESULTS,
    SIOUX_ALPHA_MAX,
    SIOUX_GAMMA,
    SIOUX_HORIZON,
    SIOUX_STRESS,
    direct_importance_table,
    parse_sioux_falls,
    sioux_candidates,
)
from run_all_experiments import (
    GRID_ALPHA_BAR,
    GRID_COLUMNS,
    GRID_GAMMA,
    GRID_ROWS,
    GRID_STRESS,
    HORIZON,
    SEED as BASE_EXPERIMENT_SEED,
    TOPOLOGY_ALPHA_BAR,
    TOPOLOGY_FAMILIES,
    TOPOLOGY_REPLICATES,
    TOPOLOGY_SIZES,
)


HERE = Path(__file__).resolve().parent
LOGS = HERE / "sioux_outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOGS / "propagation_scale_audit.log", mode="a", encoding="utf-8"
            ),
        ],
        force=True,
    )


def source_time_averaged_participation(
    hamiltonian: np.ndarray,
    horizon: float,
    order: int,
) -> np.ndarray:
    """Return the horizon-averaged participation index for every source."""

    times, weights = gauss_rule(order, 0.0, float(horizon))
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    propagators = propagators_from_eigendecomposition(
        eigenvalues, eigenvectors, times
    )
    occupation = np.abs(propagators) ** 2
    concentration = np.sum(occupation * occupation, axis=1)
    participation = 1.0 / concentration
    return np.sum(weights[:, None] * participation, axis=0) / float(horizon)


def traversal_diagnostic(network) -> dict[str, float]:
    """Compute a transparent hop-based propagation-time diagnostic."""

    graph = network.as_networkx()
    hop_diameter = int(nx.diameter(graph))
    median_coupling = float(np.median(network.weights))
    strongest_coupling = float(np.max(network.weights))
    median_edge_transfer_time = float(np.pi / (2.0 * median_coupling))
    strongest_edge_transfer_time = float(np.pi / (2.0 * strongest_coupling))
    traversal_time = float(hop_diameter * median_edge_transfer_time)
    return {
        "hop_diameter": hop_diameter,
        "median_coupling": median_coupling,
        "strongest_coupling": strongest_coupling,
        "median_edge_transfer_time": median_edge_transfer_time,
        "strongest_edge_transfer_time": strongest_edge_transfer_time,
        "diameter_times_median_edge_transfer": traversal_time,
    }


def baseline_row(
    network,
    network_group: str,
    horizon: float,
    source_weights: np.ndarray | None,
    alpha_max: float,
    gamma: float | None,
    order: int = 128,
) -> dict[str, object]:
    source_average = source_time_averaged_participation(
        network.h0, horizon, order
    )
    declared_weights = normalise_source_weights(source_weights, network.n)
    dpc_declared = float(declared_weights @ source_average)
    dpc_uniform = float(np.mean(source_average))
    scale = traversal_diagnostic(network)
    n = network.n
    constant = float(
        16.0 * n**2 * (1.0 + 6.0 * n + 6.0 * n**2) * horizon**3
    )
    kappa_triangle = float(alpha_max * np.sum(network.weights))
    kappa_operator = float(
        np.linalg.norm(alpha_max * network.h0, ord=2)
    )
    epsilon_triangle = float(constant * kappa_triangle**3 / 6.0)
    epsilon_operator = float(constant * kappa_operator**3 / 6.0)
    threshold_margin = (
        float((1.0 - gamma) * dpc_declared) if gamma is not None else np.nan
    )
    return {
        "network": network.name,
        "network_group": network_group,
        "nodes": network.n,
        "edges": network.m,
        "horizon": float(horizon),
        "dpc_uniform_sources": dpc_uniform,
        "dpc_declared_sources": dpc_declared,
        "source_participation_min": float(np.min(source_average)),
        "source_participation_median": float(np.median(source_average)),
        "source_participation_max": float(np.max(source_average)),
        "source_participation_q25": float(np.quantile(source_average, 0.25)),
        "source_participation_q75": float(np.quantile(source_average, 0.75)),
        **scale,
        "horizon_to_traversal_ratio": float(
            horizon / scale["diameter_times_median_edge_transfer"]
        ),
        "alpha_max": float(alpha_max),
        "c_n_l": constant,
        "kappa_triangle": kappa_triangle,
        "kappa_operator": kappa_operator,
        "epsilon_triangle": epsilon_triangle,
        "epsilon_operator": epsilon_operator,
        "threshold_margin": threshold_margin,
        "epsilon_triangle_to_margin": (
            float(epsilon_triangle / threshold_margin)
            if threshold_margin > 0.0
            else np.nan
        ),
        "epsilon_operator_to_margin": (
            float(epsilon_operator / threshold_margin)
            if threshold_margin > 0.0
            else np.nan
        ),
        "quadrature_order": int(order),
    }


def run_baseline_audit() -> None:
    rows: list[dict[str, object]] = []
    for network in motif_networks().values():
        rows.append(
            baseline_row(
                network,
                "validation motif",
                HORIZON,
                None,
                0.25,
                None,
            )
        )

    grid = grid_network(GRID_ROWS, GRID_COLUMNS, BASE_EXPERIMENT_SEED)
    rows.append(
        baseline_row(
            grid,
            "spatial grid",
            HORIZON,
            None,
            GRID_ALPHA_BAR,
            GRID_GAMMA,
        )
    )

    for family_index, family in enumerate(TOPOLOGY_FAMILIES):
        for node_count in TOPOLOGY_SIZES:
            for replicate in range(TOPOLOGY_REPLICATES):
                network_seed = (
                    BASE_EXPERIMENT_SEED
                    + 100000
                    + family_index * 10000
                    + node_count * 100
                    + replicate
                )
                network = synthetic_topology_network(
                    family,
                    node_count,
                    network_seed,
                    name=f"{family}_n{node_count}_r{replicate + 1}",
                )
                rows.append(
                    baseline_row(
                        network,
                        "synthetic topology",
                        HORIZON,
                        None,
                        TOPOLOGY_ALPHA_BAR,
                        0.97,
                        order=96,
                    )
                )

    sioux, _, source_weights, _ = parse_sioux_falls()
    rows.append(
        baseline_row(
            sioux,
            "recognised benchmark",
            SIOUX_HORIZON,
            source_weights,
            SIOUX_ALPHA_MAX,
            SIOUX_GAMMA,
        )
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "baseline_propagation_scale_audit.csv", index=False)

    summary = (
        frame.groupby("network_group", dropna=False)
        .agg(
            networks=("network", "count"),
            dpc_min=("dpc_declared_sources", "min"),
            dpc_median=("dpc_declared_sources", "median"),
            dpc_max=("dpc_declared_sources", "max"),
            horizon_ratio_min=("horizon_to_traversal_ratio", "min"),
            horizon_ratio_median=("horizon_to_traversal_ratio", "median"),
            horizon_ratio_max=("horizon_to_traversal_ratio", "max"),
        )
        .reset_index()
    )
    summary.to_csv(
        RESULTS / "baseline_propagation_scale_summary.csv", index=False
    )
    logging.info("Baseline propagation-scale audit wrote %d network rows", len(frame))


def deterministic_top_overlap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    count: int = 5,
) -> float:
    ref_ids = set(
        reference.sort_values(
            ["direct_empirical_importance", "candidate_id"],
            ascending=[False, True],
        )["candidate_id"].head(count)
    )
    comp_ids = set(
        comparison.sort_values(
            ["direct_empirical_importance", "candidate_id"],
            ascending=[False, True],
        )["candidate_id"].head(count)
    )
    return float(len(ref_ids & comp_ids) / max(min(count, len(ref_ids)), 1))


def horizon_comparison_rows(
    case: str,
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    reference_horizon: float,
    horizon: float,
    traversal_time: float,
    quadrature_difference: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for form in sorted(reference["form"].unique()):
        ref = reference[reference["form"] == form].sort_values("candidate_id")
        comp = comparison[comparison["form"] == form].sort_values("candidate_id")
        merged = ref[["candidate_id", "direct_empirical_importance"]].merge(
            comp[["candidate_id", "direct_empirical_importance"]],
            on="candidate_id",
            suffixes=("_reference", "_comparison"),
            validate="one_to_one",
        )
        x = merged["direct_empirical_importance_reference"].to_numpy(float)
        y = merged["direct_empirical_importance_comparison"].to_numpy(float)
        correlation = (
            float(spearmanr(rankdata(x), rankdata(y)).statistic)
            if np.ptp(x) > 0.0 and np.ptp(y) > 0.0
            else np.nan
        )
        sign_agreement = float(np.mean(np.sign(x) == np.sign(y)))
        best = comp.sort_values(
            ["direct_empirical_importance", "candidate_id"],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            {
                "case": case,
                "form": form,
                "reference_horizon": float(reference_horizon),
                "comparison_horizon": float(horizon),
                "comparison_to_traversal_ratio": float(horizon / traversal_time),
                "spearman_vs_reference": correlation,
                "top5_overlap_vs_reference": deterministic_top_overlap(ref, comp),
                "sign_agreement_vs_reference": sign_agreement,
                "selected_candidate": str(best["candidate_id"]),
                "selected_importance": float(best["direct_empirical_importance"]),
                "baseline_reliability": float(best["baseline_reliability"]),
                "baseline_mean_performance": float(best["baseline_mean_performance"]),
                "threshold": float(best["threshold"]),
                "quadrature_validation_max_difference": float(
                    quadrature_difference
                ),
            }
        )
    return rows


def run_case_horizon_sensitivity(
    case: str,
    network,
    candidates,
    scenarios: np.ndarray,
    alpha_bar: np.ndarray,
    source_weights: np.ndarray | None,
    gamma: float,
    stress: float,
    checkpoint_tag: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    scale = traversal_diagnostic(network)
    traversal_time = scale["diameter_times_median_edge_transfer"]
    horizons = [float(HORIZON), float(0.5 * traversal_time), float(traversal_time)]
    horizons = list(dict.fromkeys(round(value, 10) for value in horizons))
    frames: dict[float, pd.DataFrame] = {}
    metadata: dict[str, object] = {
        "case": case,
        "candidate_count": len(candidates),
        "scenario_count": int(len(scenarios)),
        "traversal_diagnostic": scale,
        "horizons": horizons,
        "primary_quadrature_order": 96,
        "validation_quadrature_order": 128,
    }
    validation: dict[float, float] = {}
    started = perf_counter()

    for horizon in horizons:
        stem_value = f"{horizon:.6f}".replace(".", "p")
        logging.info(
            "Starting %s horizon %.6f with %d candidates",
            case,
            horizon,
            len(candidates),
        )
        with threadpool_limits(limits=1, user_api="blas"):
            frame, _ = direct_importance_table(
                network,
                candidates,
                scenarios,
                alpha_bar,
                horizon,
                gamma,
                stress,
                n_time=96,
                source_weights=source_weights,
                checkpoint_stem=f"propagation_scale_{case}_{checkpoint_tag}{stem_value}",
                metric="dpc",
            )
        frame = frame.assign(
            case=case,
            horizon=horizon,
            horizon_to_traversal_ratio=horizon / traversal_time,
        )
        frames[horizon] = frame

        validation_count = min(16, len(scenarios))
        order96 = dpc_batch(
            network,
            scenarios[:validation_count],
            horizon,
            n_time=96,
            source_weights=source_weights,
        )
        order128 = dpc_batch(
            network,
            scenarios[:validation_count],
            horizon,
            n_time=128,
            source_weights=source_weights,
        )
        validation[horizon] = float(np.max(np.abs(order96 - order128)))
        logging.info(
            "Completed %s horizon %.6f; quadrature difference %.3e",
            case,
            horizon,
            validation[horizon],
        )

    reference_horizon = min(horizons, key=lambda value: abs(value - HORIZON))
    reference = frames[reference_horizon]
    comparison_rows: list[dict[str, object]] = []
    for horizon in horizons:
        comparison_rows.extend(
            horizon_comparison_rows(
                case,
                reference,
                frames[horizon],
                reference_horizon,
                horizon,
                traversal_time,
                validation[horizon],
            )
        )
    metadata["runtime_seconds"] = float(perf_counter() - started)
    metadata["quadrature_validation_max_difference"] = {
        str(key): value for key, value in validation.items()
    }
    return (
        pd.concat(frames.values(), ignore_index=True),
        pd.DataFrame(comparison_rows),
        metadata,
    )


def run_horizon_sensitivity() -> None:
    grid = grid_network(GRID_ROWS, GRID_COLUMNS, BASE_EXPERIMENT_SEED)
    grid_candidates_all = grid_candidates(grid, GRID_ROWS, GRID_COLUMNS)
    grid_archive = np.load(HERE / "results" / "grid_experiment_arrays.npz")
    grid_results, grid_summary, grid_metadata = run_case_horizon_sensitivity(
        "grid",
        grid,
        grid_candidates_all,
        grid_archive["scenarios"],
        grid_archive["alpha_bar"],
        None,
        GRID_GAMMA,
        GRID_STRESS,
        checkpoint_tag=f"seed{BASE_EXPERIMENT_SEED}_",
    )

    sioux, _, source_weights, _ = parse_sioux_falls()
    sioux_archive = np.load(RESULTS / "sioux_scenarios.npz")
    sioux_results, sioux_summary, sioux_metadata = run_case_horizon_sensitivity(
        "sioux_falls",
        sioux,
        sioux_candidates(sioux),
        sioux_archive["scenarios"],
        sioux_archive["alpha_bar"],
        source_weights,
        SIOUX_GAMMA,
        SIOUX_STRESS,
    )

    pd.concat([grid_results, sioux_results], ignore_index=True).to_csv(
        RESULTS / "horizon_scale_candidate_importance.csv", index=False
    )
    pd.concat([grid_summary, sioux_summary], ignore_index=True).to_csv(
        RESULTS / "horizon_scale_sensitivity.csv", index=False
    )
    (RESULTS / "horizon_scale_audit.json").write_text(
        json.dumps(
            {"grid": grid_metadata, "sioux_falls": sioux_metadata},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_grid_horizon_sensitivity_only() -> None:
    grid = grid_network(GRID_ROWS, GRID_COLUMNS, BASE_EXPERIMENT_SEED)
    grid_archive = np.load(HERE / "results" / "grid_experiment_arrays.npz")
    grid_results, grid_summary, grid_metadata = run_case_horizon_sensitivity(
        "grid",
        grid,
        grid_candidates(grid, GRID_ROWS, GRID_COLUMNS),
        grid_archive["scenarios"],
        grid_archive["alpha_bar"],
        None,
        GRID_GAMMA,
        GRID_STRESS,
        checkpoint_tag=f"seed{BASE_EXPERIMENT_SEED}_",
    )

    candidate_path = RESULTS / "horizon_scale_candidate_importance.csv"
    summary_path = RESULTS / "horizon_scale_sensitivity.csv"
    audit_path = RESULTS / "horizon_scale_audit.json"

    if candidate_path.exists():
        existing_candidates = pd.read_csv(candidate_path)
        existing_candidates = existing_candidates[existing_candidates["case"] != "grid"]
        grid_results = pd.concat([grid_results, existing_candidates], ignore_index=True)
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        existing_summary = existing_summary[existing_summary["case"] != "grid"]
        grid_summary = pd.concat([grid_summary, existing_summary], ignore_index=True)

    grid_results.to_csv(candidate_path, index=False)
    grid_summary.to_csv(summary_path, index=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audit["grid"] = grid_metadata
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("baseline", "sensitivity", "grid-sensitivity", "all"),
        default="all",
    )
    args = parser.parse_args()
    configure_logging()
    if args.stage in {"baseline", "all"}:
        run_baseline_audit()
    if args.stage in {"sensitivity", "all"}:
        run_horizon_sensitivity()
    if args.stage == "grid-sensitivity":
        run_grid_horizon_sensitivity_only()


if __name__ == "__main__":
    main()
