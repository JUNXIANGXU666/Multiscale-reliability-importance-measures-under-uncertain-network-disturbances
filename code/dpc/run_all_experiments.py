"""Numerical experiments for motifs, grids, synthetic networks and scaling."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import scipy
from scipy import sparse
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from dpc_core import (
    DEFAULT_SEED,
    Candidate,
    classical_edge_measures,
    dpc,
    dpc_batch,
    directional_loss_derivatives,
    exact_binary_scenarios,
    exact_reliability,
    finite_difference_coefficients,
    gaussian_copula_beta_scenarios,
    grid_candidates,
    grid_network,
    incremental_quadratic_loss,
    moment_interval,
    motif_candidates,
    motif_networks,
    quadratic_loss,
    random_connected_network,
    rank_aware_coefficients,
    rank_metrics,
    save_npz,
    sparse_scoring_kernel,
    stress_scenarios,
    synthetic_topology_network,
    theoretical_epsilon,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

SEED = DEFAULT_SEED
HORIZON = 1.8
EXACT_TIME_ORDER = 48
GRID_ROWS = 5
GRID_COLUMNS = 5
GRID_ALPHA_BAR = 0.22
GRID_RHO = 0.35
GRID_SCENARIOS = 256
GRID_STRESS = 0.35
GRID_GAMMA = 0.9625
TOPOLOGY_FAMILIES = ("erdos_renyi", "small_world", "scale_free")
TOPOLOGY_SIZES = (10, 14, 18)
TOPOLOGY_REPLICATES = 3
TOPOLOGY_SCENARIOS = 128
TOPOLOGY_ALPHA_BAR = 0.18
TOPOLOGY_RHO = 0.35
TOPOLOGY_STRESS = 0.35
TOPOLOGY_GAMMA = 0.97
TOPOLOGY_COEFFICIENT_TIME = 14
TOPOLOGY_COEFFICIENT_INNER = 5
TOPOLOGY_EXACT_TIME_ORDER = 32


def configure_logging() -> None:
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS / "full_experiment_run.log", mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def topology_metadata_payload() -> dict[str, object]:
    return {
        "random_seed": SEED,
        "network_families": list(TOPOLOGY_FAMILIES),
        "network_sizes": list(TOPOLOGY_SIZES),
        "replicates_per_family_and_size": TOPOLOGY_REPLICATES,
        "network_count": (
            len(TOPOLOGY_FAMILIES)
            * len(TOPOLOGY_SIZES)
            * TOPOLOGY_REPLICATES
        ),
        "scenario_count_per_network": TOPOLOGY_SCENARIOS,
        "alpha_bar": TOPOLOGY_ALPHA_BAR,
        "rho": TOPOLOGY_RHO,
        "stress": TOPOLOGY_STRESS,
        "gamma": TOPOLOGY_GAMMA,
        "threshold_rule": "tau = gamma times baseline DPC",
        "coefficient_time_order": TOPOLOGY_COEFFICIENT_TIME,
        "coefficient_inner_order": TOPOLOGY_COEFFICIENT_INNER,
        "exact_candidate_time_order": TOPOLOGY_EXACT_TIME_ORDER,
        "reference_validation_time_order": EXACT_TIME_ORDER,
        "validation_scenarios_per_network": 8,
        "top_k": 5,
        "ranking_saturation_rule": (
            "Rank metrics are undefined when all exact edge importance "
            "values are equal."
        ),
    }


def status_from_interval(lower: float, upper: float) -> str:
    if lower > 0.0:
        return "reliability-reducing"
    if upper < 0.0:
        return "reliability-increasing"
    return "unresolved"


def safe_spearman(reference: np.ndarray, score: np.ndarray) -> float:
    reference_values = np.asarray(reference, dtype=float)
    score_values = np.asarray(score, dtype=float)
    if (
        len(reference_values) < 2
        or np.unique(reference_values).size < 2
        or np.unique(score_values).size < 2
    ):
        return float("nan")
    return float(spearmanr(reference_values, score_values).statistic)


def stable_top_ids(
    candidate_ids: np.ndarray,
    values: np.ndarray,
    count: int,
) -> set[str]:
    ids = np.asarray(candidate_ids, dtype=str)
    scores = np.asarray(values, dtype=float)
    order = np.lexsort((ids, -scores))
    return set(ids[order[: min(int(count), len(ids))]].tolist())


def signed_agreement(
    reference: np.ndarray,
    estimate: np.ndarray,
    tolerance: float,
) -> float:
    reference_values = np.asarray(reference, dtype=float)
    estimate_values = np.asarray(estimate, dtype=float)
    mask = np.abs(reference_values) > float(tolerance)
    if not np.any(mask):
        return float("nan")
    return float(
        np.mean(
            np.sign(reference_values[mask])
            == np.sign(estimate_values[mask])
        )
    )


def environment_metadata() -> dict:
    return {
        "seed": SEED,
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 3),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "networkx": nx.__version__,
        "matplotlib": matplotlib.__version__,
        "blas_threads_omp": os.environ.get("OMP_NUM_THREADS", "not fixed"),
        "blas_threads_mkl": os.environ.get("MKL_NUM_THREADS", "not fixed"),
        "horizon": HORIZON,
        "exact_time_quadrature_order": EXACT_TIME_ORDER,
    }


def run_coefficient_and_motif_validation() -> None:
    logging.info("Starting coefficient convergence and motif validation")
    networks = motif_networks()
    diamond = networks["diamond4"]
    reference = rank_aware_coefficients(diamond, HORIZON, n_time=32, n_inner=8)
    d0_reference, g_reference, q_reference, reference_timing = reference
    save_npz(
        RESULTS / "diamond_reference_coefficients.npz",
        d0=np.array([d0_reference]),
        g=g_reference,
        q=q_reference,
        edges=diamond.edges,
        weights=diamond.weights,
    )

    convergence_rows: list[dict] = []
    for n_time, n_inner in [(6, 2), (8, 3), (12, 4), (16, 5), (20, 6), (24, 7), (32, 8)]:
        d0, g, q, timing = rank_aware_coefficients(
            diamond, HORIZON, n_time=n_time, n_inner=n_inner
        )
        convergence_rows.append(
            {
                "method": "rank-aware quadrature",
                "n_time": n_time,
                "n_inner": n_inner,
                "n_quad": n_time * n_inner * n_inner,
                "d0_absolute_error": abs(d0 - d0_reference),
                "g_relative_error": np.linalg.norm(g - g_reference)
                / max(np.linalg.norm(g_reference), 1.0e-15),
                "q_relative_error": np.linalg.norm(q - q_reference)
                / max(np.linalg.norm(q_reference), 1.0e-15),
                "seconds": timing["coefficient_seconds"],
            }
        )
    pd.DataFrame(convergence_rows).to_csv(
        RESULTS / "coefficient_quadrature_convergence.csv", index=False
    )

    difference_rows: list[dict] = []
    for step in [5.0e-2, 2.0e-2, 1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3, 5.0e-4]:
        d0, g, q, timing = finite_difference_coefficients(
            diamond,
            HORIZON,
            n_time=64,
            step=step,
            fourth_order_edges=True,
            richardson_mixed=False,
        )
        difference_rows.append(
            {
                "step": step,
                "d0_absolute_error": abs(d0 - d0_reference),
                "g_relative_error": np.linalg.norm(g - g_reference)
                / max(np.linalg.norm(g_reference), 1.0e-15),
                "q_relative_error": np.linalg.norm(q - q_reference)
                / max(np.linalg.norm(q_reference), 1.0e-15),
                "seconds": timing["coefficient_seconds"],
                "dpc_evaluations": timing["dpc_evaluations"],
            }
        )
    pd.DataFrame(difference_rows).to_csv(
        RESULTS / "coefficient_finite_difference_validation.csv", index=False
    )

    motif_summary_rows: list[dict] = []
    motif_candidate_rows: list[dict] = []
    delta = 0.22
    for network_name, network in networks.items():
        d0, g, q, timing = rank_aware_coefficients(
            network, HORIZON, n_time=32, n_inner=8
        )
        motif_summary_rows.append(
            {
                "network": network_name,
                "n": network.n,
                "m": network.m,
                "d0": d0,
                "negative_g_count": int(np.sum(g < 0.0)),
                "positive_g_count": int(np.sum(g > 0.0)),
                "q_min": float(np.min(q)),
                "q_max": float(np.max(q)),
                "coefficient_seconds": timing["coefficient_seconds"],
            }
        )
        for candidate in motif_candidates(network):
            selected = np.asarray(candidate.edge_ids, dtype=int)
            alpha = np.zeros(network.m)
            alpha[selected] = delta
            exact_loss = d0 - dpc(network, alpha, HORIZON, n_time=64)
            direct = delta * float(np.sum(g[selected]))
            additive_second = direct + 0.5 * delta**2 * float(
                np.sum(np.diag(q)[selected])
            )
            quadratic = float(quadratic_loss(alpha, g, q))
            motif_candidate_rows.append(
                {
                    "network": network_name,
                    "candidate_id": candidate.candidate_id,
                    "scale": candidate.scale,
                    "size": candidate.size,
                    "edge_ids": " ".join(str(value) for value in candidate.edge_ids),
                    "exact_loss": exact_loss,
                    "first_order_loss": direct,
                    "additive_second_order_loss": additive_second,
                    "quadratic_loss": quadratic,
                    "pair_interaction": quadratic - additive_second,
                    "quadratic_residual": exact_loss - quadratic,
                }
            )
        save_npz(
            RESULTS / f"{network_name}_coefficients.npz",
            d0=np.array([d0]),
            g=g,
            q=q,
            edges=network.edges,
            weights=network.weights,
            positions=np.asarray([network.positions[node] for node in range(network.n)]),
        )
    pd.DataFrame(motif_summary_rows).to_csv(RESULTS / "motif_summary.csv", index=False)
    pd.DataFrame(motif_candidate_rows).to_csv(
        RESULTS / "motif_candidate_losses.csv", index=False
    )

    certificate_cases = [
        {
            "case_id": "reducing_cycle5",
            "network": networks["cycle5"],
            "target_candidate": "C08",
            "direction": "reducing",
            "amplitude": 0.015,
            "alpha_bar": 0.25,
            "stress": 0.80,
        },
        {
            "case_id": "increasing_diamond4",
            "network": networks["diamond4"],
            "target_candidate": "C03",
            "direction": "increasing",
            "amplitude": 0.015,
            "alpha_bar": 0.25,
            "stress": 0.80,
        },
    ]
    binary_rows: list[dict] = []
    case_metadata: list[dict] = []
    for case in certificate_cases:
        network = case["network"]
        coefficient_archive = np.load(RESULTS / f"{network.name}_coefficients.npz")
        d0_case = float(coefficient_archive["d0"][0])
        g_case = coefficient_archive["g"]
        q_case = coefficient_archive["q"]
        scenarios, probabilities = exact_binary_scenarios(
            network.m, amplitude=case["amplitude"]
        )
        alpha_bar = np.full(network.m, case["alpha_bar"])
        candidates = motif_candidates(network)
        exact_losses = d0_case - dpc_batch(network, scenarios, HORIZON, n_time=64)
        approximate_losses = np.asarray(quadratic_loss(scenarios, g_case, q_case))
        candidate_cache: list[dict] = []
        support_epsilon = float(np.max(np.abs(exact_losses - approximate_losses)))
        for candidate in candidates:
            stressed = stress_scenarios(
                scenarios, candidate.edge_ids, case["stress"], alpha_bar
            )
            stressed_exact_losses = d0_case - dpc_batch(
                network, stressed, HORIZON, n_time=64
            )
            stressed_approximate_losses = np.asarray(
                quadratic_loss(stressed, g_case, q_case)
            )
            support_epsilon = max(
                support_epsilon,
                float(
                    np.max(
                        np.abs(stressed_exact_losses - stressed_approximate_losses)
                    )
                ),
            )
            candidate_cache.append(
                {
                    "candidate": candidate,
                    "exact_losses": stressed_exact_losses,
                    "approximate_losses": stressed_approximate_losses,
                }
            )
        target = next(
            cached
            for cached in candidate_cache
            if cached["candidate"].candidate_id == case["target_candidate"]
        )
        if case["direction"] == "reducing":
            gap = float(np.min(target["exact_losses"]) - np.max(exact_losses))
            theta = 0.5 * (
                float(np.min(target["exact_losses"])) + float(np.max(exact_losses))
            )
        else:
            gap = float(np.min(exact_losses) - np.max(target["exact_losses"]))
            theta = 0.5 * (
                float(np.min(exact_losses)) + float(np.max(target["exact_losses"]))
            )
        if gap <= 0.0:
            raise RuntimeError(
                f"Finite-support validation case {case['case_id']} has no separation gap"
            )
        tau = d0_case - theta
        global_epsilon = theoretical_epsilon(network, HORIZON, alpha_bar)
        baseline_exact_reliability = exact_reliability(exact_losses, theta)
        baseline_approximate_reliability = exact_reliability(
            approximate_losses, theta
        )
        baseline_support = moment_interval(
            approximate_losses, theta, support_epsilon
        )
        baseline_global = moment_interval(
            approximate_losses, theta, global_epsilon
        )
        for cached in candidate_cache:
            candidate = cached["candidate"]
            stressed_exact_losses = cached["exact_losses"]
            stressed_approximate_losses = cached["approximate_losses"]
            exact_importance = baseline_exact_reliability - exact_reliability(
                stressed_exact_losses, theta
            )
            approximate_importance = (
                baseline_approximate_reliability
                - exact_reliability(stressed_approximate_losses, theta)
            )
            stressed_support = moment_interval(
                stressed_approximate_losses, theta, support_epsilon
            )
            support_lower = baseline_support[0] - stressed_support[1]
            support_upper = baseline_support[1] - stressed_support[0]
            stressed_global = moment_interval(
                stressed_approximate_losses, theta, global_epsilon
            )
            binary_rows.append(
                {
                    "case_id": case["case_id"],
                    "network": network.name,
                    "candidate_id": candidate.candidate_id,
                    "is_target_candidate": candidate.candidate_id
                    == case["target_candidate"],
                    "scale": candidate.scale,
                    "size": candidate.size,
                    "exact_importance": exact_importance,
                    "quadratic_importance": approximate_importance,
                    "support_lower": support_lower,
                    "support_upper": support_upper,
                    "support_status": status_from_interval(
                        support_lower, support_upper
                    ),
                    "support_contains_exact": support_lower - 1.0e-12
                    <= exact_importance
                    <= support_upper + 1.0e-12,
                    "global_lower": baseline_global[0] - stressed_global[1],
                    "global_upper": baseline_global[1] - stressed_global[0],
                }
            )
        case_metadata.append(
            {
                "case_id": case["case_id"],
                "network": network.name,
                "target_candidate": case["target_candidate"],
                "direction": case["direction"],
                "scenario_count": int(len(scenarios)),
                "probability_checksum": float(np.sum(probabilities)),
                "binary_amplitude": case["amplitude"],
                "alpha_bar": case["alpha_bar"],
                "stress": case["stress"],
                "separation_gap": gap,
                "theta": theta,
                "tau": tau,
                "baseline_exact_reliability": baseline_exact_reliability,
                "support_epsilon": support_epsilon,
                "global_epsilon": global_epsilon,
            }
        )
    pd.DataFrame(binary_rows).to_csv(
        RESULTS / "small_exact_reliability_validation.csv", index=False
    )
    write_json(
        RESULTS / "small_exact_reliability_metadata.json",
        {"cases": case_metadata, "reference_timing": reference_timing},
    )
    logging.info("Completed coefficient convergence and motif validation")


def run_grid_experiment() -> None:
    logging.info("Starting 5 by 5 grid experiment")
    network = grid_network(GRID_ROWS, GRID_COLUMNS, SEED)
    final_configuration = (24, 6)
    configurations = [(8, 3), (12, 4), (16, 5), (20, 6), final_configuration]
    coefficient_results: dict[tuple[int, int], tuple] = {}
    for n_time, n_inner in configurations:
        logging.info("Grid coefficients with n_time=%d and n_inner=%d", n_time, n_inner)
        coefficient_results[(n_time, n_inner)] = rank_aware_coefficients(
            network, HORIZON, n_time=n_time, n_inner=n_inner
        )
    d0, g, q, final_timing = coefficient_results[final_configuration]
    convergence_rows = []
    for configuration, (d0_current, g_current, q_current, timing) in coefficient_results.items():
        convergence_rows.append(
            {
                "n_time": configuration[0],
                "n_inner": configuration[1],
                "n_quad": configuration[0] * configuration[1] ** 2,
                "d0_absolute_error": abs(d0_current - d0),
                "g_relative_error": np.linalg.norm(g_current - g)
                / max(np.linalg.norm(g), 1.0e-15),
                "q_relative_error": np.linalg.norm(q_current - q)
                / max(np.linalg.norm(q), 1.0e-15),
                "seconds": timing["coefficient_seconds"],
            }
        )
    pd.DataFrame(convergence_rows).to_csv(
        RESULTS / "grid_coefficient_convergence.csv", index=False
    )

    alpha_bar = np.full(network.m, GRID_ALPHA_BAR)
    scenarios = gaussian_copula_beta_scenarios(
        network.m,
        GRID_SCENARIOS,
        alpha_bar,
        GRID_RHO,
        SEED,
    )
    candidates = grid_candidates(network, GRID_ROWS, GRID_COLUMNS)
    tau = GRID_GAMMA * d0
    theta = d0 - tau
    logging.info(
        "Grid has n=%d, m=%d, candidates=%d, scenarios=%d",
        network.n,
        network.m,
        len(candidates),
        len(scenarios),
    )

    exact_baseline_dpc = dpc_batch(
        network, scenarios, HORIZON, n_time=EXACT_TIME_ORDER
    )
    exact_baseline_losses = d0 - exact_baseline_dpc
    approximate_baseline_losses = np.asarray(quadratic_loss(scenarios, g, q))
    qx = scenarios @ q.T
    baseline_exact_reliability = exact_reliability(exact_baseline_losses, theta)
    baseline_approximate_reliability = exact_reliability(
        approximate_baseline_losses, theta
    )

    exact_stressed_matrix = np.empty((len(candidates), len(scenarios)), dtype=float)
    approximate_stressed_matrix = np.empty_like(exact_stressed_matrix)
    support_epsilon = float(
        np.max(np.abs(exact_baseline_losses - approximate_baseline_losses))
    )
    intermediate_rows: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        stressed_scenario_values = stress_scenarios(
            scenarios, candidate.edge_ids, GRID_STRESS, alpha_bar
        )
        stressed_exact_losses = d0 - dpc_batch(
            network,
            stressed_scenario_values,
            HORIZON,
            n_time=EXACT_TIME_ORDER,
        )
        stressed_approximate_losses = incremental_quadratic_loss(
            scenarios,
            candidate.edge_ids,
            GRID_STRESS,
            alpha_bar,
            g,
            q,
            baseline_losses=approximate_baseline_losses,
            qx=qx,
        )
        exact_stressed_matrix[candidate_index] = stressed_exact_losses
        approximate_stressed_matrix[candidate_index] = stressed_approximate_losses
        support_epsilon = max(
            support_epsilon,
            float(np.max(np.abs(stressed_exact_losses - stressed_approximate_losses))),
        )
        exact_importance = baseline_exact_reliability - exact_reliability(
            stressed_exact_losses, theta
        )
        approximate_importance = baseline_approximate_reliability - exact_reliability(
            stressed_approximate_losses, theta
        )
        selected = np.asarray(candidate.edge_ids, dtype=int)
        mean_increment = np.mean(
            GRID_STRESS * (alpha_bar[selected][None, :] - scenarios[:, selected]),
            axis=0,
        )
        first_order_mean_change = float(np.dot(g[selected], mean_increment))
        full_mean_change = float(
            np.mean(stressed_approximate_losses - approximate_baseline_losses)
        )
        intermediate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "scale": candidate.scale,
                "size": candidate.size,
                "edge_ids": " ".join(str(value) for value in candidate.edge_ids),
                "exact_importance": exact_importance,
                "quadratic_importance": approximate_importance,
                "importance_error": approximate_importance - exact_importance,
                "mean_loss_change": full_mean_change,
                "first_order_mean_change": first_order_mean_change,
                "interaction_mean_change": full_mean_change - first_order_mean_change,
                "maximum_absolute_residual": float(
                    np.max(np.abs(stressed_exact_losses - stressed_approximate_losses))
                ),
            }
        )
        if (candidate_index + 1) % 10 == 0 or candidate_index + 1 == len(candidates):
            logging.info(
                "Exact grid validation completed for %d of %d candidates",
                candidate_index + 1,
                len(candidates),
            )

    global_epsilon = theoretical_epsilon(network, HORIZON, alpha_bar)
    baseline_support_interval = moment_interval(
        approximate_baseline_losses, theta, support_epsilon
    )
    baseline_global_interval = moment_interval(
        approximate_baseline_losses, theta, global_epsilon
    )
    result_rows = []
    for candidate_index, row in enumerate(intermediate_rows):
        stressed_support_interval = moment_interval(
            approximate_stressed_matrix[candidate_index], theta, support_epsilon
        )
        support_lower = baseline_support_interval[0] - stressed_support_interval[1]
        support_upper = baseline_support_interval[1] - stressed_support_interval[0]
        stressed_global_interval = moment_interval(
            approximate_stressed_matrix[candidate_index], theta, global_epsilon
        )
        global_lower = baseline_global_interval[0] - stressed_global_interval[1]
        global_upper = baseline_global_interval[1] - stressed_global_interval[0]
        row.update(
            {
                "support_lower": support_lower,
                "support_upper": support_upper,
                "support_status": status_from_interval(support_lower, support_upper),
                "support_contains_exact": support_lower - 1.0e-12
                <= row["exact_importance"]
                <= support_upper + 1.0e-12,
                "global_lower": global_lower,
                "global_upper": global_upper,
                "global_status": status_from_interval(global_lower, global_upper),
            }
        )
        result_rows.append(row)
    candidate_frame = pd.DataFrame(result_rows)
    candidate_frame["exact_rank"] = candidate_frame.groupby("scale")[
        "exact_importance"
    ].rank(ascending=False, method="min")
    candidate_frame["quadratic_rank"] = candidate_frame.groupby("scale")[
        "quadratic_importance"
    ].rank(ascending=False, method="min")
    candidate_frame["support_screening_rank"] = candidate_frame.groupby("scale")[
        "support_lower"
    ].rank(ascending=False, method="min")
    candidate_frame.to_csv(RESULTS / "grid_candidate_results.csv", index=False)

    calibration = pd.DataFrame(
        {
            "scenario_index": np.arange(1, len(scenarios) + 1),
            "disturbance_l1": np.sum(scenarios, axis=1),
            "disturbance_l2": np.linalg.norm(scenarios, axis=1),
            "exact_loss": exact_baseline_losses,
            "quadratic_loss": approximate_baseline_losses,
            "residual": exact_baseline_losses - approximate_baseline_losses,
        }
    )
    calibration.to_csv(RESULTS / "grid_scenario_calibration.csv", index=False)

    candidate_definition_frame = pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "scale": candidate.scale,
                "size": candidate.size,
                "edge_ids": " ".join(str(value) for value in candidate.edge_ids),
            }
            for candidate in candidates
        ]
    )
    candidate_definition_frame.to_csv(RESULTS / "grid_candidates.csv", index=False)
    topology = pd.DataFrame(
        {
            "edge_id": np.arange(network.m),
            "candidate_id": [f"E{edge + 1:02d}" for edge in range(network.m)],
            "u": network.edges[:, 0],
            "v": network.edges[:, 1],
            "weight": network.weights,
            "midpoint_x": network.edge_midpoints[:, 0],
            "midpoint_y": network.edge_midpoints[:, 1],
        }
    )
    topology.to_csv(RESULTS / "grid_topology.csv", index=False)
    pd.DataFrame(
        [
            {"node": node, "x": network.positions[node][0], "y": network.positions[node][1]}
            for node in range(network.n)
        ]
    ).to_csv(RESULTS / "grid_nodes.csv", index=False)

    classical = classical_edge_measures(network, HORIZON, n_time=EXACT_TIME_ORDER)
    edge_results = candidate_frame[candidate_frame["scale"] == "edge"]
    classical = classical.merge(
        edge_results[
            [
                "candidate_id",
                "exact_importance",
                "quadratic_importance",
                "mean_loss_change",
                "interaction_mean_change",
            ]
        ],
        on="candidate_id",
        how="left",
    )
    classical["direct_coefficient"] = g
    classical.to_csv(RESULTS / "grid_edge_baselines.csv", index=False)

    rank_rows = []
    for scale, group in candidate_frame.groupby("scale"):
        metrics = rank_metrics(group["exact_importance"], group["quadratic_importance"])
        top_count = min(5, len(group))
        exact_top = set(group.nlargest(top_count, "exact_importance")["candidate_id"])
        approximate_top = set(
            group.nlargest(top_count, "quadratic_importance")["candidate_id"]
        )
        rank_rows.append(
            {
                "scale": scale,
                **metrics,
                "top5_overlap": len(exact_top & approximate_top) / top_count,
                "maximum_importance_error": float(np.max(np.abs(group["importance_error"]))),
            }
        )
    for measure in [
        "direct_coefficient",
        "betweenness",
        "resistance_leverage",
        "deletion_dpc_loss",
    ]:
        rank_rows.append(
            {
                "scale": f"edge baseline: {measure}",
                "spearman": float(
                    spearmanr(classical["exact_importance"], classical[measure]).statistic
                ),
                "mean_absolute_rank_error": float("nan"),
                "top5_overlap": len(
                    set(classical.nlargest(5, "exact_importance")["candidate_id"])
                    & set(classical.nlargest(5, measure)["candidate_id"])
                )
                / 5.0,
                "maximum_importance_error": float("nan"),
            }
        )
    pd.DataFrame(rank_rows).to_csv(RESULTS / "grid_rank_metrics.csv", index=False)

    save_npz(
        RESULTS / "grid_experiment_arrays.npz",
        d0=np.array([d0]),
        g=g,
        q=q,
        edges=network.edges,
        weights=network.weights,
        positions=np.asarray([network.positions[node] for node in range(network.n)]),
        scenarios=scenarios,
        alpha_bar=alpha_bar,
        exact_baseline_losses=exact_baseline_losses,
        approximate_baseline_losses=approximate_baseline_losses,
        exact_stressed_losses=exact_stressed_matrix,
        approximate_stressed_losses=approximate_stressed_matrix,
        tau=np.array([tau]),
        theta=np.array([theta]),
        support_epsilon=np.array([support_epsilon]),
        global_epsilon=np.array([global_epsilon]),
    )
    write_json(
        RESULTS / "grid_experiment_metadata.json",
        {
            "network": network.name,
            "n": network.n,
            "m": network.m,
            "candidate_count": len(candidates),
            "candidate_counts": candidate_frame.groupby("scale").size().to_dict(),
            "scenario_count": len(scenarios),
            "rho": GRID_RHO,
            "alpha_bar": GRID_ALPHA_BAR,
            "stress": GRID_STRESS,
            "gamma": GRID_GAMMA,
            "tau": tau,
            "theta": theta,
            "d0": d0,
            "baseline_exact_reliability": baseline_exact_reliability,
            "baseline_quadratic_reliability": baseline_approximate_reliability,
            "support_epsilon": support_epsilon,
            "global_epsilon": global_epsilon,
            "maximum_baseline_residual": float(
                np.max(np.abs(exact_baseline_losses - approximate_baseline_losses))
            ),
            "baseline_rmse": float(
                np.sqrt(np.mean((exact_baseline_losses - approximate_baseline_losses) ** 2))
            ),
            "final_coefficient_timing": final_timing,
            "all_support_intervals_contain_exact": bool(
                candidate_frame["support_contains_exact"].all()
            ),
        },
    )
    logging.info("Completed 5 by 5 grid experiment")


def load_grid_state() -> tuple:
    arrays = np.load(RESULTS / "grid_experiment_arrays.npz")
    network = grid_network(GRID_ROWS, GRID_COLUMNS, SEED)
    candidates = grid_candidates(network, GRID_ROWS, GRID_COLUMNS)
    frame = pd.read_csv(RESULTS / "grid_candidate_results.csv")
    return arrays, network, candidates, frame


def run_grid_ablation_experiment() -> None:
    logging.info("Starting grid interaction ablation")
    arrays, _, candidates, grid_results = load_grid_state()
    scenarios = arrays["scenarios"]
    alpha_bar = arrays["alpha_bar"]
    g = arrays["g"]
    q = arrays["q"]
    theta = float(arrays["theta"][0])
    exact_lookup = grid_results.set_index("candidate_id")["exact_importance"]
    model_matrices = {
        "first_order": np.zeros_like(q),
        "diagonal_second_order": np.diag(np.diag(q)),
        "full_second_order": q,
    }
    rows: list[dict] = []
    for method, q_model in model_matrices.items():
        baseline_losses = np.asarray(quadratic_loss(scenarios, g, q_model))
        baseline_reliability = exact_reliability(baseline_losses, theta)
        qx = scenarios @ q_model.T
        for candidate in candidates:
            stressed_losses = incremental_quadratic_loss(
                scenarios,
                candidate.edge_ids,
                GRID_STRESS,
                alpha_bar,
                g,
                q_model,
                baseline_losses=baseline_losses,
                qx=qx,
            )
            estimated_importance = (
                baseline_reliability - exact_reliability(stressed_losses, theta)
            )
            exact_importance = float(exact_lookup.loc[candidate.candidate_id])
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "scale": candidate.scale,
                    "size": candidate.size,
                    "method": method,
                    "exact_importance": exact_importance,
                    "estimated_importance": estimated_importance,
                    "importance_error": estimated_importance - exact_importance,
                    "absolute_importance_error": abs(
                        estimated_importance - exact_importance
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["exact_rank"] = frame.groupby(["method", "scale"])[
        "exact_importance"
    ].rank(ascending=False, method="average")
    frame["estimated_rank"] = frame.groupby(["method", "scale"])[
        "estimated_importance"
    ].rank(ascending=False, method="average")
    frame.to_csv(RESULTS / "grid_ablation_candidate_results.csv", index=False)

    metric_rows: list[dict] = []
    sign_tolerance = 0.5 / len(scenarios)
    for (method, scale), group in frame.groupby(["method", "scale"], sort=False):
        candidate_ids = group["candidate_id"].to_numpy(dtype=str)
        exact_values = group["exact_importance"].to_numpy(dtype=float)
        estimated_values = group["estimated_importance"].to_numpy(dtype=float)
        top_count = min(5, len(group))
        metric_rows.append(
            {
                "method": method,
                "scale": scale,
                "candidate_count": len(group),
                "mean_absolute_importance_error": float(
                    np.mean(np.abs(estimated_values - exact_values))
                ),
                "root_mean_squared_importance_error": float(
                    np.sqrt(np.mean((estimated_values - exact_values) ** 2))
                ),
                "maximum_absolute_importance_error": float(
                    np.max(np.abs(estimated_values - exact_values))
                ),
                "spearman": safe_spearman(exact_values, estimated_values),
                "mean_absolute_rank_error": float(
                    np.mean(
                        np.abs(
                            group["estimated_rank"].to_numpy(dtype=float)
                            - group["exact_rank"].to_numpy(dtype=float)
                        )
                    )
                ),
                "top5_overlap": len(
                    stable_top_ids(candidate_ids, exact_values, top_count)
                    & stable_top_ids(candidate_ids, estimated_values, top_count)
                )
                / top_count,
                "signed_agreement": signed_agreement(
                    exact_values,
                    estimated_values,
                    sign_tolerance,
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "grid_ablation_metrics.csv", index=False)
    logging.info("Completed grid interaction ablation")


def run_topology_robustness_experiment() -> None:
    logging.info("Starting topology robustness experiment")
    candidate_rows: list[dict] = []
    metric_rows: list[dict] = []
    network_rows: list[dict] = []
    importance_measures = {
        "first_order_importance",
        "diagonal_second_order_importance",
        "full_second_order_importance",
    }
    ranking_measures = [
        "full_second_order_importance",
        "diagonal_second_order_importance",
        "first_order_importance",
        "direct_coefficient",
        "deletion_dpc_loss",
        "betweenness",
        "resistance_leverage",
        "algebraic_connectivity_loss",
    ]
    total_networks = (
        len(TOPOLOGY_FAMILIES)
        * len(TOPOLOGY_SIZES)
        * TOPOLOGY_REPLICATES
    )
    completed = 0
    for family_index, family in enumerate(TOPOLOGY_FAMILIES):
        for n in TOPOLOGY_SIZES:
            for replicate in range(TOPOLOGY_REPLICATES):
                network_seed = (
                    SEED
                    + 100000
                    + family_index * 10000
                    + n * 100
                    + replicate
                )
                network_id = f"{family}_n{n}_r{replicate + 1}"
                network = synthetic_topology_network(
                    family,
                    n,
                    network_seed,
                    name=network_id,
                )
                with threadpool_limits(limits=1):
                    d0_coefficient, g, q, coefficient_timing = (
                        rank_aware_coefficients(
                            network,
                            HORIZON,
                            n_time=TOPOLOGY_COEFFICIENT_TIME,
                            n_inner=TOPOLOGY_COEFFICIENT_INNER,
                        )
                    )
                    d0_exact = dpc(
                        network,
                        np.zeros(network.m),
                        HORIZON,
                        n_time=EXACT_TIME_ORDER,
                    )
                    alpha_bar = np.full(network.m, TOPOLOGY_ALPHA_BAR)
                    scenarios = gaussian_copula_beta_scenarios(
                        network.m,
                        TOPOLOGY_SCENARIOS,
                        alpha_bar,
                        TOPOLOGY_RHO,
                        network_seed + 1,
                    )
                    exact_start = perf_counter()
                    exact_baseline_dpc = dpc_batch(
                        network,
                        scenarios,
                        HORIZON,
                        n_time=TOPOLOGY_EXACT_TIME_ORDER,
                    )
                    exact_baseline_losses = d0_exact - exact_baseline_dpc
                    tau = TOPOLOGY_GAMMA * d0_exact
                    theta = d0_exact - tau
                    exact_baseline_reliability = exact_reliability(
                        exact_baseline_losses,
                        theta,
                    )
                    model_matrices = {
                        "first_order_importance": np.zeros_like(q),
                        "diagonal_second_order_importance": np.diag(np.diag(q)),
                        "full_second_order_importance": q,
                    }
                    model_state: dict[str, dict[str, np.ndarray | float]] = {}
                    for measure, q_model in model_matrices.items():
                        losses = np.asarray(quadratic_loss(scenarios, g, q_model))
                        model_state[measure] = {
                            "q": q_model,
                            "baseline_losses": losses,
                            "baseline_reliability": exact_reliability(losses, theta),
                            "qx": scenarios @ q_model.T,
                        }

                    network_candidates: list[dict] = []
                    for edge_id, (u, v) in enumerate(network.edges):
                        stressed_values = stress_scenarios(
                            scenarios,
                            (edge_id,),
                            TOPOLOGY_STRESS,
                            alpha_bar,
                        )
                        stressed_exact_losses = d0_exact - dpc_batch(
                            network,
                            stressed_values,
                            HORIZON,
                            n_time=TOPOLOGY_EXACT_TIME_ORDER,
                        )
                        row = {
                            "network_id": network_id,
                            "family": family,
                            "n": network.n,
                            "m": network.m,
                            "replicate": replicate + 1,
                            "seed": network_seed,
                            "candidate_id": f"E{edge_id + 1:02d}",
                            "edge_id": edge_id,
                            "u": int(u),
                            "v": int(v),
                            "weight": float(network.weights[edge_id]),
                            "exact_importance": (
                                exact_baseline_reliability
                                - exact_reliability(stressed_exact_losses, theta)
                            ),
                            "direct_coefficient": float(g[edge_id]),
                        }
                        for measure, state in model_state.items():
                            stressed_losses = incremental_quadratic_loss(
                                scenarios,
                                (edge_id,),
                                TOPOLOGY_STRESS,
                                alpha_bar,
                                g,
                                np.asarray(state["q"]),
                                baseline_losses=np.asarray(
                                    state["baseline_losses"]
                                ),
                                qx=np.asarray(state["qx"]),
                            )
                            row[measure] = float(
                                state["baseline_reliability"]
                                - exact_reliability(stressed_losses, theta)
                            )
                        network_candidates.append(row)
                    exact_seconds = perf_counter() - exact_start

                    classical = classical_edge_measures(
                        network,
                        HORIZON,
                        n_time=TOPOLOGY_EXACT_TIME_ORDER,
                    ).set_index("edge_id")
                    for row in network_candidates:
                        classical_row = classical.loc[int(row["edge_id"])]
                        for measure in [
                            "betweenness",
                            "resistance_leverage",
                            "deletion_dpc_loss",
                            "algebraic_connectivity_loss",
                            "is_bridge",
                        ]:
                            row[measure] = float(classical_row[measure])

                    validation_count = min(8, len(scenarios))
                    high_order_dpc = dpc_batch(
                        network,
                        scenarios[:validation_count],
                        HORIZON,
                        n_time=EXACT_TIME_ORDER,
                    )
                    quadrature_error = float(
                        np.max(
                            np.abs(
                                high_order_dpc
                                - exact_baseline_dpc[:validation_count]
                            )
                        )
                    )

                candidate_frame = pd.DataFrame(network_candidates)
                candidate_rows.extend(network_candidates)
                candidate_ids = candidate_frame["candidate_id"].to_numpy(dtype=str)
                exact_values = candidate_frame["exact_importance"].to_numpy(
                    dtype=float
                )
                exact_importance_range = float(np.ptp(exact_values))
                ranking_defined = exact_importance_range > 1.0e-15
                top_count = min(5, len(candidate_frame))
                sign_tolerance = 0.5 / TOPOLOGY_SCENARIOS
                network_metric_rows = []
                for measure in ranking_measures:
                    measure_values = candidate_frame[measure].to_numpy(dtype=float)
                    row = {
                        "network_id": network_id,
                        "family": family,
                        "n": network.n,
                        "m": network.m,
                        "replicate": replicate + 1,
                        "measure": measure,
                        "spearman": (
                            safe_spearman(exact_values, measure_values)
                            if ranking_defined
                            else float("nan")
                        ),
                        "top5_overlap": (
                            len(
                                stable_top_ids(
                                    candidate_ids,
                                    exact_values,
                                    top_count,
                                )
                                & stable_top_ids(
                                    candidate_ids,
                                    measure_values,
                                    top_count,
                                )
                            )
                            / top_count
                            if ranking_defined
                            else float("nan")
                        ),
                        "mean_absolute_importance_error": float("nan"),
                        "root_mean_squared_importance_error": float("nan"),
                        "signed_agreement": float("nan"),
                    }
                    if measure in importance_measures:
                        row["mean_absolute_importance_error"] = float(
                            np.mean(np.abs(measure_values - exact_values))
                        )
                        row["root_mean_squared_importance_error"] = float(
                            np.sqrt(np.mean((measure_values - exact_values) ** 2))
                        )
                        row["signed_agreement"] = signed_agreement(
                            exact_values,
                            measure_values,
                            sign_tolerance,
                        )
                    network_metric_rows.append(row)
                metric_rows.extend(network_metric_rows)
                full_metric = next(
                    row
                    for row in network_metric_rows
                    if row["measure"] == "full_second_order_importance"
                )
                q_diagonal = np.diag(np.diag(q))
                q_norm = max(float(np.linalg.norm(q)), 1.0e-15)
                network_rows.append(
                    {
                        "network_id": network_id,
                        "family": family,
                        "n": network.n,
                        "m": network.m,
                        "replicate": replicate + 1,
                        "seed": network_seed,
                        "scenario_count": TOPOLOGY_SCENARIOS,
                        "alpha_bar": TOPOLOGY_ALPHA_BAR,
                        "rho": TOPOLOGY_RHO,
                        "stress": TOPOLOGY_STRESS,
                        "gamma": TOPOLOGY_GAMMA,
                        "d0_exact": d0_exact,
                        "d0_coefficient": d0_coefficient,
                        "d0_absolute_difference": abs(
                            d0_exact - d0_coefficient
                        ),
                        "tau": tau,
                        "baseline_exact_reliability": exact_baseline_reliability,
                        "coefficient_seconds": coefficient_timing[
                            "coefficient_seconds"
                        ],
                        "exact_candidate_seconds": exact_seconds,
                        "exact_time_order_validation_error": quadrature_error,
                        "q_symmetry_error": float(np.max(np.abs(q - q.T))),
                        "off_diagonal_q_fraction": float(
                            np.linalg.norm(q - q_diagonal) / q_norm
                        ),
                        "negative_exact_importance_share": float(
                            np.mean(exact_values < -sign_tolerance)
                        ),
                        "positive_exact_importance_share": float(
                            np.mean(exact_values > sign_tolerance)
                        ),
                        "exact_importance_range": exact_importance_range,
                        "ranking_defined": ranking_defined,
                        "full_second_order_mae": full_metric[
                            "mean_absolute_importance_error"
                        ],
                        "full_second_order_spearman": full_metric["spearman"],
                    }
                )
                completed += 1
                logging.info(
                    "Topology robustness completed for %s (%d of %d)",
                    network_id,
                    completed,
                    total_networks,
                )

    candidate_frame = pd.DataFrame(candidate_rows)
    metric_frame = pd.DataFrame(metric_rows)
    network_frame = pd.DataFrame(network_rows)
    candidate_frame.to_csv(RESULTS / "topology_edge_results.csv", index=False)
    metric_frame.to_csv(RESULTS / "topology_rank_metrics.csv", index=False)
    network_frame.to_csv(RESULTS / "topology_network_summary.csv", index=False)

    aggregate_rows: list[dict] = []
    family_groups = list(TOPOLOGY_FAMILIES) + ["all"]
    for family in family_groups:
        family_frame = (
            metric_frame
            if family == "all"
            else metric_frame[metric_frame["family"] == family]
        )
        for measure in ranking_measures:
            group = family_frame[family_frame["measure"] == measure]
            spearman_values = group["spearman"].dropna().to_numpy(dtype=float)
            top_values = group["top5_overlap"].dropna().to_numpy(dtype=float)
            signed_values = group["signed_agreement"].dropna().to_numpy(dtype=float)
            aggregate_rows.append(
                {
                    "family": family,
                    "measure": measure,
                    "network_count": len(group),
                    "valid_spearman_count": len(spearman_values),
                    "valid_top5_count": len(top_values),
                    "median_spearman": (
                        float(np.median(spearman_values))
                        if len(spearman_values)
                        else float("nan")
                    ),
                    "spearman_q25": (
                        float(np.quantile(spearman_values, 0.25))
                        if len(spearman_values)
                        else float("nan")
                    ),
                    "spearman_q75": (
                        float(np.quantile(spearman_values, 0.75))
                        if len(spearman_values)
                        else float("nan")
                    ),
                    "median_top5_overlap": (
                        float(np.median(top_values))
                        if len(top_values)
                        else float("nan")
                    ),
                    "mean_signed_agreement": (
                        float(np.mean(signed_values))
                        if len(signed_values)
                        else float("nan")
                    ),
                }
            )
    pd.DataFrame(aggregate_rows).to_csv(
        RESULTS / "topology_rank_aggregate.csv",
        index=False,
    )
    write_json(
        RESULTS / "topology_robustness_metadata.json",
        topology_metadata_payload(),
    )
    logging.info("Completed topology robustness experiment")


def run_strengthening_experiments() -> None:
    run_grid_ablation_experiment()
    run_topology_robustness_experiment()


def run_correlation_experiment() -> None:
    logging.info("Starting correlation experiment")
    arrays, network, candidates, grid_results = load_grid_state()
    d0 = float(arrays["d0"][0])
    g = arrays["g"]
    q = arrays["q"]
    alpha_bar = arrays["alpha_bar"]
    theta = float(arrays["theta"][0])
    tau = float(arrays["tau"][0])
    top_candidates: dict[str, Candidate] = {}
    for scale in ("edge", "corridor", "region"):
        top_id = grid_results[grid_results["scale"] == scale].nlargest(
            1, "exact_importance"
        )["candidate_id"].iloc[0]
        top_candidates[scale] = next(
            candidate for candidate in candidates if candidate.candidate_id == top_id
        )

    rho_values = [0.0, 0.25, 0.50, 0.75]
    scenario_count = 320
    summary_rows = []
    candidate_rows = []
    rank_rows = []
    reference_importance: dict[str, pd.Series] = {}
    for rho_index, rho in enumerate(rho_values):
        scenarios = gaussian_copula_beta_scenarios(
            network.m,
            scenario_count,
            alpha_bar,
            rho,
            SEED + 100 + rho_index,
        )
        exact_losses = d0 - dpc_batch(
            network, scenarios, HORIZON, n_time=EXACT_TIME_ORDER
        )
        approximate_losses = np.asarray(quadratic_loss(scenarios, g, q))
        exact_baseline_reliability = exact_reliability(exact_losses, theta)
        approximate_baseline_reliability = exact_reliability(approximate_losses, theta)
        summary_rows.append(
            {
                "rho": rho,
                "scenario_count": scenario_count,
                "exact_reliability": exact_baseline_reliability,
                "quadratic_reliability": approximate_baseline_reliability,
                "loss_rmse": float(
                    np.sqrt(np.mean((exact_losses - approximate_losses) ** 2))
                ),
                "loss_spearman": float(
                    spearmanr(exact_losses, approximate_losses).statistic
                ),
                "tau": tau,
            }
        )
        qx = scenarios @ q.T
        all_rows = []
        for candidate in candidates:
            stressed_approximate = incremental_quadratic_loss(
                scenarios,
                candidate.edge_ids,
                GRID_STRESS,
                alpha_bar,
                g,
                q,
                baseline_losses=approximate_losses,
                qx=qx,
            )
            all_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "scale": candidate.scale,
                    "rho": rho,
                    "quadratic_importance": approximate_baseline_reliability
                    - exact_reliability(stressed_approximate, theta),
                }
            )
        current_frame = pd.DataFrame(all_rows)
        candidate_rows.extend(all_rows)
        for scale in ("edge", "corridor", "region"):
            current = current_frame[current_frame["scale"] == scale].set_index(
                "candidate_id"
            )["quadratic_importance"]
            if rho_index == 0:
                reference_importance[scale] = current
                stability = 1.0
            else:
                stability = float(
                    spearmanr(
                        reference_importance[scale].sort_index(),
                        current.sort_index(),
                    ).statistic
                )
            rank_rows.append({"rho": rho, "scale": scale, "rank_spearman": stability})

        for scale, candidate in top_candidates.items():
            stressed = stress_scenarios(
                scenarios, candidate.edge_ids, GRID_STRESS, alpha_bar
            )
            stressed_exact_losses = d0 - dpc_batch(
                network, stressed, HORIZON, n_time=EXACT_TIME_ORDER
            )
            stressed_approximate_losses = np.asarray(quadratic_loss(stressed, g, q))
            candidate_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "scale": f"{scale}_exact_top",
                    "rho": rho,
                    "quadratic_importance": approximate_baseline_reliability
                    - exact_reliability(stressed_approximate_losses, theta),
                    "exact_importance": exact_baseline_reliability
                    - exact_reliability(stressed_exact_losses, theta),
                }
            )
        logging.info("Completed correlation level rho=%.2f", rho)
    pd.DataFrame(summary_rows).to_csv(
        RESULTS / "correlation_reliability_summary.csv", index=False
    )
    pd.DataFrame(candidate_rows).to_csv(
        RESULTS / "correlation_candidate_importance.csv", index=False
    )
    pd.DataFrame(rank_rows).to_csv(
        RESULTS / "correlation_rank_stability.csv", index=False
    )
    logging.info("Completed correlation experiment")


def run_sensitivity_experiment() -> None:
    logging.info("Starting stress and threshold sensitivity experiment")
    arrays, network, candidates, grid_results = load_grid_state()
    d0 = float(arrays["d0"][0])
    g = arrays["g"]
    q = arrays["q"]
    scenarios = arrays["scenarios"]
    alpha_bar = arrays["alpha_bar"]
    exact_baseline_losses = arrays["exact_baseline_losses"]
    approximate_baseline_losses = arrays["approximate_baseline_losses"]
    global_epsilon = float(arrays["global_epsilon"][0])
    qx = scenarios @ q.T

    selected_candidates = []
    for scale in ("edge", "corridor", "region"):
        top_id = grid_results[grid_results["scale"] == scale].nlargest(
            1, "exact_importance"
        )["candidate_id"].iloc[0]
        selected_candidates.append(
            next(candidate for candidate in candidates if candidate.candidate_id == top_id)
        )

    lambda_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    gamma_values = [0.94, 0.95, 0.96, 0.97, 0.98]
    rows = []
    for candidate in selected_candidates:
        for stress in lambda_values:
            stressed_scenario_values = stress_scenarios(
                scenarios, candidate.edge_ids, stress, alpha_bar
            )
            exact_stressed_losses = d0 - dpc_batch(
                network,
                stressed_scenario_values,
                HORIZON,
                n_time=EXACT_TIME_ORDER,
            )
            approximate_stressed_losses = incremental_quadratic_loss(
                scenarios,
                candidate.edge_ids,
                stress,
                alpha_bar,
                g,
                q,
                baseline_losses=approximate_baseline_losses,
                qx=qx,
            )
            support_epsilon = max(
                float(
                    np.max(
                        np.abs(exact_baseline_losses - approximate_baseline_losses)
                    )
                ),
                float(
                    np.max(
                        np.abs(exact_stressed_losses - approximate_stressed_losses)
                    )
                ),
            )
            for gamma in gamma_values:
                tau = gamma * d0
                theta = d0 - tau
                exact_baseline_reliability = exact_reliability(
                    exact_baseline_losses, theta
                )
                approximate_baseline_reliability = exact_reliability(
                    approximate_baseline_losses, theta
                )
                exact_importance = exact_baseline_reliability - exact_reliability(
                    exact_stressed_losses, theta
                )
                approximate_importance = (
                    approximate_baseline_reliability
                    - exact_reliability(approximate_stressed_losses, theta)
                )
                baseline_support = moment_interval(
                    approximate_baseline_losses, theta, support_epsilon
                )
                stressed_support = moment_interval(
                    approximate_stressed_losses, theta, support_epsilon
                )
                support_lower = baseline_support[0] - stressed_support[1]
                support_upper = baseline_support[1] - stressed_support[0]
                baseline_global = moment_interval(
                    approximate_baseline_losses, theta, global_epsilon
                )
                stressed_global = moment_interval(
                    approximate_stressed_losses, theta, global_epsilon
                )
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "scale": candidate.scale,
                        "stress": stress,
                        "gamma": gamma,
                        "tau": tau,
                        "exact_importance": exact_importance,
                        "quadratic_importance": approximate_importance,
                        "support_epsilon": support_epsilon,
                        "support_lower": support_lower,
                        "support_upper": support_upper,
                        "support_status": status_from_interval(
                            support_lower, support_upper
                        ),
                        "support_contains_exact": support_lower - 1.0e-12
                        <= exact_importance
                        <= support_upper + 1.0e-12,
                        "global_lower": baseline_global[0] - stressed_global[1],
                        "global_upper": baseline_global[1] - stressed_global[0],
                    }
                )
            logging.info(
                "Sensitivity completed for %s at stress %.2f",
                candidate.candidate_id,
                stress,
            )
    pd.DataFrame(rows).to_csv(RESULTS / "sensitivity_results.csv", index=False)
    logging.info("Completed stress and threshold sensitivity experiment")


def generate_candidate_arrays(
    rng: np.random.Generator,
    m: int,
    count: int,
    size: int,
) -> tuple[list[np.ndarray], float]:
    start = perf_counter()
    candidates = [
        np.sort(rng.choice(m, size=min(size, m), replace=False)).astype(int)
        for _ in range(count)
    ]
    return candidates, perf_counter() - start


def time_scoring_case(
    m: int,
    scenario_count: int,
    candidate_count: int,
    candidate_size: int,
    route: str,
    seed: int,
    repetitions: int = 3,
) -> dict:
    rng = np.random.default_rng(seed)
    scenarios = rng.uniform(0.0, 0.20, size=(scenario_count, m))
    alpha_bar = np.full(m, 0.22)
    g = rng.normal(0.0, 0.08, size=m)
    candidates, generation_seconds = generate_candidate_arrays(
        rng, m, candidate_count, candidate_size
    )
    if route == "dense":
        raw = rng.normal(0.0, 0.04 / np.sqrt(max(m, 1)), size=(m, m))
        q = 0.5 * (raw + raw.T)
        density = 1.0
    else:
        raw = sparse.random(
            m,
            m,
            density=0.03,
            random_state=rng,
            data_rvs=lambda count: rng.normal(0.0, 0.04, size=count),
            format="csr",
        )
        q = 0.5 * (raw + raw.T)
        q = q + sparse.diags(rng.normal(0.0, 0.04, size=m))
        density = float(q.nnz / (m * m))
    sparse_scoring_kernel(scenarios[: min(20, scenario_count)], candidates[:5], 0.35, alpha_bar, g, q)
    timings = []
    for _ in range(repetitions):
        timings.append(
            sparse_scoring_kernel(scenarios, candidates, 0.35, alpha_bar, g, q)
        )
    return {
        "m": m,
        "scenario_count": scenario_count,
        "candidate_count": candidate_count,
        "candidate_size": candidate_size,
        "route": route,
        "density": density,
        "candidate_generation_seconds": generation_seconds,
        "scenario_preprocessing_seconds": float(
            np.median([value["scenario_preprocessing_seconds"] for value in timings])
        ),
        "candidate_scoring_seconds": float(
            np.median([value["candidate_scoring_seconds"] for value in timings])
        ),
        "sorting_seconds": float(
            np.median([value["sorting_seconds"] for value in timings])
        ),
        "checksum": timings[-1]["checksum"],
    }


def run_runtime_experiment() -> None:
    logging.info("Starting runtime and complexity experiment")
    coefficient_rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        for n in [6, 8, 10, 12, 14, 16]:
            for replicate in range(3):
                network = random_connected_network(
                    n,
                    min(0.45, 4.0 / max(n - 1, 1)),
                    SEED + 500 + 10 * n + replicate,
                    name=f"runtime_n{n}_r{replicate}",
                )
                _, _, _, timing = rank_aware_coefficients(
                    network, HORIZON, n_time=8, n_inner=3
                )
                theoretical_work = timing["n_quad"] * (
                    n**3 + network.m**2 * n**2
                )
                coefficient_rows.append(
                    {
                        "n": n,
                        "m": network.m,
                        "replicate": replicate,
                        "n_time": 8,
                        "n_inner": 3,
                        "n_quad": timing["n_quad"],
                        "theoretical_work": theoretical_work,
                        "coefficient_seconds": timing["coefficient_seconds"],
                        "second_integral_seconds": timing["second_integral_seconds"],
                        "blas_threads": 1,
                    }
                )
            logging.info("Coefficient timing completed for n=%d", n)
    pd.DataFrame(coefficient_rows).to_csv(
        RESULTS / "runtime_coefficient_scaling.csv", index=False
    )

    scoring_rows = []
    case_index = 0
    with threadpool_limits(limits=1, user_api="blas"):
        for route in ("dense", "sparse"):
            for m in [50, 100, 200, 400]:
                row = time_scoring_case(
                    m, 500, 250, 5, route, SEED + 1000 + case_index, repetitions=5
                )
                row.update({"experiment_axis": "edge_count", "axis_value": m, "blas_threads": 1})
                scoring_rows.append(row)
                case_index += 1
            for scenario_count in [100, 250, 500, 1000, 2000, 4000]:
                row = time_scoring_case(
                    120,
                    scenario_count,
                    250,
                    5,
                    route,
                    SEED + 1000 + case_index,
                    repetitions=5,
                )
                row.update(
                    {
                        "experiment_axis": "scenario_count",
                        "axis_value": scenario_count,
                        "blas_threads": 1,
                    }
                )
                scoring_rows.append(row)
                case_index += 1
            for candidate_count in [50, 100, 250, 500, 1000]:
                row = time_scoring_case(
                    120,
                    500,
                    candidate_count,
                    5,
                    route,
                    SEED + 1000 + case_index,
                    repetitions=5,
                )
                row.update(
                    {
                        "experiment_axis": "candidate_count",
                        "axis_value": candidate_count,
                        "blas_threads": 1,
                    }
                )
                scoring_rows.append(row)
                case_index += 1
            for candidate_size in [1, 2, 4, 8, 16, 24]:
                row = time_scoring_case(
                    120,
                    500,
                    250,
                    candidate_size,
                    route,
                    SEED + 1000 + case_index,
                    repetitions=5,
                )
                row.update(
                    {
                        "experiment_axis": "candidate_size",
                        "axis_value": candidate_size,
                        "blas_threads": 1,
                    }
                )
                scoring_rows.append(row)
                case_index += 1
            logging.info("Scoring timing completed for %s route", route)
    pd.DataFrame(scoring_rows).to_csv(
        RESULTS / "runtime_scoring_scaling.csv", index=False
    )

    network = grid_network(4, 4, SEED + 77)
    with threadpool_limits(limits=1, user_api="blas"):
        full_start = perf_counter()
        _, g, q, full_timing = rank_aware_coefficients(
            network, HORIZON, n_time=10, n_inner=3
        )
        full_build_seconds = perf_counter() - full_start
    rng = np.random.default_rng(SEED + 88)
    directions = rng.uniform(-0.2, 0.2, size=(100, network.m))
    full_query_times = []
    full_values = []
    matrix_free_times = []
    matrix_free_values = []
    with threadpool_limits(limits=1, user_api="blas"):
        for direction in directions:
            start = perf_counter()
            full_values.append(float(direction @ q @ direction))
            full_query_times.append(perf_counter() - start)
        for index, direction in enumerate(directions, start=1):
            start = perf_counter()
            _, second = directional_loss_derivatives(
                network,
                direction,
                HORIZON,
                n_time=EXACT_TIME_ORDER,
                step=8.0e-4,
            )
            matrix_free_times.append(perf_counter() - start)
            matrix_free_values.append(second)
            if index % 20 == 0:
                logging.info("Matrix-free timing completed for %d directions", index)
    crossover_rows = []
    for query_count in [1, 2, 5, 10, 20, 50, 100]:
        relative_error = np.linalg.norm(
            np.asarray(matrix_free_values[:query_count])
            - np.asarray(full_values[:query_count])
        ) / max(np.linalg.norm(full_values[:query_count]), 1.0e-15)
        crossover_rows.append(
            {
                "query_count": query_count,
                "full_interaction_seconds": full_build_seconds
                + float(np.sum(full_query_times[:query_count])),
                "matrix_free_seconds": float(
                    np.sum(matrix_free_times[:query_count])
                ),
                "relative_directional_error": float(relative_error),
                "n": network.n,
                "m": network.m,
                "full_n_quad": full_timing["n_quad"],
            }
        )
    pd.DataFrame(crossover_rows).to_csv(
        RESULTS / "runtime_matrix_free_crossover.csv", index=False
    )
    logging.info("Completed runtime and complexity experiment")


def run_result_audit() -> None:
    logging.info("Starting numerical result audit")
    checks: dict[str, object] = {}
    pilot = json.loads((LOGS / "pilot_validation.json").read_text(encoding="utf-8"))
    checks["pilot_q_symmetry_error"] = pilot["q_symmetry_error"]
    checks["pilot_incremental_identity_error"] = pilot["incremental_identity_error"]
    checks["pilot_small_delta_residual_slope"] = pilot["small_delta_residual_slope"]
    checks["pilot_pass"] = bool(
        pilot["q_symmetry_error"] < 1.0e-12
        and pilot["incremental_identity_error"] < 1.0e-12
        and 2.7 < pilot["small_delta_residual_slope"] < 3.3
        and pilot["q_relative_frobenius_difference"] < 1.0e-5
    )

    grid = pd.read_csv(RESULTS / "grid_candidate_results.csv")
    calibration = pd.read_csv(RESULTS / "grid_scenario_calibration.csv")
    small = pd.read_csv(RESULTS / "small_exact_reliability_validation.csv")
    sensitivity = pd.read_csv(RESULTS / "sensitivity_results.csv")
    crossover = pd.read_csv(RESULTS / "runtime_matrix_free_crossover.csv")
    ablation = pd.read_csv(RESULTS / "grid_ablation_metrics.csv")
    ablation_candidates = pd.read_csv(
        RESULTS / "grid_ablation_candidate_results.csv"
    )
    topology_candidates = pd.read_csv(RESULTS / "topology_edge_results.csv")
    topology_metrics = pd.read_csv(RESULTS / "topology_rank_metrics.csv")
    topology_networks = pd.read_csv(RESULTS / "topology_network_summary.csv")
    checks["grid_support_interval_coverage"] = float(
        np.mean(grid["support_contains_exact"].astype(bool))
    )
    checks["small_support_interval_coverage"] = float(
        np.mean(small["support_contains_exact"].astype(bool))
    )
    checks["sensitivity_support_interval_coverage"] = float(
        np.mean(sensitivity["support_contains_exact"].astype(bool))
    )
    checks["grid_loss_rmse"] = float(
        np.sqrt(np.mean(calibration["residual"] ** 2))
    )
    checks["grid_loss_max_absolute_error"] = float(
        np.max(np.abs(calibration["residual"]))
    )
    checks["grid_loss_spearman"] = float(
        spearmanr(calibration["exact_loss"], calibration["quadratic_loss"]).statistic
    )
    checks["grid_candidate_importance_mae"] = float(
        np.mean(np.abs(grid["importance_error"]))
    )
    checks["grid_candidate_importance_max_error"] = float(
        np.max(np.abs(grid["importance_error"]))
    )
    checks["matrix_free_max_relative_error"] = float(
        np.max(crossover["relative_directional_error"])
    )
    first_order_ablation = ablation[
        ablation["method"] == "first_order"
    ].set_index("scale")
    full_ablation = ablation[
        ablation["method"] == "full_second_order"
    ].set_index("scale")
    checks["grid_ablation_first_order_mae"] = float(
        first_order_ablation["mean_absolute_importance_error"].mean()
    )
    checks["grid_ablation_full_second_order_mae"] = float(
        full_ablation["mean_absolute_importance_error"].mean()
    )
    checks["grid_ablation_full_improves_every_scale"] = bool(
        (
            full_ablation["mean_absolute_importance_error"]
            < first_order_ablation["mean_absolute_importance_error"]
        ).all()
    )
    full_topology_metrics = topology_metrics[
        topology_metrics["measure"] == "full_second_order_importance"
    ]
    checks["topology_network_count"] = int(
        topology_networks["network_id"].nunique()
    )
    checks["topology_family_count"] = int(topology_networks["family"].nunique())
    checks["topology_seed_count"] = int(topology_networks["seed"].nunique())
    checks["topology_candidate_count"] = int(len(topology_candidates))
    checks["topology_exact_time_order_max_error"] = float(
        topology_networks["exact_time_order_validation_error"].max()
    )
    checks["topology_q_symmetry_max_error"] = float(
        topology_networks["q_symmetry_error"].max()
    )
    checks["topology_full_median_spearman"] = float(
        full_topology_metrics["spearman"].median()
    )
    checks["topology_full_median_top5_overlap"] = float(
        full_topology_metrics["top5_overlap"].median()
    )
    checks["topology_network_share_with_negative_importance"] = float(
        np.mean(topology_networks["negative_exact_importance_share"] > 0.0)
    )
    exact_score_ranges = topology_candidates.groupby("network_id")[
        "exact_importance"
    ].agg(lambda values: float(values.max() - values.min()))
    constant_exact_networks = set(
        exact_score_ranges[exact_score_ranges <= 1.0e-15].index
    )
    undefined_full_spearman_networks = set(
        topology_networks.loc[
            topology_networks["full_second_order_spearman"].isna(),
            "network_id",
        ]
    )
    checks["topology_constant_exact_score_network_count"] = int(
        len(constant_exact_networks)
    )
    checks["topology_undefined_full_spearman_count"] = int(
        len(undefined_full_spearman_networks)
    )
    checks["topology_undefined_spearman_explained"] = bool(
        undefined_full_spearman_networks == constant_exact_networks
    )
    finite_topology_networks = topology_networks.drop(
        columns=["full_second_order_spearman"]
    )
    finite_rank_values = topology_metrics["spearman"].dropna().to_numpy(dtype=float)
    checks["topology_rank_metrics_bounded"] = bool(
        np.isfinite(finite_rank_values).all()
        and (np.abs(finite_rank_values) <= 1.0 + 1.0e-12).all()
        and topology_metrics["top5_overlap"]
        .dropna()
        .between(0.0, 1.0)
        .all()
    )
    checks["all_finite"] = bool(
        np.isfinite(
            grid.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            sensitivity.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            ablation_candidates.select_dtypes(include=[np.number]).to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            topology_candidates.select_dtypes(include=[np.number]).to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            finite_topology_networks.select_dtypes(include=[np.number]).to_numpy(
                dtype=float
            )
        ).all()
    )
    checks["all_required_checks_pass"] = bool(
        checks["pilot_pass"]
        and checks["grid_support_interval_coverage"] == 1.0
        and checks["small_support_interval_coverage"] == 1.0
        and checks["sensitivity_support_interval_coverage"] == 1.0
        and checks["grid_ablation_full_improves_every_scale"]
        and checks["topology_network_count"]
        == len(TOPOLOGY_FAMILIES) * len(TOPOLOGY_SIZES) * TOPOLOGY_REPLICATES
        and checks["topology_family_count"] == len(TOPOLOGY_FAMILIES)
        and checks["topology_seed_count"] == checks["topology_network_count"]
        and checks["topology_exact_time_order_max_error"] < 1.0e-6
        and checks["topology_q_symmetry_max_error"] < 1.0e-12
        and checks["topology_undefined_spearman_explained"]
        and checks["topology_rank_metrics_bounded"]
        and checks["all_finite"]
    )
    write_json(RESULTS / "numerical_audit.json", checks)
    logging.info("Numerical audit outcome: %s", checks["all_required_checks_pass"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "validation",
            "grid",
            "strengthening",
            "correlation",
            "sensitivity",
            "runtime",
            "audit",
        ],
        default="all",
    )
    args = parser.parse_args()
    configure_logging()
    write_json(RESULTS / "environment_metadata.json", environment_metadata())
    start = perf_counter()
    if args.phase in ("all", "validation"):
        run_coefficient_and_motif_validation()
    if args.phase in ("all", "grid"):
        run_grid_experiment()
    if args.phase in ("all", "strengthening"):
        run_strengthening_experiments()
    if args.phase in ("all", "correlation"):
        run_correlation_experiment()
    if args.phase in ("all", "sensitivity"):
        run_sensitivity_experiment()
    if args.phase in ("all", "runtime"):
        run_runtime_experiment()
    if args.phase in ("all", "audit"):
        run_result_audit()
    elapsed = perf_counter() - start
    logging.info("Requested phase %s completed in %.3f seconds", args.phase, elapsed)


if __name__ == "__main__":
    main()
