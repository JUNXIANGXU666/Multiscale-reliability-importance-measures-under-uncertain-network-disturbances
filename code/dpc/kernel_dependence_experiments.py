"""Kernel contrasts, dependence calibration, interventions and runtime experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.spatial.distance import cdist
from scipy.stats import beta, gamma as gamma_distribution, kendalltau, norm
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from dpc_core import (
    Candidate,
    WeightedNetwork,
    dpc_batch,
    exact_binary_scenarios,
    exact_reliability,
    gaussian_copula_beta_scenarios,
    grid_candidates,
    grid_network,
    incremental_quadratic_loss,
    moment_interval,
    motif_candidates,
    motif_networks,
    quadratic_loss,
    rank_aware_coefficients,
    stress_scenarios,
    synthetic_topology_network,
    theoretical_epsilon,
)
from sioux_and_sensitivity import parse_sioux_falls, sioux_candidates


HERE = Path(__file__).resolve().parent
R1_RESULTS = HERE / "results"
R1_EXTENDED = HERE / "sioux_outputs" / "results"
R2_ROOT = HERE / "robustness_outputs"
RESULTS = R2_ROOT / "results"
CHECKPOINTS = R2_ROOT / "checkpoints"
LOGS = R2_ROOT / "logs"
SEED = 20260904

GRID_HORIZON = 1.8
GRID_GAMMA = 0.9625
GRID_STRESS = 0.35
SIOUX_HORIZON = 1.8
SIOUX_GAMMA = 0.97
SIOUX_STRESS = 0.35

KERNELS = ("adjacency_quantum", "laplacian_quantum", "classical_diffusion")


def ensure_directories() -> None:
    for path in (R2_ROOT, RESULTS, CHECKPOINTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    ensure_directories()
    handlers = [
        logging.FileHandler(LOGS / "kernel_dependence_experiments.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def environment_payload() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "seed": SEED,
        "blas_threads": 1,
    }


def normalise_source_weights(
    source_weights: np.ndarray | None, node_count: int
) -> np.ndarray:
    if source_weights is None:
        return np.full(node_count, 1.0 / node_count, dtype=float)
    weights = np.asarray(source_weights, dtype=float)
    if weights.shape != (node_count,) or np.any(weights < 0.0):
        raise ValueError("Invalid source weights")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Source weights must have positive sum")
    return weights / total


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    x = pd.Series(np.asarray(first, dtype=float))
    y = pd.Series(np.asarray(second, dtype=float))
    if len(x) < 2 or np.ptp(x) <= 1.0e-14 or np.ptp(y) <= 1.0e-14:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def sign_class(values: Sequence[float], tolerance: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(array > tolerance, 1, np.where(array < -tolerance, -1, 0))


def top_overlap(
    frame: pd.DataFrame,
    first_column: str,
    second_column: str,
    count: int = 5,
) -> float:
    use_count = min(int(count), len(frame))
    first = set(
        frame.sort_values([first_column, "candidate_id"], ascending=[False, True])
        .head(use_count)["candidate_id"]
        .astype(str)
    )
    second = set(
        frame.sort_values([second_column, "candidate_id"], ascending=[False, True])
        .head(use_count)["candidate_id"]
        .astype(str)
    )
    return float(len(first & second) / max(use_count, 1))


def weighted_matrix(network: WeightedNetwork, alpha: np.ndarray) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    if values.shape != (network.m,):
        raise ValueError(f"alpha must have shape ({network.m},)")
    matrix = np.zeros((network.n, network.n), dtype=float)
    active = network.weights * (1.0 - values)
    for (u, v), weight in zip(network.edges, active, strict=True):
        matrix[int(u), int(v)] += float(weight)
        matrix[int(v), int(u)] += float(weight)
    return matrix


def kernel_matrix(
    network: WeightedNetwork,
    alpha: np.ndarray,
    kernel: str,
    weight_scale: float = 1.0,
) -> np.ndarray:
    adjacency = float(weight_scale) * weighted_matrix(network, alpha)
    if kernel == "adjacency_quantum":
        return adjacency
    if kernel in {"laplacian_quantum", "classical_diffusion"}:
        return np.diag(adjacency.sum(axis=1)) - adjacency
    raise ValueError(f"Unknown propagation kernel: {kernel}")


def metric_from_matrix(
    generator: np.ndarray,
    horizon: float,
    n_time: int,
    source_weights: np.ndarray | None,
    quantum: bool,
) -> float:
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(generator, dtype=float))
    nodes, weights = leggauss(int(n_time))
    times = 0.5 * float(horizon) * (nodes + 1.0)
    quadrature_weights = 0.5 * float(horizon) * weights
    if quantum:
        factors = np.exp(-1j * times[:, None] * eigenvalues[None, :])
        propagators = np.einsum(
            "ia,ta,ja->tij", eigenvectors, factors, eigenvectors, optimize=True
        )
        occupations = np.abs(propagators) ** 2
    else:
        factors = np.exp(-times[:, None] * eigenvalues[None, :])
        propagators = np.einsum(
            "ia,ta,ja->tij", eigenvectors, factors, eigenvectors, optimize=True
        )
        occupations = np.clip(propagators, 0.0, None)
        column_sums = occupations.sum(axis=1, keepdims=True)
        occupations = occupations / np.maximum(column_sums, 1.0e-15)
    concentration = np.sum(occupations * occupations, axis=1)
    participation = 1.0 / concentration
    source_mass = normalise_source_weights(source_weights, generator.shape[0])
    return float(
        np.sum(quadrature_weights[:, None] * participation * source_mass[None, :])
        / float(horizon)
    )


def propagation_metric(
    network: WeightedNetwork,
    alpha: np.ndarray,
    horizon: float,
    n_time: int,
    kernel: str,
    source_weights: np.ndarray | None = None,
    weight_scale: float = 1.0,
) -> float:
    generator = kernel_matrix(network, alpha, kernel, weight_scale=weight_scale)
    return metric_from_matrix(
        generator,
        horizon,
        n_time,
        source_weights,
        quantum=kernel != "classical_diffusion",
    )


def propagation_metric_batch(
    network: WeightedNetwork,
    scenarios: np.ndarray,
    horizon: float,
    n_time: int,
    kernel: str,
    source_weights: np.ndarray | None = None,
    weight_scale: float = 1.0,
) -> np.ndarray:
    values = np.asarray(scenarios, dtype=float)
    return np.fromiter(
        (
            propagation_metric(
                network,
                row,
                horizon,
                n_time,
                kernel,
                source_weights=source_weights,
                weight_scale=weight_scale,
            )
            for row in values
        ),
        dtype=float,
        count=len(values),
    )


def pathwise_equal_total_stress(
    scenarios: np.ndarray,
    candidate: Candidate,
    alpha_bar: np.ndarray,
    total_increment: float,
) -> np.ndarray:
    values = np.asarray(scenarios, dtype=float)
    upper = np.asarray(alpha_bar, dtype=float)
    ids = np.asarray(candidate.edge_ids, dtype=int)
    headroom = upper[None, ids] - values[:, ids]
    total_headroom = headroom.sum(axis=1)
    if np.any(total_headroom + 1.0e-14 < total_increment):
        raise ValueError(
            f"Infeasible pathwise increment for candidate {candidate.candidate_id}"
        )
    stressed = values.copy()
    stressed[:, ids] += (
        float(total_increment) * headroom / total_headroom[:, None]
    )
    return stressed


def direct_candidate_table(
    case_name: str,
    network: WeightedNetwork,
    candidates: Sequence[Candidate],
    scenarios: np.ndarray,
    alpha_bar: np.ndarray,
    horizon: float,
    gamma: float,
    n_time: int,
    kernel: str,
    source_weights: np.ndarray | None,
    stress_mode: str,
    stress_value: float,
    weight_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    stem = (
        f"{case_name}_{kernel}_{stress_mode}_"
        f"{stress_value:.10f}_scale_{weight_scale:.6f}"
    ).replace(".", "p")
    checkpoint = CHECKPOINTS / f"{stem}.csv"
    zero = propagation_metric(
        network,
        np.zeros(network.m),
        horizon,
        n_time,
        kernel,
        source_weights=source_weights,
        weight_scale=weight_scale,
    )
    baseline = propagation_metric_batch(
        network,
        scenarios,
        horizon,
        n_time,
        kernel,
        source_weights=source_weights,
        weight_scale=weight_scale,
    )
    threshold = float(gamma) * zero
    baseline_reliability = float(np.mean(baseline >= threshold))
    completed: dict[str, dict[str, object]] = {}
    if checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        completed = {
            str(row["candidate_id"]): row.to_dict() for _, row in prior.iterrows()
        }
        logging.info("Resuming %s at %d candidates", stem, len(completed))
    rows: list[dict[str, object]] = list(completed.values())
    for index, candidate in enumerate(candidates, start=1):
        if candidate.candidate_id in completed:
            continue
        if stress_mode == "affine":
            stressed = stress_scenarios(
                scenarios, candidate.edge_ids, float(stress_value), alpha_bar
            )
        elif stress_mode == "pathwise_equal_total":
            stressed = pathwise_equal_total_stress(
                scenarios, candidate, alpha_bar, float(stress_value)
            )
        else:
            raise ValueError(stress_mode)
        stressed_values = propagation_metric_batch(
            network,
            stressed,
            horizon,
            n_time,
            kernel,
            source_weights=source_weights,
            weight_scale=weight_scale,
        )
        realised_increment = (stressed - scenarios).sum(axis=1)
        rows.append(
            {
                "case": case_name,
                "candidate_id": candidate.candidate_id,
                "form": candidate.scale,
                "size": candidate.size,
                "kernel": kernel,
                "stress_mode": stress_mode,
                "stress_value": float(stress_value),
                "weight_scale": float(weight_scale),
                "baseline_performance": float(np.mean(baseline)),
                "baseline_reliability": baseline_reliability,
                "stressed_performance": float(np.mean(stressed_values)),
                "stressed_reliability": float(np.mean(stressed_values >= threshold)),
                "direct_empirical_importance": baseline_reliability
                - float(np.mean(stressed_values >= threshold)),
                "threshold": threshold,
                "total_increment_min": float(realised_increment.min()),
                "total_increment_max": float(realised_increment.max()),
            }
        )
        if index % 10 == 0 or index == len(candidates):
            pd.DataFrame(rows).sort_values("candidate_id").to_csv(
                checkpoint, index=False
            )
            logging.info("%s completed %d of %d", stem, index, len(candidates))
    frame = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    metadata = {
        "zero_performance": float(zero),
        "baseline_mean_performance": float(np.mean(baseline)),
        "baseline_reliability": baseline_reliability,
        "threshold": threshold,
        "candidate_count": int(len(candidates)),
    }
    return frame, metadata


def comparison_rows(
    frame: pd.DataFrame,
    reference_kernel: str,
    comparison_kernel: str,
    case_name: str,
) -> list[dict[str, object]]:
    reference = frame[frame["kernel"] == reference_kernel][
        ["candidate_id", "form", "direct_empirical_importance"]
    ].rename(columns={"direct_empirical_importance": "reference_importance"})
    comparison = frame[frame["kernel"] == comparison_kernel][
        ["candidate_id", "form", "direct_empirical_importance"]
    ].rename(columns={"direct_empirical_importance": "comparison_importance"})
    merged = reference.merge(comparison, on=["candidate_id", "form"], validate="one_to_one")
    rows: list[dict[str, object]] = []
    for form, group in merged.groupby("form", sort=True):
        tolerance = 1.0e-12
        difference = group["reference_importance"] - group["comparison_importance"]
        rows.append(
            {
                "case": case_name,
                "form": form,
                "reference_kernel": reference_kernel,
                "comparison_kernel": comparison_kernel,
                "candidate_count": int(len(group)),
                "spearman": safe_spearman(
                    group["reference_importance"], group["comparison_importance"]
                ),
                "top5_overlap": top_overlap(
                    group,
                    "reference_importance",
                    "comparison_importance",
                    5,
                ),
                "sign_agreement": float(
                    np.mean(
                        sign_class(group["reference_importance"], tolerance)
                        == sign_class(group["comparison_importance"], tolerance)
                    )
                ),
                "reference_greater_count": int(np.sum(difference > 1.0e-12)),
                "comparison_greater_count": int(np.sum(difference < -1.0e-12)),
                "equal_count": int(np.sum(np.abs(difference) <= 1.0e-12)),
                "median_reference_minus_comparison": float(np.median(difference)),
            }
        )
    return rows


def run_e1_kernel_comparison() -> None:
    logging.info("E1: starting kernel and Hamiltonian comparison")
    grid_archive = np.load(R1_RESULTS / "grid_experiment_arrays.npz")
    grid = grid_network(5, 5, 20260716)
    grid_cases = grid_candidates(grid, 5, 5)
    grid_scenarios = grid_archive["scenarios"]
    grid_upper = grid_archive["alpha_bar"]

    sioux, _, sioux_sources, _ = parse_sioux_falls()
    sioux_cases = sioux_candidates(sioux)
    sioux_archive = np.load(R1_EXTENDED / "sioux_scenarios.npz")
    sioux_scenarios = sioux_archive["scenarios"]
    sioux_upper = sioux_archive["alpha_bar"]

    all_frames: list[pd.DataFrame] = []
    metadata: dict[str, object] = {}
    for kernel in KERNELS:
        frame, details = direct_candidate_table(
            "grid",
            grid,
            grid_cases,
            grid_scenarios,
            grid_upper,
            GRID_HORIZON,
            GRID_GAMMA,
            48,
            kernel,
            None,
            "affine",
            GRID_STRESS,
        )
        all_frames.append(frame)
        metadata[f"grid_{kernel}"] = details
        frame, details = direct_candidate_table(
            "sioux_falls",
            sioux,
            sioux_cases,
            sioux_scenarios,
            sioux_upper,
            SIOUX_HORIZON,
            SIOUX_GAMMA,
            32,
            kernel,
            sioux_sources,
            "affine",
            SIOUX_STRESS,
        )
        all_frames.append(frame)
        metadata[f"sioux_falls_{kernel}"] = details

    for name, network in motif_networks().items():
        candidates = motif_candidates(network)
        upper = np.full(network.m, 0.12, dtype=float)
        scenarios, _ = exact_binary_scenarios(network.m, amplitude=0.03)
        for kernel in KERNELS:
            frame, details = direct_candidate_table(
                name,
                network,
                candidates,
                scenarios,
                upper,
                GRID_HORIZON,
                0.98,
                64,
                kernel,
                None,
                "affine",
                0.50,
            )
            all_frames.append(frame)
            metadata[f"{name}_{kernel}"] = details

    results = pd.concat(all_frames, ignore_index=True)
    results.to_csv(RESULTS / "kernel_candidate_importance.csv", index=False)
    comparisons: list[dict[str, object]] = []
    for case_name in sorted(results["case"].unique()):
        case_frame = results[results["case"] == case_name]
        comparisons.extend(
            comparison_rows(
                case_frame,
                "adjacency_quantum",
                "laplacian_quantum",
                case_name,
            )
        )
        comparisons.extend(
            comparison_rows(
                case_frame,
                "laplacian_quantum",
                "classical_diffusion",
                case_name,
            )
        )
        comparisons.extend(
            comparison_rows(
                case_frame,
                "adjacency_quantum",
                "classical_diffusion",
                case_name,
            )
        )
    pd.DataFrame(comparisons).to_csv(
        RESULTS / "kernel_rank_comparisons.csv", index=False
    )

    convergence_rows: list[dict[str, object]] = []
    for case_name, network, scenarios, horizon, sources, low, high in (
        ("grid", grid, grid_scenarios[:8], GRID_HORIZON, None, 48, 64),
        (
            "sioux_falls",
            sioux,
            sioux_scenarios[:8],
            SIOUX_HORIZON,
            sioux_sources,
            32,
            48,
        ),
    ):
        for kernel in KERNELS:
            first = propagation_metric_batch(
                network, scenarios, horizon, low, kernel, source_weights=sources
            )
            second = propagation_metric_batch(
                network, scenarios, horizon, high, kernel, source_weights=sources
            )
            convergence_rows.append(
                {
                    "case": case_name,
                    "kernel": kernel,
                    "lower_order": low,
                    "higher_order": high,
                    "state_count": len(scenarios),
                    "maximum_absolute_difference": float(np.max(np.abs(first - second))),
                }
            )
    pd.DataFrame(convergence_rows).to_csv(
        RESULTS / "kernel_quadrature_audit.csv", index=False
    )
    metadata["candidate_rows"] = int(len(results))
    metadata["comparison_rows"] = int(len(comparisons))
    write_json(RESULTS / "kernel_comparison_metadata.json", metadata)
    membership_checks: dict[str, bool] = {}
    for case_name, case_frame in results.groupby("case"):
        memberships = {
            kernel: tuple(
                sorted(case_frame.loc[case_frame["kernel"] == kernel, "candidate_id"])
            )
            for kernel in KERNELS
        }
        membership_checks[str(case_name)] = len(set(memberships.values())) == 1
    convergence_frame = pd.DataFrame(convergence_rows)
    audit = {
        "candidate_membership_equal_across_kernels": membership_checks,
        "all_candidate_memberships_equal": all(membership_checks.values()),
        "all_numeric_values_finite": bool(
            np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "maximum_quadrature_difference": float(
            convergence_frame["maximum_absolute_difference"].max()
        ),
        "grid_candidate_count_per_kernel": int(
            len(results[(results["case"] == "grid") & (results["kernel"] == KERNELS[0])])
        ),
        "sioux_candidate_count_per_kernel": int(
            len(
                results[
                    (results["case"] == "sioux_falls")
                    & (results["kernel"] == KERNELS[0])
                ]
            )
        ),
    }
    audit["pass"] = bool(
        audit["all_candidate_memberships_equal"]
        and audit["all_numeric_values_finite"]
        and audit["maximum_quadrature_difference"] <= 1.0e-8
        and audit["grid_candidate_count_per_kernel"] == 104
        and audit["sioux_candidate_count_per_kernel"] == 797
    )
    write_json(RESULTS / "kernel_comparison_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"E1 audit failed: {audit}")
    logging.info("E1: completed %d candidate-kernel rows", len(results))


def run_e2_scaling_and_horizon() -> None:
    logging.info("E2: starting coupling-time scaling audit")
    grid_archive = np.load(R1_RESULTS / "grid_experiment_arrays.npz")
    grid = grid_network(5, 5, 20260716)
    grid_cases = grid_candidates(grid, 5, 5)
    grid_scenarios = grid_archive["scenarios"]
    grid_upper = grid_archive["alpha_bar"]

    sioux, _, sioux_sources, _ = parse_sioux_falls()
    sioux_all_cases = sioux_candidates(sioux)
    sioux_cases = [candidate for candidate in sioux_all_cases if candidate.scale == "edge"]
    sioux_archive = np.load(R1_EXTENDED / "sioux_scenarios.npz")
    sioux_scenarios = sioux_archive["scenarios"]
    sioux_upper = sioux_archive["alpha_bar"]

    invariance_rows: list[dict[str, object]] = []
    for case_name, network, scenarios, horizon, sources in (
        ("grid", grid, grid_scenarios[:12], GRID_HORIZON, None),
        ("sioux_falls", sioux, sioux_scenarios[:12], SIOUX_HORIZON, sioux_sources),
    ):
        states = [np.zeros(network.m)] + [row for row in scenarios]
        for scale in (0.5, 2.0):
            differences = []
            for alpha in states:
                generator = kernel_matrix(network, alpha, "adjacency_quantum")
                original = metric_from_matrix(
                    generator, horizon, 64, sources, quantum=True
                )
                rescaled = metric_from_matrix(
                    scale * generator, horizon / scale, 64, sources, quantum=True
                )
                differences.append(abs(original - rescaled))
            invariance_rows.append(
                {
                    "case": case_name,
                    "scale": scale,
                    "state_count": len(states),
                    "original_horizon": horizon,
                    "reciprocal_horizon": horizon / scale,
                    "maximum_absolute_invariance_error": float(max(differences)),
                }
            )
    pd.DataFrame(invariance_rows).to_csv(
        RESULTS / "coupling_time_invariance.csv", index=False
    )

    frames: list[pd.DataFrame] = []
    for scale in (0.5, 1.0, 2.0):
        frame, _ = direct_candidate_table(
            "grid_scale",
            grid,
            grid_cases,
            grid_scenarios,
            grid_upper,
            GRID_HORIZON,
            GRID_GAMMA,
            48,
            "adjacency_quantum",
            None,
            "affine",
            GRID_STRESS,
            weight_scale=scale,
        )
        frames.append(frame)
        frame, _ = direct_candidate_table(
            "sioux_edge_scale",
            sioux,
            sioux_cases,
            sioux_scenarios,
            sioux_upper,
            SIOUX_HORIZON,
            SIOUX_GAMMA,
            48,
            "adjacency_quantum",
            sioux_sources,
            "affine",
            SIOUX_STRESS,
            weight_scale=scale,
        )
        frames.append(frame)
    scale_frame = pd.concat(frames, ignore_index=True)
    scale_frame.to_csv(RESULTS / "fixed_horizon_weight_scaling.csv", index=False)
    stability_rows: list[dict[str, object]] = []
    for case_name, case_frame in scale_frame.groupby("case"):
        reference = case_frame[case_frame["weight_scale"] == 1.0][
            ["candidate_id", "form", "direct_empirical_importance"]
        ].rename(columns={"direct_empirical_importance": "reference_importance"})
        for scale in (0.5, 2.0):
            current = case_frame[case_frame["weight_scale"] == scale][
                ["candidate_id", "form", "direct_empirical_importance"]
            ].rename(columns={"direct_empirical_importance": "current_importance"})
            merged = reference.merge(current, on=["candidate_id", "form"])
            for form, group in merged.groupby("form"):
                scenario_count = (
                    len(grid_scenarios)
                    if case_name == "grid_scale"
                    else len(sioux_scenarios)
                )
                current_baseline = float(
                    case_frame.loc[
                        (case_frame["weight_scale"] == scale)
                        & (case_frame["form"] == form),
                        "baseline_reliability",
                    ].iloc[0]
                )
                stability_rows.append(
                    {
                        "case": case_name,
                        "form": form,
                        "scale": scale,
                        "candidate_count": len(group),
                        "spearman": safe_spearman(
                            group["reference_importance"], group["current_importance"]
                        ),
                        "top5_overlap": top_overlap(
                            group, "reference_importance", "current_importance", 5
                        ),
                        "sign_agreement": float(
                            np.mean(
                                sign_class(group["reference_importance"], 1.0e-12)
                                == sign_class(group["current_importance"], 1.0e-12)
                            )
                        ),
                        "baseline_reliability": current_baseline,
                        "baseline_reliability_degenerate": bool(
                            current_baseline <= 1.0 / scenario_count
                            or current_baseline >= 1.0 - 1.0 / scenario_count
                        ),
                    }
                )
    stability_frame = pd.DataFrame(stability_rows)
    stability_frame.to_csv(
        RESULTS / "fixed_horizon_scaling_stability.csv", index=False
    )

    quadrature_rows = []
    for case_name, network, scenarios, sources in (
        ("grid", grid, grid_scenarios[:8], None),
        ("sioux_falls", sioux, sioux_scenarios[:8], sioux_sources),
    ):
        low = propagation_metric_batch(
            network,
            scenarios,
            1.8,
            64,
            "adjacency_quantum",
            source_weights=sources,
            weight_scale=2.0,
        )
        high = propagation_metric_batch(
            network,
            scenarios,
            1.8,
            96,
            "adjacency_quantum",
            source_weights=sources,
            weight_scale=2.0,
        )
        quadrature_rows.append(
            {
                "case": case_name,
                "effective_horizon": 3.6,
                "state_count": len(scenarios),
                "maximum_absolute_order64_order96_difference": float(
                    np.max(np.abs(low - high))
                ),
            }
        )
    quadrature_frame = pd.DataFrame(quadrature_rows)
    quadrature_frame.to_csv(
        RESULTS / "scaling_quadrature_audit.csv", index=False
    )
    invariance_frame = pd.DataFrame(invariance_rows)
    audit = {
        "maximum_scaling_invariance_error": float(
            invariance_frame["maximum_absolute_invariance_error"].max()
        ),
        "maximum_order64_order96_difference": float(
            quadrature_frame["maximum_absolute_order64_order96_difference"].max()
        ),
        "stability_row_count": len(stability_frame),
        "degenerate_stability_row_count": int(
            stability_frame["baseline_reliability_degenerate"].sum()
        ),
        "all_numeric_values_finite": bool(
            np.isfinite(scale_frame.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(
                invariance_frame.select_dtypes(include=[np.number]).to_numpy()
            ).all()
            and np.isfinite(
                quadrature_frame.select_dtypes(include=[np.number]).to_numpy()
            ).all()
        ),
    }
    audit["pass"] = bool(
        audit["maximum_scaling_invariance_error"] <= 1.0e-10
        and audit["maximum_order64_order96_difference"] <= 1.0e-8
        and audit["stability_row_count"] == 8
        and audit["all_numeric_values_finite"]
    )
    write_json(RESULTS / "coupling_time_scaling_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"E2 audit failed: {audit}")
    logging.info("E2: completed scaling and horizon audit")


def size_and_rank_summaries(
    common: pd.DataFrame, controlled: pd.DataFrame, case_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = common[
        ["candidate_id", "form", "size", "direct_empirical_importance"]
    ].rename(columns={"direct_empirical_importance": "common_importance"}).merge(
        controlled[
            ["candidate_id", "form", "size", "direct_empirical_importance"]
        ].rename(columns={"direct_empirical_importance": "pathwise_importance"}),
        on=["candidate_id", "form", "size"],
        validate="one_to_one",
    )
    association_rows = []
    for form, group in merged.groupby("form"):
        association_rows.append(
            {
                "case": case_name,
                "form": form,
                "candidate_count": len(group),
                "unique_sizes": int(group["size"].nunique()),
                "size_common_importance_spearman": safe_spearman(
                    group["size"], group["common_importance"]
                ),
                "size_pathwise_importance_spearman": safe_spearman(
                    group["size"], group["pathwise_importance"]
                ),
                "common_pathwise_rank_spearman": safe_spearman(
                    group["common_importance"], group["pathwise_importance"]
                ),
                "common_pathwise_top5_overlap": top_overlap(
                    group, "common_importance", "pathwise_importance", 5
                ),
            }
        )
    strata = (
        merged.groupby(["form", "size"], as_index=False)
        .agg(
            candidate_count=("candidate_id", "size"),
            common_median=("common_importance", "median"),
            common_minimum=("common_importance", "min"),
            common_maximum=("common_importance", "max"),
            pathwise_median=("pathwise_importance", "median"),
            pathwise_minimum=("pathwise_importance", "min"),
            pathwise_maximum=("pathwise_importance", "max"),
        )
    )
    strata.insert(0, "case", case_name)
    return pd.DataFrame(association_rows), strata


def equal_family_resampling(
    values: dict[str, np.ndarray], draw_count: int = 20_000, seed: int = SEED + 31
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare equally sized families, retaining the actual fractional tie scores."""
    rng = np.random.default_rng(seed)
    forms = ("edge", "corridor", "region")
    equal_count = min(len(values[form]) for form in forms)
    scores = np.zeros((draw_count, len(forms)), dtype=float)
    maximum_rows = []
    tie_draws = 0
    for draw in range(draw_count):
        maxima = {
            form: float(np.max(rng.choice(values[form], equal_count, replace=False)))
            for form in forms
        }
        best = max(maxima.values())
        winners = [i for i, form in enumerate(forms) if abs(maxima[form] - best) <= 1.0e-12]
        tie_draws += int(len(winners) > 1)
        scores[draw, winners] = 1.0 / len(winners)
        maximum_rows.append({"draw": draw, **maxima})
    rows = [
        {
            "form": form,
            "candidate_count_per_draw": equal_count,
            "draw_count": draw_count,
            "fractional_win_probability": float(np.mean(scores[:, i])),
            "monte_carlo_standard_error": float(np.std(scores[:, i], ddof=1) / np.sqrt(draw_count)),
            "tie_draw_rate": tie_draws / draw_count,
        }
        for i, form in enumerate(forms)
    ]
    score_frame = pd.DataFrame(scores, columns=forms)
    score_frame.insert(0, "draw", np.arange(draw_count))
    return pd.DataFrame(rows), pd.DataFrame(maximum_rows), score_frame


def run_e3_pathwise_size_and_family() -> None:
    logging.info("E3: starting pathwise equal-total-stress comparison")
    grid_archive = np.load(R1_RESULTS / "grid_experiment_arrays.npz")
    grid = grid_network(5, 5, 20260716)
    grid_cases = grid_candidates(grid, 5, 5)
    grid_scenarios = grid_archive["scenarios"]
    grid_upper = grid_archive["alpha_bar"]
    grid_total = GRID_STRESS * float(np.mean(grid_upper - grid_scenarios.mean(axis=0)))

    sioux, _, sioux_sources, _ = parse_sioux_falls()
    sioux_cases = sioux_candidates(sioux)
    sioux_archive = np.load(R1_EXTENDED / "sioux_scenarios.npz")
    sioux_scenarios = sioux_archive["scenarios"]
    sioux_upper = sioux_archive["alpha_bar"]
    sioux_total = SIOUX_STRESS * float(
        np.mean(sioux_upper - sioux_scenarios.mean(axis=0))
    )

    grid_pathwise, grid_meta = direct_candidate_table(
        "grid",
        grid,
        grid_cases,
        grid_scenarios,
        grid_upper,
        GRID_HORIZON,
        GRID_GAMMA,
        48,
        "adjacency_quantum",
        None,
        "pathwise_equal_total",
        grid_total,
    )
    sioux_pathwise, sioux_meta = direct_candidate_table(
        "sioux_falls",
        sioux,
        sioux_cases,
        sioux_scenarios,
        sioux_upper,
        SIOUX_HORIZON,
        SIOUX_GAMMA,
        32,
        "adjacency_quantum",
        sioux_sources,
        "pathwise_equal_total",
        sioux_total,
    )
    pathwise = pd.concat([grid_pathwise, sioux_pathwise], ignore_index=True)
    pathwise.to_csv(RESULTS / "pathwise_equal_total_importance.csv", index=False)

    common = pd.read_csv(RESULTS / "kernel_candidate_importance.csv")
    common = common[
        (common["kernel"] == "adjacency_quantum")
        & (common["stress_mode"] == "affine")
    ]
    association_frames = []
    strata_frames = []
    for case_name, controlled in (
        ("grid", grid_pathwise),
        ("sioux_falls", sioux_pathwise),
    ):
        associations, strata = size_and_rank_summaries(
            common[common["case"] == case_name], controlled, case_name
        )
        association_frames.append(associations)
        strata_frames.append(strata)
    pd.concat(association_frames, ignore_index=True).to_csv(
        RESULTS / "candidate_size_rank_summary.csv", index=False
    )
    pd.concat(strata_frames, ignore_index=True).to_csv(
        RESULTS / "candidate_size_strata.csv", index=False
    )

    distribution = (
        sioux_pathwise.groupby("form", as_index=False)["direct_empirical_importance"]
        .agg(
            candidate_count="size",
            minimum="min",
            q10=lambda x: x.quantile(0.10),
            median="median",
            q90=lambda x: x.quantile(0.90),
            q95=lambda x: x.quantile(0.95),
            maximum="max",
        )
    )
    distribution.to_csv(RESULTS / "sioux_form_distributions.csv", index=False)

    forms = ("edge", "corridor", "region")
    values = {
        form: sioux_pathwise.loc[
            sioux_pathwise["form"] == form, "direct_empirical_importance"
        ].to_numpy(float)
        for form in forms
    }
    draw_count = 20_000
    equal_count = min(len(values[form]) for form in forms)
    resampling_frame, maxima_frame, score_frame = equal_family_resampling(values, draw_count)
    maxima_frame.iloc[:2000].to_csv(
        RESULTS / "sioux_equal_count_maximum_sample.csv", index=False
    )
    resampling_frame.to_csv(
        RESULTS / "sioux_equal_family_count_resampling.csv", index=False
    )
    score_frame.to_csv(RESULTS / "sioux_equal_count_fractional_scores.csv", index=False)
    audit = {
        "grid_total_increment": grid_total,
        "grid_minimum_available_single_edge_headroom": float(
            np.min(grid_upper - grid_scenarios)
        ),
        "grid_maximum_recorded_increment_spread": float(
            np.max(grid_pathwise["total_increment_max"] - grid_pathwise["total_increment_min"])
        ),
        "sioux_total_increment": sioux_total,
        "sioux_minimum_available_single_edge_headroom": float(
            np.min(sioux_upper - sioux_scenarios)
        ),
        "sioux_maximum_recorded_increment_spread": float(
            np.max(
                sioux_pathwise["total_increment_max"]
                - sioux_pathwise["total_increment_min"]
            )
        ),
        "grid_metadata": grid_meta,
        "sioux_metadata": sioux_meta,
        "equal_family_resampling_draws": draw_count,
        "equal_family_candidate_count": equal_count,
        "maximum_total_increment_target_error": float(
            max(
                np.max(np.abs(grid_pathwise["total_increment_min"] - grid_total)),
                np.max(np.abs(grid_pathwise["total_increment_max"] - grid_total)),
                np.max(np.abs(sioux_pathwise["total_increment_min"] - sioux_total)),
                np.max(np.abs(sioux_pathwise["total_increment_max"] - sioux_total)),
            )
        ),
        "fractional_win_probability_sum": float(
            resampling_frame["fractional_win_probability"].sum()
        ),
        "all_candidates_evaluated": bool(
            len(grid_pathwise) == 104 and len(sioux_pathwise) == 797
        ),
    }
    audit["pass"] = bool(
        audit["grid_total_increment"]
        <= audit["grid_minimum_available_single_edge_headroom"] + 1.0e-12
        and audit["sioux_total_increment"]
        <= audit["sioux_minimum_available_single_edge_headroom"] + 1.0e-12
        and audit["maximum_total_increment_target_error"] <= 1.0e-12
        and abs(audit["fractional_win_probability_sum"] - 1.0) <= 1.0e-12
        and audit["equal_family_candidate_count"] == 20
        and audit["all_candidates_evaluated"]
    )
    write_json(RESULTS / "pathwise_equal_total_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"E3 audit failed: {audit}")
    logging.info("E3: completed pathwise and family-count analyses")


def mean_cross_kendall(
    scenarios: np.ndarray, selected: Sequence[int]
) -> float:
    values = np.asarray(scenarios, dtype=float)
    selected_ids = set(int(value) for value in selected)
    unselected = [index for index in range(values.shape[1]) if index not in selected_ids]
    pairs = [(first, second) for first in sorted(selected_ids) for second in unselected]
    if not pairs:
        return float("nan")
    coefficients = [
        float(kendalltau(values[:, first], values[:, second]).statistic)
        for first, second in pairs
    ]
    return float(np.nanmean(coefficients))


def run_e4_probability_counterfactuals() -> None:
    logging.info("E4: starting direct probability counterfactuals")
    archive = np.load(R1_RESULTS / "grid_experiment_arrays.npz")
    network = grid_network(5, 5, 20260716)
    candidates = {candidate.candidate_id: candidate for candidate in grid_candidates(network, 5, 5)}
    scenarios = archive["scenarios"]
    alpha_bar = archive["alpha_bar"]
    d0 = propagation_metric(
        network,
        np.zeros(network.m),
        GRID_HORIZON,
        48,
        "adjacency_quantum",
    )
    threshold = GRID_GAMMA * d0
    baseline_values = propagation_metric_batch(
        network,
        scenarios,
        GRID_HORIZON,
        48,
        "adjacency_quantum",
    )
    baseline_reliability = float(np.mean(baseline_values >= threshold))
    baseline_mean = scenarios.mean(axis=0)
    rng = np.random.default_rng(SEED + 44)
    rows = []
    saved_arrays: dict[str, np.ndarray] = {"baseline": scenarios}
    for candidate_id in ("E40", "R03"):
        candidate = candidates[candidate_id]
        ids = np.asarray(candidate.edge_ids, dtype=int)
        scale = np.ones(network.m, dtype=float)
        scale[ids] = 1.0 - GRID_STRESS
        stressed_mean = baseline_mean.copy()
        stressed_mean[ids] = (
            (1.0 - GRID_STRESS) * baseline_mean[ids]
            + GRID_STRESS * alpha_bar[ids]
        )
        mean_only = scenarios.copy()
        mean_only[:, ids] += stressed_mean[ids] - baseline_mean[ids]
        fluctuation_only = baseline_mean[None, :] + (
            scenarios - baseline_mean[None, :]
        ) * scale[None, :]
        complete = stress_scenarios(
            scenarios, candidate.edge_ids, GRID_STRESS, alpha_bar
        )
        permutation = rng.permutation(len(scenarios))
        dependence_only = scenarios.copy()
        dependence_only[:, ids] = scenarios[permutation][:, ids]
        laws = {
            "baseline": scenarios,
            "mean_only": mean_only,
            "fluctuation_only": fluctuation_only,
            "dependence_only": dependence_only,
            "complete_affine_stress": complete,
        }
        for law_name, law in laws.items():
            if np.min(law) < -1.0e-12 or np.max(law - alpha_bar[None, :]) > 1.0e-12:
                raise RuntimeError(f"Counterfactual support violation: {candidate_id} {law_name}")
            values = (
                baseline_values
                if law_name == "baseline"
                else propagation_metric_batch(
                    network,
                    law,
                    GRID_HORIZON,
                    48,
                    "adjacency_quantum",
                )
            )
            covariance = np.cov(law, rowvar=False, bias=True)
            selected_variance = float(np.trace(covariance[np.ix_(ids, ids)]))
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "form": candidate.scale,
                    "size": candidate.size,
                    "counterfactual": law_name,
                    "reliability": float(np.mean(values >= threshold)),
                    "importance_from_baseline": baseline_reliability
                    - float(np.mean(values >= threshold)),
                    "mean_performance": float(np.mean(values)),
                    "selected_mean_sum": float(law[:, ids].mean(axis=0).sum()),
                    "selected_variance_trace": selected_variance,
                    "selected_unselected_mean_kendall": mean_cross_kendall(law, ids),
                    "minimum_disturbance": float(np.min(law)),
                    "maximum_headroom_violation": float(
                        np.max(law - alpha_bar[None, :])
                    ),
                }
            )
            saved_arrays[f"{candidate_id}_{law_name}"] = law
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "negative_importance_probability_counterfactuals.csv", index=False)
    np.savez_compressed(
        CHECKPOINTS / "probability_counterfactual_scenarios.npz", **saved_arrays
    )
    validation_values = propagation_metric_batch(
        network,
        np.vstack([saved_arrays["E40_mean_only"][:4], saved_arrays["R03_complete_affine_stress"][:4]]),
        GRID_HORIZON,
        64,
        "adjacency_quantum",
    )
    reference_values = propagation_metric_batch(
        network,
        np.vstack([saved_arrays["E40_mean_only"][:4], saved_arrays["R03_complete_affine_stress"][:4]]),
        GRID_HORIZON,
        48,
        "adjacency_quantum",
    )
    audit = {
            "baseline_reliability": baseline_reliability,
            "threshold": threshold,
            "row_count": len(frame),
            "all_values_finite": bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all()),
            "maximum_order48_order64_difference": float(
                np.max(np.abs(reference_values - validation_values))
            ),
            "mean_only_support_verified": True,
            "fluctuation_only_mean_tolerance": float(
                max(
                    abs(
                        frame.loc[
                            (frame["candidate_id"] == candidate_id)
                            & (frame["counterfactual"] == "fluctuation_only"),
                            "selected_mean_sum",
                        ].iloc[0]
                        - frame.loc[
                            (frame["candidate_id"] == candidate_id)
                            & (frame["counterfactual"] == "baseline"),
                            "selected_mean_sum",
                        ].iloc[0]
                    )
                    for candidate_id in ("E40", "R03")
                )
            ),
        }
    audit["pass"] = bool(
        audit["row_count"] == 10
        and audit["all_values_finite"]
        and audit["maximum_order48_order64_difference"] <= 1.0e-8
        and audit["mean_only_support_verified"]
        and audit["fluctuation_only_mean_tolerance"] <= 1.0e-12
    )
    write_json(RESULTS / "probability_counterfactual_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"E4 audit failed: {audit}")
    logging.info("E4: completed direct probability counterfactuals")


def rank_matched_beta_marginals(
    uniforms: np.ndarray,
    alpha_bar: np.ndarray,
    beta_a: float = 2.0,
    beta_b: float = 8.0,
) -> np.ndarray:
    """Give every dependence family exactly the same finite marginal support."""

    values = np.asarray(uniforms, dtype=float)
    probabilities = (np.arange(len(values), dtype=float) + 0.5) / len(values)
    target = beta.ppf(probabilities, beta_a, beta_b)
    ranks = np.argsort(np.argsort(values, axis=0), axis=0)
    return target[ranks] * np.asarray(alpha_bar, dtype=float)[None, :]


def all_pair_dependence(values: np.ndarray) -> dict[str, object]:
    """Summarise every distinct column pair, rather than a selected subset."""

    array = np.asarray(values, dtype=float)
    kendall_values: list[float] = []
    pearson_values: list[float] = []
    for first in range(array.shape[1]):
        for second in range(first + 1, array.shape[1]):
            tau = kendalltau(array[:, first], array[:, second]).statistic
            kendall_values.append(float(tau))
            pearson_values.append(
                float(np.corrcoef(array[:, first], array[:, second])[0, 1])
            )
    kendall_array = np.asarray(kendall_values, dtype=float)
    pearson_array = np.asarray(pearson_values, dtype=float)
    return {
        "pair_count": int(len(kendall_array)),
        "kendall_mean": float(np.mean(kendall_array)),
        "kendall_standard_deviation": float(np.std(kendall_array, ddof=0)),
        "kendall_minimum": float(np.min(kendall_array)),
        "kendall_q05": float(np.quantile(kendall_array, 0.05)),
        "kendall_median": float(np.median(kendall_array)),
        "kendall_q95": float(np.quantile(kendall_array, 0.95)),
        "kendall_maximum": float(np.max(kendall_array)),
        "pearson_mean": float(np.mean(pearson_array)),
        "pearson_standard_deviation": float(np.std(pearson_array, ddof=0)),
        "pearson_minimum": float(np.min(pearson_array)),
        "pearson_maximum": float(np.max(pearson_array)),
    }


def dependence_uniforms(
    family: str,
    parameter: float,
    bases: dict[str, np.ndarray],
    edge_distances: np.ndarray,
) -> np.ndarray:
    """Generate a deterministic copula sample from fixed base variates."""

    if family == "independent":
        latent = bases["independent"]
        return np.clip(norm.cdf(latent), 1.0e-12, 1.0 - 1.0e-12)
    if family in {"gaussian", "student_t4"}:
        rho = float(parameter)
        latent = (
            np.sqrt(rho) * bases[f"{family}_common"]
            + np.sqrt(1.0 - rho) * bases[f"{family}_independent"]
        )
        if family == "gaussian":
            uniforms = norm.cdf(latent)
        else:
            uniforms = student_t.cdf(latent / bases["student_t4_scale"], df=4)
        return np.clip(uniforms, 1.0e-12, 1.0 - 1.0e-12)
    if family == "clayton":
        theta = float(parameter)
        frailty = gamma_distribution.ppf(
            bases["clayton_frailty_uniform"], a=1.0 / theta, scale=1.0
        )
        uniforms = (
            1.0 + bases["clayton_exponential"] / frailty
        ) ** (-1.0 / theta)
        return np.clip(uniforms, 1.0e-12, 1.0 - 1.0e-12)
    if family == "spatial_gaussian":
        correlation = np.exp(-edge_distances / float(parameter))
        np.fill_diagonal(correlation, 1.0)
        cholesky = np.linalg.cholesky(
            correlation + 1.0e-12 * np.eye(correlation.shape[0])
        )
        latent = bases["spatial_gaussian"] @ cholesky.T
        return np.clip(norm.cdf(latent), 1.0e-12, 1.0 - 1.0e-12)
    raise ValueError(family)


def calibrate_dependence_parameter(
    family: str,
    bases: dict[str, np.ndarray],
    edge_distances: np.ndarray,
    target: float,
) -> tuple[float, float, int]:
    """Calibrate mean empirical all-pair Kendall association by bisection."""

    if family == "gaussian" or family == "student_t4":
        lower, upper = 0.0, 0.995
    elif family == "clayton":
        lower, upper = 0.01, 30.0
    elif family == "spatial_gaussian":
        positive = edge_distances[edge_distances > 0.0]
        lower = max(float(np.min(positive)) * 1.0e-3, 1.0e-6)
        upper = float(np.max(positive)) * 1.0e3
    else:
        raise ValueError(family)

    evaluations = 0
    best_parameter = lower
    best_tau = float("nan")
    best_error = float("inf")
    for _ in range(36):
        if family == "spatial_gaussian":
            parameter = math.sqrt(lower * upper)
        else:
            parameter = 0.5 * (lower + upper)
        uniforms = dependence_uniforms(
            family, parameter, bases, edge_distances
        )
        tau = float(all_pair_dependence(uniforms)["kendall_mean"])
        evaluations += 1
        error = abs(tau - target)
        if error < best_error:
            best_parameter, best_tau, best_error = parameter, tau, error
        if tau < target:
            lower = parameter
        else:
            upper = parameter
    return float(best_parameter), float(best_tau), evaluations


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest().upper()


def run_e5_matched_dependence() -> None:
    logging.info("E5: starting matched-strength dependence experiment")
    network = grid_network(5, 5, 20260716)
    candidates = grid_candidates(network, 5, 5)
    scenario_count = 256
    alpha_bar = np.full(network.m, 0.22, dtype=float)
    rng = np.random.default_rng(SEED + 500)
    bases = {
        "independent": rng.standard_normal((scenario_count, network.m)),
        "gaussian_common": rng.standard_normal((scenario_count, 1)),
        "gaussian_independent": rng.standard_normal((scenario_count, network.m)),
        "student_t4_common": rng.standard_normal((scenario_count, 1)),
        "student_t4_independent": rng.standard_normal((scenario_count, network.m)),
        "student_t4_scale": np.sqrt(
            rng.chisquare(4, size=(scenario_count, 1)) / 4.0
        ),
        "clayton_frailty_uniform": rng.uniform(
            1.0e-12, 1.0 - 1.0e-12, size=(scenario_count, 1)
        ),
        "clayton_exponential": rng.exponential(
            scale=1.0, size=(scenario_count, network.m)
        ),
        "spatial_gaussian": rng.standard_normal((scenario_count, network.m)),
    }
    midpoints = np.asarray(
        [
            0.5
            * (
                np.asarray(network.positions[int(u)], dtype=float)
                + np.asarray(network.positions[int(v)], dtype=float)
            )
            for u, v in network.edges
        ]
    )
    edge_distances = cdist(midpoints, midpoints)
    families = (
        "independent",
        "gaussian",
        "student_t4",
        "clayton",
        "spatial_gaussian",
    )
    scenarios_by_family: dict[str, np.ndarray] = {}
    calibration_rows: list[dict[str, object]] = []
    rerun_hash_match: dict[str, bool] = {}
    for family in families:
        if family == "independent":
            parameter = 0.0
            calibration_tau = float("nan")
            evaluations = 1
        else:
            parameter, calibration_tau, evaluations = calibrate_dependence_parameter(
                family, bases, edge_distances, target=0.35
            )
        uniforms = dependence_uniforms(
            family, parameter, bases, edge_distances
        )
        scenarios = rank_matched_beta_marginals(uniforms, alpha_bar)
        repeat = rank_matched_beta_marginals(
            dependence_uniforms(family, parameter, bases, edge_distances),
            alpha_bar,
        )
        rerun_hash_match[family] = array_sha256(scenarios) == array_sha256(repeat)
        scenarios_by_family[family] = scenarios
        dependence = all_pair_dependence(scenarios)
        calibration_rows.append(
            {
                "family": family,
                "parameter": parameter,
                "target_mean_kendall": 0.0 if family == "independent" else 0.35,
                "calibration_mean_kendall": calibration_tau,
                "calibration_evaluations": evaluations,
                **dependence,
                "scenario_sha256": array_sha256(scenarios),
            }
        )
        logging.info(
            "E5 calibrated %s at parameter %.8g with mean tau %.6f",
            family,
            parameter,
            dependence["kendall_mean"],
        )

    np.savez_compressed(
        CHECKPOINTS / "matched_dependence_scenarios.npz",
        **scenarios_by_family,
        alpha_bar=alpha_bar,
    )
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(
        RESULTS / "matched_dependence_calibration.csv", index=False
    )

    candidate_frames: list[pd.DataFrame] = []
    metadata: dict[str, object] = {}
    for family in families:
        frame, details = direct_candidate_table(
            f"dependence_{family}",
            network,
            candidates,
            scenarios_by_family[family],
            alpha_bar,
            GRID_HORIZON,
            GRID_GAMMA,
            48,
            "adjacency_quantum",
            None,
            "affine",
            GRID_STRESS,
        )
        frame["family"] = family
        candidate_frames.append(frame)
        metadata[family] = details
    candidate_frame = pd.concat(candidate_frames, ignore_index=True)
    candidate_frame.to_csv(
        RESULTS / "matched_dependence_candidate_importance.csv", index=False
    )

    reference = candidate_frame[candidate_frame["family"] == "gaussian"][
        ["candidate_id", "form", "direct_empirical_importance"]
    ].rename(columns={"direct_empirical_importance": "reference_importance"})
    stability_rows: list[dict[str, object]] = []
    tolerance = 0.5 / scenario_count
    for (family, form), group in candidate_frame.groupby(
        ["family", "form"], sort=True
    ):
        merged = group.merge(reference, on=["candidate_id", "form"], validate="one_to_one")
        stability_rows.append(
            {
                "family": family,
                "reference_family": "gaussian",
                "form": form,
                "candidate_count": len(merged),
                "rank_spearman": safe_spearman(
                    merged["reference_importance"],
                    merged["direct_empirical_importance"],
                ),
                "top5_overlap": top_overlap(
                    merged,
                    "reference_importance",
                    "direct_empirical_importance",
                    5,
                ),
                "sign_agreement": float(
                    np.mean(
                        sign_class(merged["reference_importance"], tolerance)
                        == sign_class(
                            merged["direct_empirical_importance"], tolerance
                        )
                    )
                ),
            }
        )
    pd.DataFrame(stability_rows).to_csv(
        RESULTS / "matched_dependence_stability.csv", index=False
    )

    sorted_reference = np.sort(scenarios_by_family[families[0]], axis=0)
    marginal_error = max(
        float(
            np.max(
                np.abs(np.sort(scenarios_by_family[family], axis=0) - sorted_reference)
            )
        )
        for family in families[1:]
    )
    dependent_error = float(
        np.max(
            np.abs(
                calibration_frame.loc[
                    calibration_frame["family"] != "independent", "kendall_mean"
                ].to_numpy(dtype=float)
                - 0.35
            )
        )
    )
    audit = {
        "family_count": len(families),
        "candidate_count_per_family": len(candidates),
        "all_pair_count_per_family": int(network.m * (network.m - 1) / 2),
        "maximum_sorted_marginal_difference": marginal_error,
        "maximum_dependent_mean_kendall_error": dependent_error,
        "calibration_tolerance": 0.005,
        "rerun_hash_match": rerun_hash_match,
        "all_values_finite": bool(
            np.isfinite(
                candidate_frame.select_dtypes(include=[np.number]).to_numpy()
            ).all()
        ),
        "metadata": metadata,
    }
    audit["pass"] = bool(
        marginal_error <= 1.0e-15
        and dependent_error <= 0.005
        and all(rerun_hash_match.values())
        and audit["all_values_finite"]
    )
    write_json(RESULTS / "matched_dependence_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"E5 audit failed: {audit}")
    logging.info("E5: completed matched-strength dependence experiment")


def interval_status(lower: float, upper: float) -> int:
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def certificate_quantity_rows(
    case_name: str,
    condition_id: str,
    candidates: Sequence[Candidate],
    exact_importance: np.ndarray,
    approximate_baseline_losses: np.ndarray,
    approximate_stressed_losses: np.ndarray,
    theta: float,
    epsilon: float,
    allowance_type: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return quantity-level certificate summaries and candidate intervals."""

    exact_values = np.asarray(exact_importance, dtype=float)
    approximate_stressed = np.asarray(approximate_stressed_losses, dtype=float)
    baseline_interval = moment_interval(
        approximate_baseline_losses, float(theta), float(epsilon)
    )
    lower = np.empty(len(candidates), dtype=float)
    upper = np.empty(len(candidates), dtype=float)
    for index in range(len(candidates)):
        stressed_interval = moment_interval(
            approximate_stressed[index], float(theta), float(epsilon)
        )
        lower[index] = baseline_interval[0] - stressed_interval[1]
        upper[index] = baseline_interval[1] - stressed_interval[0]
    statuses = np.asarray(
        [interval_status(lo, hi) for lo, hi in zip(lower, upper, strict=True)],
        dtype=int,
    )
    truth = sign_class(exact_values, 1.0e-12)
    coverage = (lower - 1.0e-12 <= exact_values) & (
        exact_values <= upper + 1.0e-12
    )
    detail_rows = [
        {
            "case": case_name,
            "condition_id": condition_id,
            "allowance_type": allowance_type,
            "candidate_id": candidate.candidate_id,
            "form": candidate.scale,
            "size": candidate.size,
            "exact_importance": float(exact_values[index]),
            "lower": float(lower[index]),
            "upper": float(upper[index]),
            "width": float(upper[index] - lower[index]),
            "sign_status": int(statuses[index]),
            "true_sign": int(truth[index]),
            "covered": bool(coverage[index]),
        }
        for index, candidate in enumerate(candidates)
    ]

    forms = np.asarray([candidate.scale for candidate in candidates], dtype=object)
    scopes: list[tuple[str, np.ndarray]] = [
        ("all", np.arange(len(candidates), dtype=int))
    ]
    scopes.extend(
        (str(form), np.flatnonzero(forms == form))
        for form in sorted(set(forms))
    )
    summary_rows: list[dict[str, object]] = []
    for scope, ids in scopes:
        values = exact_values[ids]
        lows = lower[ids]
        highs = upper[ids]
        local_status = statuses[ids]
        local_truth = truth[ids]
        local_coverage = coverage[ids]
        widths = highs - lows
        strict_pair_count = 0
        correct_pair_count = 0
        false_pair_count = 0
        for first in range(len(ids)):
            for second in range(first + 1, len(ids)):
                difference = values[first] - values[second]
                if abs(difference) <= 1.0e-12:
                    continue
                strict_pair_count += 1
                higher, lower_id = (
                    (first, second) if difference > 0.0 else (second, first)
                )
                if lows[higher] > highs[lower_id]:
                    correct_pair_count += 1
                if lows[lower_id] > highs[higher]:
                    false_pair_count += 1

        top_value = float(np.max(values))
        true_top = np.flatnonzero(np.abs(values - top_value) <= 1.0e-12)
        certified_top = []
        for first in range(len(ids)):
            competitors = np.delete(highs, first)
            if len(competitors) == 0 or lows[first] > float(np.max(competitors)):
                certified_top.append(first)
        top1_resolved = len(certified_top) == 1
        false_top1 = bool(
            top1_resolved
            and not (
                len(true_top) == 1 and certified_top[0] == int(true_top[0])
            )
        )

        top5_applicable = len(ids) > 5
        top5_resolved = False
        top5_order_resolved = False
        false_top5 = False
        exact_top5: set[int] = set()
        certified_top5: set[int] = set()
        if top5_applicable:
            order = np.argsort(-values, kind="mergesort")
            exact_top5 = set(int(value) for value in order[:5])
            fifth_value = float(values[order[4]])
            sixth_value = float(values[order[5]])
            boundary_unique = fifth_value > sixth_value + 1.0e-12
            if boundary_unique:
                smallest_selected_lower = min(lows[list(exact_top5)])
                largest_other_upper = max(
                    highs[index]
                    for index in range(len(ids))
                    if index not in exact_top5
                )
                top5_resolved = bool(
                    smallest_selected_lower > largest_other_upper
                )
                if top5_resolved:
                    certified_top5 = set(exact_top5)
                    top5_order_resolved = bool(
                        all(lows[order[k]] > highs[order[k + 1]] for k in range(4))
                    )
            false_top5 = bool(top5_resolved and certified_top5 != exact_top5)

        summary_rows.append(
            {
                "case": case_name,
                "condition_id": condition_id,
                "allowance_type": allowance_type,
                "scope": scope,
                "theta": float(theta),
                "epsilon": float(epsilon),
                "epsilon_to_theta": float(epsilon / max(abs(theta), 1.0e-300)),
                "candidate_count": int(len(ids)),
                "sign_resolved_count": int(np.sum(local_status != 0)),
                "sign_resolved_rate": float(np.mean(local_status != 0)),
                "false_sign_count": int(
                    np.sum((local_status != 0) & (local_status != local_truth))
                ),
                "strict_pair_count": int(strict_pair_count),
                "pair_resolved_count": int(correct_pair_count),
                "pair_resolved_rate": float(
                    correct_pair_count / max(strict_pair_count, 1)
                ),
                "false_pair_count": int(false_pair_count),
                "top1_unique_truth": bool(len(true_top) == 1),
                "top1_resolved": bool(top1_resolved),
                "false_top1": false_top1,
                "top5_applicable": bool(top5_applicable),
                "top5_resolved": bool(top5_resolved),
                "top5_order_resolved": bool(top5_order_resolved),
                "false_top5": false_top5,
                "interval_width_mean": float(np.mean(widths)),
                "interval_width_median": float(np.median(widths)),
                "interval_width_maximum": float(np.max(widths)),
                "coverage_count": int(np.sum(local_coverage)),
                "coverage_rate": float(np.mean(local_coverage)),
            }
        )
    return summary_rows, detail_rows


def run_e6_certificate_quantities() -> None:
    logging.info("E6: starting quantity-level certificate audit")
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    physical_condition_count = 0

    motif_cases = ("cycle5", "diamond4")
    alpha_caps = (
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.025,
        0.05,
        0.10,
        0.25,
    )
    motifs = motif_networks()
    for case_name in motif_cases:
        network = motifs[case_name]
        candidates = motif_candidates(network)
        d0, g, q, _ = rank_aware_coefficients(
            network, GRID_HORIZON, n_time=24, n_inner=6
        )
        for cap in alpha_caps:
            alpha_bar = np.full(network.m, cap, dtype=float)
            scenarios, _ = exact_binary_scenarios(
                network.m, amplitude=0.06 * cap
            )
            exact_baseline = d0 - dpc_batch(
                network, scenarios, GRID_HORIZON, n_time=64
            )
            approximate_baseline = np.asarray(quadratic_loss(scenarios, g, q))
            exact_stressed = np.empty(
                (len(candidates), len(scenarios)), dtype=float
            )
            approximate_stressed = np.empty_like(exact_stressed)
            support_epsilon = float(
                np.max(np.abs(exact_baseline - approximate_baseline))
            )
            for index, candidate in enumerate(candidates):
                stressed = stress_scenarios(
                    scenarios, candidate.edge_ids, 0.8, alpha_bar
                )
                exact_stressed[index] = d0 - dpc_batch(
                    network, stressed, GRID_HORIZON, n_time=64
                )
                approximate_stressed[index] = quadratic_loss(stressed, g, q)
                support_epsilon = max(
                    support_epsilon,
                    float(
                        np.max(
                            np.abs(exact_stressed[index] - approximate_stressed[index])
                        )
                    ),
                )
            combined = np.concatenate(
                [exact_baseline.ravel(), exact_stressed.ravel()]
            )
            theta_lower, theta_upper = np.quantile(combined, [0.02, 0.98])
            theta_values = np.linspace(
                max(0.0, float(theta_lower)), min(float(d0), float(theta_upper)), 21
            )
            analytical_epsilon = theoretical_epsilon(
                network, GRID_HORIZON, alpha_bar
            )
            for threshold_index, theta in enumerate(theta_values):
                physical_condition_count += 1
                exact_baseline_reliability = exact_reliability(
                    exact_baseline, float(theta)
                )
                exact_importance = np.asarray(
                    [
                        exact_baseline_reliability
                        - exact_reliability(row, float(theta))
                        for row in exact_stressed
                    ],
                    dtype=float,
                )
                condition_id = (
                    f"{case_name}_cap_{cap:.7f}_threshold_{threshold_index:02d}"
                )
                for allowance_type, epsilon in (
                    ("analytical_global", analytical_epsilon),
                    ("finite_support_diagnostic", support_epsilon),
                ):
                    summaries, details = certificate_quantity_rows(
                        case_name,
                        condition_id,
                        candidates,
                        exact_importance,
                        approximate_baseline,
                        approximate_stressed,
                        float(theta),
                        float(epsilon),
                        allowance_type,
                    )
                    for row in summaries:
                        row.update(
                            {
                                "alpha_cap": cap,
                                "threshold_index": threshold_index,
                                "total_coupling_radius": float(
                                    np.dot(alpha_bar, network.weights)
                                ),
                            }
                        )
                    summary_rows.extend(summaries)
                    detail_rows.extend(details)
            logging.info("E6 completed %s at cap %.4g", case_name, cap)

    grid_archive = np.load(R1_RESULTS / "grid_experiment_arrays.npz")
    grid = grid_network(5, 5, 20260716)
    grid_cases = grid_candidates(grid, 5, 5)
    grid_exact_baseline = grid_archive["exact_baseline_losses"]
    grid_exact_stressed = grid_archive["exact_stressed_losses"]
    grid_theta = float(grid_archive["theta"][0])
    grid_baseline_reliability = exact_reliability(
        grid_exact_baseline, grid_theta
    )
    grid_importance = np.asarray(
        [
            grid_baseline_reliability - exact_reliability(row, grid_theta)
            for row in grid_exact_stressed
        ]
    )
    for allowance_type, epsilon in (
        ("analytical_global", float(grid_archive["global_epsilon"][0])),
        (
            "finite_support_diagnostic",
            float(grid_archive["support_epsilon"][0]),
        ),
    ):
        summaries, details = certificate_quantity_rows(
            "grid5x5",
            "grid_primary",
            grid_cases,
            grid_importance,
            grid_archive["approximate_baseline_losses"],
            grid_archive["approximate_stressed_losses"],
            grid_theta,
            epsilon,
            allowance_type,
        )
        summary_rows.extend(summaries)
        detail_rows.extend(details)

    sioux, _, sioux_sources, _ = parse_sioux_falls()
    sioux_cases = sioux_candidates(sioux)
    sioux_scenarios = np.load(R1_EXTENDED / "sioux_scenarios.npz")
    sioux_coefficients = np.load(
        R1_EXTENDED / "sioux_coefficients_weighted.npz"
    )
    sioux_d0 = float(sioux_coefficients["d0"][0])
    sioux_g = sioux_coefficients["g"]
    sioux_q = sioux_coefficients["q"]
    sioux_values = sioux_scenarios["scenarios"]
    sioux_upper = sioux_scenarios["alpha_bar"]
    sioux_approximate_baseline = np.asarray(
        quadratic_loss(sioux_values, sioux_g, sioux_q)
    )
    sioux_qx = sioux_values @ sioux_q.T
    sioux_approximate_stressed = np.empty(
        (len(sioux_cases), len(sioux_values)), dtype=float
    )
    for index, candidate in enumerate(sioux_cases):
        sioux_approximate_stressed[index] = incremental_quadratic_loss(
            sioux_values,
            candidate.edge_ids,
            SIOUX_STRESS,
            sioux_upper,
            sioux_g,
            sioux_q,
            baseline_losses=sioux_approximate_baseline,
            qx=sioux_qx,
        )
    sioux_direct = pd.read_csv(
        R1_EXTENDED / "sioux_weighted_dpc_importance.csv"
    )
    direct_by_id = sioux_direct.set_index("candidate_id")[
        "direct_empirical_importance"
    ]
    sioux_importance = np.asarray(
        [float(direct_by_id.loc[candidate.candidate_id]) for candidate in sioux_cases]
    )
    sioux_theta = sioux_d0 * (1.0 - SIOUX_GAMMA)
    sioux_epsilon = theoretical_epsilon(
        sioux, SIOUX_HORIZON, sioux_upper
    )
    summaries, details = certificate_quantity_rows(
        "sioux_falls",
        "sioux_primary",
        sioux_cases,
        sioux_importance,
        sioux_approximate_baseline,
        sioux_approximate_stressed,
        sioux_theta,
        sioux_epsilon,
        "analytical_global",
    )
    summary_rows.extend(summaries)
    detail_rows.extend(details)

    summary_frame = pd.DataFrame(summary_rows)
    detail_frame = pd.DataFrame(detail_rows)
    summary_frame.to_csv(
        RESULTS / "certificate_quantity_summary.csv", index=False
    )
    detail_frame.to_csv(
        RESULTS / "certificate_candidate_intervals.csv", index=False
    )
    motif_analytical = summary_frame[
        summary_frame["case"].isin(motif_cases)
        & (summary_frame["allowance_type"] == "analytical_global")
        & (summary_frame["scope"] == "all")
    ]
    resolved_conditions = int(
        np.sum(
            (motif_analytical["sign_resolved_count"] > 0)
            | (motif_analytical["pair_resolved_count"] > 0)
            | motif_analytical["top1_resolved"]
            | motif_analytical["top5_resolved"]
        )
    )
    audit = {
        "motif_physical_condition_count": physical_condition_count,
        "motif_analytical_all_scope_row_count": len(motif_analytical),
        "motif_analytical_resolved_condition_count": resolved_conditions,
        "false_sign_count": int(summary_frame["false_sign_count"].sum()),
        "false_pair_count": int(summary_frame["false_pair_count"].sum()),
        "false_top1_count": int(summary_frame["false_top1"].sum()),
        "false_top5_count": int(summary_frame["false_top5"].sum()),
        "minimum_coverage_rate": float(summary_frame["coverage_rate"].min()),
        "grid_candidate_count": len(grid_cases),
        "sioux_candidate_count": len(sioux_cases),
        "all_required_numeric_values_finite": bool(
            np.isfinite(
                summary_frame[
                    [
                        "theta",
                        "epsilon",
                        "epsilon_to_theta",
                        "candidate_count",
                        "sign_resolved_count",
                        "sign_resolved_rate",
                        "false_sign_count",
                        "strict_pair_count",
                        "pair_resolved_count",
                        "pair_resolved_rate",
                        "false_pair_count",
                        "interval_width_mean",
                        "interval_width_median",
                        "interval_width_maximum",
                        "coverage_count",
                        "coverage_rate",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            and np.isfinite(
                detail_frame[
                    [
                        "size",
                        "exact_importance",
                        "lower",
                        "upper",
                        "width",
                        "sign_status",
                        "true_sign",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "sioux_source_weight_sum": float(np.sum(sioux_sources)),
    }
    audit["pass"] = bool(
        physical_condition_count == 420
        and len(motif_analytical) == 420
        and audit["false_sign_count"] == 0
        and audit["false_pair_count"] == 0
        and audit["false_top1_count"] == 0
        and audit["false_top5_count"] == 0
        and audit["minimum_coverage_rate"] >= 1.0 - 1.0e-12
        and audit["grid_candidate_count"] == 104
        and audit["sioux_candidate_count"] == 797
        and audit["all_required_numeric_values_finite"]
        and abs(audit["sioux_source_weight_sum"] - 1.0) <= 1.0e-12
    )
    write_json(RESULTS / "certificate_quantity_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"E6 audit failed: {audit}")
    logging.info("E6: completed quantity-level certificate audit")


def e7_worker(
    node_count: int,
    seed: int,
    output_path: Path,
) -> None:
    """Run one isolated coefficient construction for memory-aware timing."""

    ensure_directories()
    network = synthetic_topology_network(
        "small_world",
        int(node_count),
        int(seed),
        name=f"r2_scaling_n{node_count}",
    )
    with threadpool_limits(limits=1, user_api="blas"):
        d0, g, q, timing = rank_aware_coefficients(
            network,
            GRID_HORIZON,
            n_time=8,
            n_inner=3,
        )
    peak_working_set: int | None = None
    resident_memory: int | None = None
    try:
        import psutil

        memory = psutil.Process().memory_info()
        resident_memory = int(memory.rss)
        if hasattr(memory, "peak_wset"):
            peak_working_set = int(memory.peak_wset)
    except (ImportError, OSError):
        pass
    payload = {
        "n": network.n,
        "m": network.m,
        "seed": int(seed),
        "n_time": 8,
        "n_inner": 3,
        "n_quad": 72,
        "blas_threads": 1,
        "coefficient_seconds": float(timing["coefficient_seconds"]),
        "first_integral_seconds": float(timing["first_integral_seconds"]),
        "second_integral_seconds": float(timing["second_integral_seconds"]),
        "response_seconds": float(timing["response_seconds"]),
        "d0": float(d0),
        "g_l2": float(np.linalg.norm(g)),
        "q_frobenius": float(np.linalg.norm(q)),
        "q_bytes": int(q.nbytes),
        "ordered_workspace_bytes": int(
            network.m * network.m * network.n * network.n * 16
        ),
        "resident_memory_bytes": resident_memory,
        "peak_working_set_bytes": peak_working_set,
        "theoretical_work_index": float(
            72 * network.m**2 * network.n**2
        ),
    }
    write_json(output_path, payload)


def run_e7_scaling_and_break_even() -> None:
    logging.info("E7: starting extended coefficient scaling")
    targets = ((25, 50), (38, 76), (50, 100))
    new_rows: list[dict[str, object]] = []
    warmup_paths: list[Path] = []
    for target_index, (node_count, expected_edges) in enumerate(targets):
        base_seed = SEED + 700 + target_index
        warmup_path = CHECKPOINTS / f"e7_n{node_count}_warmup.json"
        warmup_paths.append(warmup_path)
        if not warmup_path.exists():
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--e7-worker",
                str(node_count),
                str(base_seed),
                str(warmup_path),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"E7 warm-up failed for N={node_count}: {completed.stderr}"
                )
        for replicate in range(2):
            output_path = CHECKPOINTS / f"e7_n{node_count}_rep{replicate}.json"
            if not output_path.exists():
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--e7-worker",
                    str(node_count),
                    str(base_seed),
                    str(output_path),
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"E7 timed replicate failed for N={node_count}: "
                        f"{completed.stderr}"
                    )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if int(payload["m"]) != expected_edges:
                raise RuntimeError(
                    f"E7 edge-count mismatch for N={node_count}: {payload['m']}"
                )
            payload.update(
                {
                    "replicate": replicate,
                    "source": "r2_measured_after_warmup",
                    "warmup_excluded": True,
                }
            )
            new_rows.append(payload)
            logging.info(
                "E7 N=%d m=%d replicate=%d took %.3f seconds",
                node_count,
                expected_edges,
                replicate,
                payload["coefficient_seconds"],
            )

    r1_rows = pd.read_csv(R1_RESULTS / "runtime_coefficient_scaling.csv")
    r1_rows["source"] = "r1_saved_measurement"
    r1_rows["warmup_excluded"] = False
    r1_rows["q_bytes"] = 8 * r1_rows["m"] ** 2
    r1_rows["ordered_workspace_bytes"] = (
        16 * r1_rows["m"] ** 2 * r1_rows["n"] ** 2
    )
    r1_rows["resident_memory_bytes"] = np.nan
    r1_rows["peak_working_set_bytes"] = np.nan
    r1_rows["theoretical_work_index"] = (
        r1_rows["n_quad"] * r1_rows["m"] ** 2 * r1_rows["n"] ** 2
    )
    new_frame = pd.DataFrame(new_rows)
    combined = pd.concat([r1_rows, new_frame], ignore_index=True, sort=False)
    combined.to_csv(
        RESULTS / "runtime_coefficient_scaling_extended.csv", index=False
    )
    new_frame.to_csv(
        RESULTS / "runtime_coefficient_scaling_r2.csv", index=False
    )

    r1_environment = json.loads(
        (R1_EXTENDED / "r1_environment.json").read_text(encoding="utf-8")
    )
    current_environment = environment_payload()
    environment_keys = ("python", "numpy", "pandas", "scipy", "networkx")
    environment_match = all(
        str(r1_environment[key]) == str(current_environment[key])
        for key in environment_keys
    )

    medians = (
        combined.groupby(["n", "m"], as_index=False)
        .agg(
            coefficient_seconds=("coefficient_seconds", "median"),
            theoretical_work_index=("theoretical_work_index", "first"),
        )
        .sort_values("theoretical_work_index")
    )
    slope, intercept = np.polyfit(
        np.log(medians["theoretical_work_index"]),
        np.log(medians["coefficient_seconds"]),
        1,
    )

    sioux_metadata = json.loads(
        (R1_EXTENDED / "sioux_benchmark_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    scoring_metadata = json.loads(
        (R1_EXTENDED / "sioux_scoring_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_count = float(scoring_metadata["candidate_count"])
    direct_per_candidate = float(
        sioux_metadata["direct_candidate_seconds"] / candidate_count
    )
    stored_score_per_candidate = float(
        scoring_metadata["quadratic_candidate_scoring_seconds"] / candidate_count
    )
    construction_seconds = float(sioux_metadata["coefficient_wall_seconds"])
    full_break_even = construction_seconds / max(
        direct_per_candidate - stored_score_per_candidate, 1.0e-15
    )

    directional = pd.read_csv(R1_EXTENDED / "sioux_matrix_free_runtime.csv")
    row20 = directional.loc[directional["query_count"] == 20].iloc[0]
    stored_per_query = float(row20["stored_q_query_seconds"] / 20.0)
    matrix_free_per_query = float(row20["matrix_free_seconds"] / 20.0)
    directional_break_even = construction_seconds / max(
        matrix_free_per_query - stored_per_query, 1.0e-15
    )
    break_even = {
        "sioux_full_coefficient_construction_seconds": construction_seconds,
        "sioux_direct_candidate_seconds_per_candidate": direct_per_candidate,
        "sioux_stored_quadratic_seconds_per_candidate": stored_score_per_candidate,
        "full_coefficient_vs_direct_break_even_candidate_count": full_break_even,
        "stored_q_seconds_per_directional_query": stored_per_query,
        "matrix_free_seconds_per_directional_query": matrix_free_per_query,
        "stored_q_vs_matrix_free_break_even_query_count": directional_break_even,
        "directional_break_even_is_conditional_on_sioux_and_saved_orders": True,
        "empirical_log_time_vs_log_work_slope": float(slope),
        "empirical_log_time_vs_log_work_intercept": float(intercept),
    }
    write_json(RESULTS / "runtime_break_even.json", break_even)

    audit = {
        "target_node_edge_pairs": [list(value) for value in targets],
        "r2_timed_replicates_per_target": new_frame.groupby("n").size().to_dict(),
        "warmup_files_present": all(path.exists() for path in warmup_paths),
        "one_blas_thread": bool((combined["blas_threads"] == 1).all()),
        "quadrature_settings_match": bool(
            (combined["n_time"] == 8).all()
            and (combined["n_inner"] == 3).all()
            and (combined["n_quad"] == 72).all()
        ),
        "environment_match_for_combining_r1_and_r2": environment_match,
        "environment_keys_checked": list(environment_keys),
        "all_times_finite_positive": bool(
            np.isfinite(combined["coefficient_seconds"]).all()
            and (combined["coefficient_seconds"] > 0.0).all()
        ),
        "peak_working_set_available_for_r2": bool(
            new_frame["peak_working_set_bytes"].notna().all()
        ),
        "measured_edge_maximum": int(new_frame["m"].max()),
        "storage_projection_is_not_a_measurement": True,
    }
    audit["pass"] = bool(
        all(value == 2 for value in audit["r2_timed_replicates_per_target"].values())
        and audit["warmup_files_present"]
        and audit["one_blas_thread"]
        and audit["quadrature_settings_match"]
        and audit["environment_match_for_combining_r1_and_r2"]
        and audit["all_times_finite_positive"]
        and audit["measured_edge_maximum"] == 100
    )
    write_json(RESULTS / "runtime_scaling_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"E7 audit failed: {audit}")
    logging.info("E7: completed extended coefficient scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        choices=("e1", "e2", "e3", "e4", "e5", "e6", "e7", "all"),
        default="all",
    )
    parser.add_argument(
        "--e7-worker",
        nargs=3,
        metavar=("NODE_COUNT", "SEED", "OUTPUT_PATH"),
    )
    arguments = parser.parse_args()
    if arguments.e7_worker is not None:
        node_count, seed, output_path = arguments.e7_worker
        e7_worker(int(node_count), int(seed), Path(output_path))
        return
    configure_logging()
    write_json(RESULTS / "r2_environment.json", environment_payload())
    modules: dict[str, Callable[[], None]] = {
        "e1": run_e1_kernel_comparison,
        "e2": run_e2_scaling_and_horizon,
        "e3": run_e3_pathwise_size_and_family,
        "e4": run_e4_probability_counterfactuals,
        "e5": run_e5_matched_dependence,
        "e6": run_e6_certificate_quantities,
        "e7": run_e7_scaling_and_break_even,
    }
    selected = modules.keys() if arguments.module == "all" else (arguments.module,)
    for key in selected:
        modules[key]()
    logging.info("Selected comparison modules completed")


if __name__ == "__main__":
    main()
