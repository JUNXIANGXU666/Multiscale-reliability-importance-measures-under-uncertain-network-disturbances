"""Sioux Falls, dependence, parameter sensitivity and interval experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.sparse.csgraph import shortest_path
from scipy.stats import beta, norm, spearmanr, t as student_t
from threadpoolctl import threadpool_limits

from dpc_core import (
    Candidate,
    WeightedNetwork,
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
    normalise_source_weights,
    quadratic_loss,
    rank_aware_coefficients,
    stress_scenarios,
    theoretical_epsilon,
)


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "sioux_inputs"
OUTPUT_ROOT = HERE / "sioux_outputs"
RESULTS = OUTPUT_ROOT / "results"
CHECKPOINTS = OUTPUT_ROOT / "checkpoints"
LOGS = OUTPUT_ROOT / "logs"
SEED = 20260804

GRID_HORIZONS = (1.2, 1.8, 2.4)
GRID_STRESSES = (0.20, 0.35, 0.50)
GRID_GAMMAS = (0.95, 0.9625, 0.975)
GRID_SCENARIO_COUNT = 256
GRID_ALPHA_MAX = 0.22
GRID_PRIMARY_RHO = 0.35

SIOUX_HORIZON = 1.8
SIOUX_STRESS = 0.35
SIOUX_GAMMA = 0.97
SIOUX_SCENARIO_COUNT = 128
SIOUX_ALPHA_MAX = 0.18
SIOUX_RHO = 0.35


def ensure_directories() -> None:
    for path in (OUTPUT_ROOT, RESULTS, CHECKPOINTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    ensure_directories()
    handlers = [
        logging.FileHandler(LOGS / "sioux_and_sensitivity.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def write_json(path: Path, payload: dict) -> None:
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
    }


def safe_spearman(reference: Sequence[float], comparison: Sequence[float]) -> float:
    first = np.asarray(reference, dtype=float)
    second = np.asarray(comparison, dtype=float)
    if len(first) < 2 or np.ptp(first) <= 1.0e-15 or np.ptp(second) <= 1.0e-15:
        return float("nan")
    return float(spearmanr(first, second).statistic)


def top_overlap(
    frame: pd.DataFrame,
    reference_column: str,
    comparison_column: str,
    count: int,
) -> float:
    use_count = min(count, len(frame))
    reference = set(
        frame.sort_values(
            [reference_column, "candidate_id"], ascending=[False, True]
        ).head(use_count)["candidate_id"]
    )
    comparison = set(
        frame.sort_values(
            [comparison_column, "candidate_id"], ascending=[False, True]
        ).head(use_count)["candidate_id"]
    )
    return float(len(reference & comparison) / max(use_count, 1))


def sign_class(values: Sequence[float], tolerance: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(array > tolerance, 1, np.where(array < -tolerance, -1, 0))


def parse_sioux_falls() -> tuple[
    WeightedNetwork,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
]:
    directory = INPUTS / "SiouxFalls"
    network_path = directory / "SiouxFalls_net.tntp"
    node_path = directory / "SiouxFalls_node.tntp"
    trips_path = directory / "SiouxFalls_trips.tntp"

    arc_rows: list[dict[str, float | int]] = []
    for raw_line in network_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<") or line.startswith("~"):
            continue
        fields = line.replace(";", "").split()
        if len(fields) < 10 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        arc_rows.append(
            {
                "tail": int(fields[0]) - 1,
                "head": int(fields[1]) - 1,
                "capacity": float(fields[2]),
                "length": float(fields[3]),
                "free_flow_time": float(fields[4]),
                "b": float(fields[5]),
                "power": float(fields[6]),
            }
        )
    arcs = pd.DataFrame(arc_rows)
    if len(arcs) != 76:
        raise RuntimeError(f"Expected 76 Sioux Falls arcs, found {len(arcs)}")

    position_rows: list[dict[str, float | int]] = []
    for raw_line in node_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.strip().replace(";", "").split()
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        position_rows.append(
            {
                "node": int(fields[0]) - 1,
                "x": float(fields[1]),
                "y": float(fields[2]),
            }
        )
    positions_frame = pd.DataFrame(position_rows).sort_values("node")
    if len(positions_frame) != 24:
        raise RuntimeError("Sioux Falls node file did not yield 24 coordinates")
    positions = {
        int(row.node): (float(row.x), float(row.y))
        for row in positions_frame.itertuples(index=False)
    }

    pair_rows: list[dict[str, float | int]] = []
    arcs = arcs.assign(
        u=np.minimum(arcs["tail"], arcs["head"]),
        v=np.maximum(arcs["tail"], arcs["head"]),
        directional_strength=1.0 / arcs["free_flow_time"],
    )
    for (u, v), group in arcs.groupby(["u", "v"], sort=True):
        pair_rows.append(
            {
                "u": int(u),
                "v": int(v),
                "direction_count": int(len(group)),
                "capacity_mean": float(group["capacity"].mean()),
                "length_mean": float(group["length"].mean()),
                "free_flow_time_mean": float(group["free_flow_time"].mean()),
                "raw_strength": float(group["directional_strength"].mean()),
            }
        )
    physical = pd.DataFrame(pair_rows).sort_values(["u", "v"]).reset_index(drop=True)
    if len(physical) != 38:
        raise RuntimeError(f"Expected 38 physical edges, found {len(physical)}")
    physical["edge_id"] = np.arange(len(physical), dtype=int)
    physical["weight"] = physical["raw_strength"] / physical["raw_strength"].max()

    od_production = np.zeros(24, dtype=float)
    current_origin: int | None = None
    origin_pattern = re.compile(r"^Origin\s+(\d+)")
    demand_pattern = re.compile(r"(\d+)\s*:\s*([0-9.]+)")
    for raw_line in trips_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        origin_match = origin_pattern.match(line)
        if origin_match:
            current_origin = int(origin_match.group(1)) - 1
            continue
        if current_origin is None:
            continue
        for _, value in demand_pattern.findall(line):
            od_production[current_origin] += float(value)
    if not np.isclose(od_production.sum(), 360600.0):
        raise RuntimeError(
            f"Unexpected Sioux Falls total OD flow {od_production.sum():.12g}"
        )
    source_weights = normalise_source_weights(od_production, 24)

    network = WeightedNetwork(
        name="sioux_falls_tntp",
        n=24,
        edges=physical[["u", "v"]].to_numpy(dtype=int),
        weights=physical["weight"].to_numpy(dtype=float),
        positions=positions,
    )
    productions = pd.DataFrame(
        {
            "node": np.arange(24, dtype=int),
            "origin_production": od_production,
            "source_weight": source_weights,
        }
    )
    return network, physical, source_weights, productions


def edge_lookup(network: WeightedNetwork) -> dict[tuple[int, int], int]:
    return {
        tuple(sorted((int(u), int(v)))): edge_id
        for edge_id, (u, v) in enumerate(network.edges)
    }


def candidate_membership_rows(
    network: WeightedNetwork,
    candidates: Sequence[Candidate],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        edge_ids = tuple(sorted(int(value) for value in candidate.edge_ids))
        nodes = sorted(
            {
                int(node)
                for edge_id in edge_ids
                for node in network.edges[edge_id]
            }
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "form": candidate.scale,
                "size": len(edge_ids),
                "edge_ids": "|".join(str(value) for value in edge_ids),
                "node_ids": "|".join(str(value) for value in nodes),
            }
        )
    return rows


def sioux_candidates(network: WeightedNetwork) -> list[Candidate]:
    graph = network.as_networkx()
    lookup = edge_lookup(network)
    candidates: list[Candidate] = [
        Candidate(f"E{edge_id + 1:03d}", "edge", (edge_id,), f"Edge {edge_id + 1}")
        for edge_id in range(network.m)
    ]

    path_supports: set[tuple[int, ...]] = set()
    nodes = sorted(graph.nodes)
    for source_index, source in enumerate(nodes):
        for target in nodes[source_index + 1 :]:
            for path in nx.all_simple_paths(graph, source, target, cutoff=4):
                edge_count = len(path) - 1
                if edge_count < 2:
                    continue
                support = tuple(
                    sorted(
                        lookup[tuple(sorted((int(first), int(second))))]
                        for first, second in zip(path[:-1], path[1:], strict=True)
                    )
                )
                path_supports.add(support)
    for index, support in enumerate(sorted(path_supports), start=1):
        candidates.append(
            Candidate(f"C{index:04d}", "corridor", support, f"Corridor {index}")
        )

    region_supports: set[tuple[int, ...]] = set()
    for centre in nodes:
        region_nodes = {centre, *graph.neighbors(centre)}
        induced = graph.subgraph(region_nodes)
        support = tuple(
            sorted(
                lookup[tuple(sorted((int(u), int(v))))]
                for u, v in induced.edges
            )
        )
        if len(support) < 3:
            continue
        support_graph = nx.Graph()
        support_graph.add_edges_from(network.edges[list(support)])
        has_branch = any(degree >= 3 for _, degree in support_graph.degree)
        has_cycle = bool(nx.cycle_basis(support_graph))
        is_simple_path = (
            nx.is_connected(support_graph)
            and not has_cycle
            and sum(degree == 1 for _, degree in support_graph.degree) == 2
            and max(dict(support_graph.degree).values(), default=0) <= 2
        )
        if (has_branch or has_cycle) and not is_simple_path:
            region_supports.add(support)
    for index, support in enumerate(sorted(region_supports), start=1):
        candidates.append(
            Candidate(f"R{index:03d}", "region", support, f"Region {index}")
        )
    return candidates


def matched_copula_scenarios(
    family: str,
    m: int,
    count: int,
    alpha_bar: np.ndarray,
    kendall_tau: float,
    seed: int,
    beta_a: float = 2.0,
    beta_b: float = 8.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "independent":
        uniforms = rng.uniform(1.0e-10, 1.0 - 1.0e-10, size=(count, m))
    elif family in {"gaussian", "student_t4"}:
        rho = float(np.sin(0.5 * np.pi * kendall_tau))
        common = rng.standard_normal((count, 1))
        independent = rng.standard_normal((count, m))
        latent_normal = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * independent
        if family == "gaussian":
            uniforms = norm.cdf(latent_normal)
        else:
            degrees = 4
            scale = np.sqrt(rng.chisquare(degrees, size=(count, 1)) / degrees)
            uniforms = student_t.cdf(latent_normal / scale, df=degrees)
    elif family == "clayton":
        if not 0.0 < kendall_tau < 1.0:
            raise ValueError("Clayton dependence requires Kendall tau in (0,1)")
        theta = 2.0 * kendall_tau / (1.0 - kendall_tau)
        frailty = rng.gamma(shape=1.0 / theta, scale=1.0, size=(count, 1))
        exponential = rng.exponential(scale=1.0, size=(count, m))
        uniforms = (1.0 + exponential / frailty) ** (-1.0 / theta)
    else:
        raise ValueError(f"Unknown copula family {family}")
    clipped = np.clip(uniforms, 1.0e-10, 1.0 - 1.0e-10)
    proportions = beta.ppf(clipped, beta_a, beta_b)
    return proportions * np.asarray(alpha_bar, dtype=float)[None, :]


def rank_match_beta_marginals(
    scenarios: np.ndarray,
    alpha_bar: np.ndarray,
    beta_a: float = 2.0,
    beta_b: float = 8.0,
) -> np.ndarray:
    values = np.asarray(scenarios, dtype=float)
    probabilities = (np.arange(len(values), dtype=float) + 0.5) / len(values)
    target = beta.ppf(probabilities, beta_a, beta_b)
    ranks = np.argsort(np.argsort(values, axis=0), axis=0)
    return target[ranks] * np.asarray(alpha_bar, dtype=float)[None, :]


def weighted_global_efficiency_batch(
    network: WeightedNetwork,
    alphas: np.ndarray,
    source_weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.atleast_2d(np.asarray(alphas, dtype=float))
    source_mass = normalise_source_weights(source_weights, network.n)
    output = np.empty(len(values), dtype=float)
    for index, alpha in enumerate(values):
        adjacency = np.full((network.n, network.n), np.inf, dtype=float)
        np.fill_diagonal(adjacency, 0.0)
        active = network.weights * (1.0 - alpha)
        for (u, v), strength in zip(network.edges, active, strict=True):
            if strength <= 0.0:
                continue
            distance = 1.0 / strength
            adjacency[int(u), int(v)] = min(adjacency[int(u), int(v)], distance)
            adjacency[int(v), int(u)] = min(adjacency[int(v), int(u)], distance)
        distances = shortest_path(adjacency, directed=False, unweighted=False)
        reciprocal = np.zeros_like(distances)
        finite = np.isfinite(distances) & (distances > 0.0)
        reciprocal[finite] = 1.0 / distances[finite]
        per_source = np.sum(reciprocal, axis=1) / max(network.n - 1, 1)
        output[index] = float(source_mass @ per_source)
    return output


def direct_importance_table(
    network: WeightedNetwork,
    candidates: Sequence[Candidate],
    scenarios: np.ndarray,
    alpha_bar: np.ndarray,
    horizon: float,
    gamma: float,
    stresses: float | dict[str, float],
    n_time: int,
    source_weights: np.ndarray | None,
    checkpoint_stem: str,
    metric: str = "dpc",
) -> tuple[pd.DataFrame, dict[str, float]]:
    checkpoint_path = CHECKPOINTS / f"{checkpoint_stem}.csv"
    if metric == "dpc":
        zero_value = dpc(
            network,
            np.zeros(network.m),
            horizon,
            n_time,
            source_weights=source_weights,
        )
        baseline_values = dpc_batch(
            network,
            scenarios,
            horizon,
            n_time,
            source_weights=source_weights,
        )
    elif metric == "global_efficiency":
        zero_value = float(
            weighted_global_efficiency_batch(
                network,
                np.zeros((1, network.m)),
                source_weights=source_weights,
            )[0]
        )
        baseline_values = weighted_global_efficiency_batch(
            network,
            scenarios,
            source_weights=source_weights,
        )
    else:
        raise ValueError(metric)
    threshold = gamma * zero_value
    baseline_reliability = float(np.mean(baseline_values >= threshold))

    completed: dict[str, dict[str, object]] = {}
    if checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        completed = {
            str(row["candidate_id"]): row.to_dict()
            for _, row in prior.iterrows()
        }
        logging.info("Resuming %s with %d candidates", checkpoint_stem, len(completed))

    rows: list[dict[str, object]] = list(completed.values())
    for candidate_index, candidate in enumerate(candidates, start=1):
        if candidate.candidate_id in completed:
            continue
        stress = (
            float(stresses[candidate.candidate_id])
            if isinstance(stresses, dict)
            else float(stresses)
        )
        stressed = stress_scenarios(
            scenarios,
            candidate.edge_ids,
            stress,
            alpha_bar,
        )
        if metric == "dpc":
            stressed_values = dpc_batch(
                network,
                stressed,
                horizon,
                n_time,
                source_weights=source_weights,
            )
        else:
            stressed_values = weighted_global_efficiency_batch(
                network,
                stressed,
                source_weights=source_weights,
            )
        stressed_reliability = float(np.mean(stressed_values >= threshold))
        row = {
            "candidate_id": candidate.candidate_id,
            "form": candidate.scale,
            "size": candidate.size,
            "stress": stress,
            "baseline_reliability": baseline_reliability,
            "stressed_reliability": stressed_reliability,
            "direct_empirical_importance": baseline_reliability
            - stressed_reliability,
            "baseline_mean_performance": float(np.mean(baseline_values)),
            "stressed_mean_performance": float(np.mean(stressed_values)),
            "threshold": threshold,
            "metric": metric,
        }
        rows.append(row)
        if candidate_index % 10 == 0 or candidate_index == len(candidates):
            pd.DataFrame(rows).sort_values("candidate_id").to_csv(
                checkpoint_path, index=False
            )
            logging.info(
                "%s completed %d of %d candidates",
                checkpoint_stem,
                candidate_index,
                len(candidates),
            )
    frame = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    return frame, {
        "zero_performance": zero_value,
        "threshold": threshold,
        "baseline_reliability": baseline_reliability,
        "baseline_mean_performance": float(np.mean(baseline_values)),
    }


def expected_quadratic_loss_decomposition(
    scenarios: np.ndarray,
    stressed: np.ndarray,
    g: np.ndarray,
    q: np.ndarray,
) -> dict[str, float]:
    baseline_mean = np.mean(scenarios, axis=0)
    stressed_mean = np.mean(stressed, axis=0)
    baseline_covariance = np.cov(scenarios, rowvar=False, bias=True)
    stressed_covariance = np.cov(stressed, rowvar=False, bias=True)
    mean_contribution = float(
        g @ (stressed_mean - baseline_mean)
        + 0.5
        * (
            stressed_mean @ q @ stressed_mean
            - baseline_mean @ q @ baseline_mean
        )
    )
    covariance_change = stressed_covariance - baseline_covariance
    variance_contribution = float(
        0.5 * np.sum(np.diag(q) * np.diag(covariance_change))
    )
    off_diagonal_q = q - np.diag(np.diag(q))
    covariance_contribution = float(0.5 * np.sum(off_diagonal_q * covariance_change))
    direct_total = float(
        np.mean(quadratic_loss(stressed, g, q))
        - np.mean(quadratic_loss(scenarios, g, q))
    )
    reconstructed = (
        mean_contribution + variance_contribution + covariance_contribution
    )
    return {
        "mean_shift_contribution": mean_contribution,
        "marginal_variance_contribution": variance_contribution,
        "off_diagonal_covariance_contribution": covariance_contribution,
        "total_expected_quadratic_loss_change": direct_total,
        "reconstructed_change": reconstructed,
        "closure_error": reconstructed - direct_total,
    }


def run_pilot() -> None:
    logging.info("Starting Sioux Falls and sensitivity pilot")
    network, physical, source_weights, productions = parse_sioux_falls()
    candidates = sioux_candidates(network)
    candidate_frame = pd.DataFrame(candidate_membership_rows(network, candidates))

    test_network = motif_networks()["path4"]
    zero = np.zeros(test_network.m, dtype=float)
    uniform = np.full(test_network.n, 1.0 / test_network.n)
    implicit = dpc(test_network, zero, 1.8, n_time=48)
    explicit = dpc(
        test_network,
        zero,
        1.8,
        n_time=48,
        source_weights=uniform,
    )
    source_test = normalise_source_weights(np.arange(1, test_network.n + 1), test_network.n)
    d0_rank, g_rank, q_rank, _ = rank_aware_coefficients(
        test_network,
        1.8,
        n_time=20,
        n_inner=6,
        source_weights=source_test,
    )
    d0_fd, g_fd, q_fd, _ = finite_difference_coefficients(
        test_network,
        1.8,
        n_time=64,
        step=5.0e-4,
        fourth_order_edges=True,
        richardson_mixed=True,
        source_weights=source_test,
    )
    g_relative = float(np.linalg.norm(g_rank - g_fd) / max(np.linalg.norm(g_fd), 1e-15))
    q_relative = float(np.linalg.norm(q_rank - q_fd) / max(np.linalg.norm(q_fd), 1e-15))

    alpha_bar = np.full(test_network.m, 0.18)
    copula_bounds: dict[str, dict[str, float]] = {}
    for family_index, family in enumerate(
        ("independent", "gaussian", "student_t4", "clayton")
    ):
        scenarios = matched_copula_scenarios(
            family,
            test_network.m,
            512,
            alpha_bar,
            0.35,
            SEED + family_index,
        )
        copula_bounds[family] = {
            "minimum": float(np.min(scenarios)),
            "maximum": float(np.max(scenarios)),
            "mean": float(np.mean(scenarios)),
        }

    scenarios = matched_copula_scenarios(
        "gaussian",
        test_network.m,
        256,
        alpha_bar,
        0.35,
        SEED + 20,
    )
    candidate = motif_candidates(test_network)[0]
    stressed = stress_scenarios(scenarios, candidate.edge_ids, 0.35, alpha_bar)
    decomposition = expected_quadratic_loss_decomposition(
        scenarios, stressed, g_rank, q_rank
    )

    random_disturbances = np.random.default_rng(SEED + 30).uniform(
        0.0, 0.18, size=(32, network.m)
    )
    zero_efficiency = float(
        weighted_global_efficiency_batch(
            network,
            np.zeros((1, network.m)),
            source_weights=source_weights,
        )[0]
    )
    disturbed_efficiency = weighted_global_efficiency_batch(
        network,
        random_disturbances,
        source_weights=source_weights,
    )

    form_counts = candidate_frame.groupby("form").size().to_dict()
    input_hashes = {
        path.name: sha256(path)
        for path in sorted((INPUTS / "SiouxFalls").glob("*.tntp"))
    }
    checks = {
        "uniform_source_equivalence_error": abs(implicit - explicit),
        "weighted_d0_relative_error": abs(d0_rank - d0_fd) / max(abs(d0_fd), 1e-15),
        "weighted_g_relative_error": g_relative,
        "weighted_q_relative_error": q_relative,
        "decomposition_closure_error": abs(decomposition["closure_error"]),
        "all_copulas_bounded": all(
            value["minimum"] >= 0.0 and value["maximum"] <= 0.18 + 1.0e-12
            for value in copula_bounds.values()
        ),
        "efficiency_monotone": bool(
            np.all(disturbed_efficiency <= zero_efficiency + 1.0e-12)
        ),
        "sioux_node_count": network.n,
        "sioux_physical_edge_count": network.m,
        "sioux_candidate_counts": form_counts,
        "sioux_total_od_flow": float(productions["origin_production"].sum()),
        "sioux_min_weight": float(physical["weight"].min()),
        "sioux_max_weight": float(physical["weight"].max()),
        "input_hashes": input_hashes,
        "copula_bounds": copula_bounds,
    }
    checks["pass"] = bool(
        checks["uniform_source_equivalence_error"] < 1.0e-12
        and checks["weighted_d0_relative_error"] < 1.0e-7
        and checks["weighted_g_relative_error"] < 5.0e-5
        and checks["weighted_q_relative_error"] < 5.0e-4
        and checks["decomposition_closure_error"] < 1.0e-12
        and checks["all_copulas_bounded"]
        and checks["efficiency_monotone"]
        and form_counts == {"corridor": 739, "edge": 38, "region": 20}
        and np.isclose(checks["sioux_total_od_flow"], 360600.0)
    )
    write_json(RESULTS / "r1_pilot_audit.json", checks)
    if not checks["pass"]:
        raise RuntimeError(f"Numerical pilot failed: {checks}")
    logging.info("Sioux Falls and sensitivity pilot passed")


def benchmark_rank_comparison(
    dpc_frame: pd.DataFrame,
    comparator_frame: pd.DataFrame,
    uniform_edge_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = dpc_frame.merge(
        comparator_frame[
            ["candidate_id", "direct_empirical_importance"]
        ].rename(columns={"direct_empirical_importance": "efficiency_importance"}),
        on="candidate_id",
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for form, group in merged.groupby("form", sort=True):
        rows.append(
            {
                "comparison": "weighted_dpc_vs_weighted_global_efficiency",
                "form": form,
                "candidate_count": len(group),
                "spearman": safe_spearman(
                    group["direct_empirical_importance"],
                    group["efficiency_importance"],
                ),
                "top5_overlap": top_overlap(
                    group,
                    "direct_empirical_importance",
                    "efficiency_importance",
                    5,
                ),
            }
        )
    weighted_edges = dpc_frame[dpc_frame["form"] == "edge"].merge(
        uniform_edge_frame[
            ["candidate_id", "direct_empirical_importance"]
        ].rename(columns={"direct_empirical_importance": "uniform_importance"}),
        on="candidate_id",
    )
    rows.append(
        {
            "comparison": "od_weighted_vs_equal_source_dpc",
            "form": "edge",
            "candidate_count": len(weighted_edges),
            "spearman": safe_spearman(
                weighted_edges["direct_empirical_importance"],
                weighted_edges["uniform_importance"],
            ),
            "top5_overlap": top_overlap(
                weighted_edges,
                "direct_empirical_importance",
                "uniform_importance",
                5,
            ),
        }
    )
    return pd.DataFrame(rows)


def run_sioux_benchmark() -> None:
    logging.info("Starting Sioux Falls benchmark")
    network, physical, source_weights, productions = parse_sioux_falls()
    candidates = sioux_candidates(network)
    candidate_frame = pd.DataFrame(candidate_membership_rows(network, candidates))
    physical.to_csv(RESULTS / "sioux_physical_edges.csv", index=False)
    productions.to_csv(RESULTS / "sioux_source_weights.csv", index=False)
    candidate_frame.to_csv(RESULTS / "sioux_candidates.csv", index=False)

    alpha_bar = np.full(network.m, SIOUX_ALPHA_MAX, dtype=float)
    scenarios = gaussian_copula_beta_scenarios(
        network.m,
        SIOUX_SCENARIO_COUNT,
        alpha_bar,
        SIOUX_RHO,
        SEED + 100,
    )
    np.savez_compressed(
        RESULTS / "sioux_scenarios.npz",
        scenarios=scenarios,
        alpha_bar=alpha_bar,
        source_weights=source_weights,
    )

    with threadpool_limits(limits=1, user_api="blas"):
        coefficient_start = perf_counter()
        d0, g, q, coefficient_timing = rank_aware_coefficients(
            network,
            SIOUX_HORIZON,
            n_time=18,
            n_inner=5,
            source_weights=source_weights,
        )
        coefficient_wall = perf_counter() - coefficient_start
    np.savez_compressed(
        RESULTS / "sioux_coefficients_weighted.npz",
        d0=np.array([d0]),
        g=g,
        q=q,
        edges=network.edges,
        weights=network.weights,
        source_weights=source_weights,
    )

    direct_start = perf_counter()
    dpc_frame, dpc_metadata = direct_importance_table(
        network,
        candidates,
        scenarios,
        alpha_bar,
        SIOUX_HORIZON,
        SIOUX_GAMMA,
        SIOUX_STRESS,
        n_time=32,
        source_weights=source_weights,
        checkpoint_stem="sioux_weighted_dpc",
        metric="dpc",
    )
    direct_seconds = perf_counter() - direct_start
    dpc_frame.to_csv(RESULTS / "sioux_weighted_dpc_importance.csv", index=False)

    efficiency_start = perf_counter()
    efficiency_frame, efficiency_metadata = direct_importance_table(
        network,
        candidates,
        scenarios,
        alpha_bar,
        SIOUX_HORIZON,
        SIOUX_GAMMA,
        SIOUX_STRESS,
        n_time=32,
        source_weights=source_weights,
        checkpoint_stem="sioux_weighted_efficiency",
        metric="global_efficiency",
    )
    efficiency_seconds = perf_counter() - efficiency_start
    efficiency_frame.to_csv(
        RESULTS / "sioux_weighted_global_efficiency_importance.csv", index=False
    )

    edge_candidates = [candidate for candidate in candidates if candidate.scale == "edge"]
    uniform_edge_frame, uniform_metadata = direct_importance_table(
        network,
        edge_candidates,
        scenarios,
        alpha_bar,
        SIOUX_HORIZON,
        SIOUX_GAMMA,
        SIOUX_STRESS,
        n_time=32,
        source_weights=None,
        checkpoint_stem="sioux_uniform_source_edges",
        metric="dpc",
    )
    uniform_edge_frame.to_csv(
        RESULTS / "sioux_uniform_source_edge_importance.csv", index=False
    )

    baseline_losses = d0 - dpc_batch(
        network,
        scenarios,
        SIOUX_HORIZON,
        n_time=32,
        source_weights=source_weights,
    )
    approximate_baseline = np.asarray(quadratic_loss(scenarios, g, q))
    qx = scenarios @ q.T
    theta = d0 - SIOUX_GAMMA * d0
    approximate_baseline_reliability = exact_reliability(approximate_baseline, theta)
    approximation_rows: list[dict[str, object]] = []
    for candidate in candidates:
        stressed_approximate = incremental_quadratic_loss(
            scenarios,
            candidate.edge_ids,
            SIOUX_STRESS,
            alpha_bar,
            g,
            q,
            baseline_losses=approximate_baseline,
            qx=qx,
        )
        approximation_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "form": candidate.scale,
                "quadratic_empirical_importance": approximate_baseline_reliability
                - exact_reliability(stressed_approximate, theta),
            }
        )
    approximation_frame = pd.DataFrame(approximation_rows).merge(
        dpc_frame[
            ["candidate_id", "direct_empirical_importance"]
        ],
        on="candidate_id",
    )
    approximation_frame["importance_error"] = (
        approximation_frame["quadratic_empirical_importance"]
        - approximation_frame["direct_empirical_importance"]
    )
    approximation_frame.to_csv(
        RESULTS / "sioux_quadratic_accuracy.csv", index=False
    )

    convergence_count = min(24, len(scenarios))
    order32 = dpc_batch(
        network,
        scenarios[:convergence_count],
        SIOUX_HORIZON,
        n_time=32,
        source_weights=source_weights,
    )
    order48 = dpc_batch(
        network,
        scenarios[:convergence_count],
        SIOUX_HORIZON,
        n_time=48,
        source_weights=source_weights,
    )
    convergence = pd.DataFrame(
        {
            "scenario_index": np.arange(convergence_count),
            "dpc_order32": order32,
            "dpc_order48": order48,
            "absolute_difference": np.abs(order32 - order48),
        }
    )
    convergence.to_csv(RESULTS / "sioux_dpc_quadrature_check.csv", index=False)

    comparison = benchmark_rank_comparison(
        dpc_frame, efficiency_frame, uniform_edge_frame
    )
    comparison.to_csv(RESULTS / "sioux_rank_comparisons.csv", index=False)

    storage_rows = []
    for edge_count in (network.m, 100, 1000, 10000, 50000):
        dense_bytes = 8 * edge_count * edge_count
        triangular_bytes = 8 * edge_count * (edge_count + 1) // 2
        storage_rows.append(
            {
                "edge_count": edge_count,
                "dense_bytes": dense_bytes,
                "upper_triangular_bytes": triangular_bytes,
                "dense_megabytes": dense_bytes / 1024**2,
                "dense_gigabytes": dense_bytes / 1024**3,
            }
        )
    pd.DataFrame(storage_rows).to_csv(
        RESULTS / "interaction_matrix_storage_projection.csv", index=False
    )

    coupling_example = physical.iloc[0].to_dict()
    metadata = {
        "network": {
            "nodes": network.n,
            "physical_edges": network.m,
            "directed_arcs": 76,
            "candidate_counts": candidate_frame.groupby("form").size().to_dict(),
        },
        "settings": {
            "horizon": SIOUX_HORIZON,
            "stress": SIOUX_STRESS,
            "gamma": SIOUX_GAMMA,
            "scenario_count": SIOUX_SCENARIO_COUNT,
            "alpha_max": SIOUX_ALPHA_MAX,
            "gaussian_common_factor": SIOUX_RHO,
            "direct_time_order": 32,
            "coefficient_outer_order": 18,
            "coefficient_inner_order": 5,
        },
        "coefficient_timing": coefficient_timing,
        "coefficient_wall_seconds": coefficient_wall,
        "full_q_bytes": int(q.nbytes),
        "direct_candidate_seconds": direct_seconds,
        "efficiency_candidate_seconds": efficiency_seconds,
        "dpc_metadata": dpc_metadata,
        "efficiency_metadata": efficiency_metadata,
        "uniform_edge_metadata": uniform_metadata,
        "baseline_quadratic_loss_rmse": float(
            np.sqrt(np.mean((baseline_losses - approximate_baseline) ** 2))
        ),
        "quadrature_max_absolute_difference": float(
            convergence["absolute_difference"].max()
        ),
        "coupling_example_first_edge": coupling_example,
    }
    write_json(RESULTS / "sioux_benchmark_metadata.json", metadata)
    logging.info("Completed Sioux Falls benchmark")


def compute_stressed_dpc_matrix(
    network: WeightedNetwork,
    candidates: Sequence[Candidate],
    scenarios: np.ndarray,
    alpha_bar: np.ndarray,
    horizon: float,
    stresses: float | dict[str, float],
    n_time: int,
    checkpoint_name: str,
    source_weights: np.ndarray | None = None,
) -> np.ndarray:
    checkpoint_path = CHECKPOINTS / f"{checkpoint_name}.npz"
    matrix = np.full((len(candidates), len(scenarios)), np.nan, dtype=float)
    completed = 0
    if checkpoint_path.exists():
        archive = np.load(checkpoint_path, allow_pickle=False)
        prior_ids = [str(value) for value in archive["candidate_ids"]]
        expected_ids = [candidate.candidate_id for candidate in candidates]
        if prior_ids != expected_ids:
            raise RuntimeError(f"Checkpoint candidate mismatch: {checkpoint_path}")
        matrix = archive["stressed_dpc"]
        completed = int(archive["completed"][0])
        logging.info("Resuming %s at candidate %d", checkpoint_name, completed)
    for index in range(completed, len(candidates)):
        candidate = candidates[index]
        stress = (
            float(stresses[candidate.candidate_id])
            if isinstance(stresses, dict)
            else float(stresses)
        )
        stressed = stress_scenarios(
            scenarios, candidate.edge_ids, stress, alpha_bar
        )
        matrix[index] = dpc_batch(
            network,
            stressed,
            horizon,
            n_time=n_time,
            source_weights=source_weights,
        )
        if (index + 1) % 10 == 0 or index + 1 == len(candidates):
            np.savez_compressed(
                checkpoint_path,
                candidate_ids=np.asarray(
                    [candidate.candidate_id for candidate in candidates], dtype="U16"
                ),
                stressed_dpc=matrix,
                completed=np.asarray([index + 1], dtype=int),
            )
            logging.info(
                "%s completed %d of %d candidates",
                checkpoint_name,
                index + 1,
                len(candidates),
            )
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Incomplete DPC checkpoint {checkpoint_name}")
    return matrix


def run_full_sensitivity() -> None:
    logging.info("Starting full L-lambda-gamma sensitivity")
    network = grid_network(5, 5, 20260716)
    candidates = grid_candidates(network, 5, 5)
    alpha_bar = np.full(network.m, GRID_ALPHA_MAX, dtype=float)
    scenarios = gaussian_copula_beta_scenarios(
        network.m,
        GRID_SCENARIO_COUNT,
        alpha_bar,
        GRID_PRIMARY_RHO,
        20260716,
    )
    rows: list[dict[str, object]] = []
    for horizon in GRID_HORIZONS:
        zero_dpc = dpc(
            network, np.zeros(network.m), horizon, n_time=48
        )
        baseline_dpc = dpc_batch(network, scenarios, horizon, n_time=48)
        for stress in GRID_STRESSES:
            checkpoint_name = (
                f"grid_full_sensitivity_L{horizon:.4f}_lambda{stress:.4f}"
                .replace(".", "p")
            )
            stressed_matrix = compute_stressed_dpc_matrix(
                network,
                candidates,
                scenarios,
                alpha_bar,
                horizon,
                stress,
                48,
                checkpoint_name,
            )
            for gamma in GRID_GAMMAS:
                threshold = gamma * zero_dpc
                baseline_reliability = float(np.mean(baseline_dpc >= threshold))
                stressed_reliability = np.mean(
                    stressed_matrix >= threshold, axis=1
                )
                importance = baseline_reliability - stressed_reliability
                for candidate, value, reliability in zip(
                    candidates,
                    importance,
                    stressed_reliability,
                    strict=True,
                ):
                    rows.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "form": candidate.scale,
                            "size": candidate.size,
                            "horizon": horizon,
                            "stress": stress,
                            "gamma": gamma,
                            "zero_dpc": zero_dpc,
                            "baseline_reliability": baseline_reliability,
                            "stressed_reliability": float(reliability),
                            "direct_empirical_importance": float(value),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "full_parameter_sensitivity.csv", index=False)

    reference = frame[
        np.isclose(frame["horizon"], 1.8)
        & np.isclose(frame["stress"], 0.35)
        & np.isclose(frame["gamma"], 0.9625)
    ][["candidate_id", "form", "direct_empirical_importance"]].rename(
        columns={"direct_empirical_importance": "reference_importance"}
    )
    metric_rows: list[dict[str, object]] = []
    for (horizon, stress, gamma, form), group in frame.groupby(
        ["horizon", "stress", "gamma", "form"], sort=True
    ):
        merged = group.merge(reference, on=["candidate_id", "form"])
        tolerance = 0.5 / GRID_SCENARIO_COUNT
        current_sign = sign_class(
            merged["direct_empirical_importance"], tolerance
        )
        reference_sign = sign_class(merged["reference_importance"], tolerance)
        metric_rows.append(
            {
                "horizon": horizon,
                "stress": stress,
                "gamma": gamma,
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
                "sign_agreement": float(np.mean(current_sign == reference_sign)),
                "positive_count": int(np.sum(current_sign == 1)),
                "negative_count": int(np.sum(current_sign == -1)),
                "zero_count": int(np.sum(current_sign == 0)),
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        RESULTS / "full_parameter_stability_metrics.csv", index=False
    )
    logging.info("Completed full L-lambda-gamma sensitivity")


def run_dependence_families() -> None:
    logging.info("Starting non-Gaussian dependence experiment")
    network = grid_network(5, 5, 20260716)
    candidates = grid_candidates(network, 5, 5)
    alpha_bar = np.full(network.m, GRID_ALPHA_MAX, dtype=float)
    zero_dpc = dpc(network, np.zeros(network.m), 1.8, n_time=48)
    threshold = 0.9625 * zero_dpc
    candidate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    families = ("independent", "gaussian", "student_t4", "clayton")
    for family_index, family in enumerate(families):
        scenarios = matched_copula_scenarios(
            family,
            network.m,
            GRID_SCENARIO_COUNT,
            alpha_bar,
            0.35,
            SEED + 300 + family_index,
        )
        scenarios = rank_match_beta_marginals(scenarios, alpha_bar)
        baseline_dpc = dpc_batch(network, scenarios, 1.8, n_time=48)
        stressed_matrix = compute_stressed_dpc_matrix(
            network,
            candidates,
            scenarios,
            alpha_bar,
            1.8,
            0.35,
            48,
            f"grid_dependence_rankmatched_{family}",
        )
        baseline_reliability = float(np.mean(baseline_dpc >= threshold))
        stressed_reliability = np.mean(stressed_matrix >= threshold, axis=1)
        importance = baseline_reliability - stressed_reliability
        empirical_tau_values = []
        for edge in range(1, min(network.m, 8)):
            empirical_tau_values.append(
                pd.Series(scenarios[:, 0]).corr(
                    pd.Series(scenarios[:, edge]), method="kendall"
                )
            )
        summary_rows.append(
            {
                "family": family,
                "scenario_count": len(scenarios),
                "target_kendall_tau": 0.0 if family == "independent" else 0.35,
                "mean_empirical_kendall_tau_first_eight": float(
                    np.nanmean(empirical_tau_values)
                ),
                "marginal_mean": float(np.mean(scenarios)),
                "marginal_standard_deviation": float(np.std(scenarios)),
                "baseline_reliability": baseline_reliability,
                "baseline_mean_dpc": float(np.mean(baseline_dpc)),
            }
        )
        for candidate, value, reliability in zip(
            candidates, importance, stressed_reliability, strict=True
        ):
            candidate_rows.append(
                {
                    "family": family,
                    "candidate_id": candidate.candidate_id,
                    "form": candidate.scale,
                    "size": candidate.size,
                    "direct_empirical_importance": float(value),
                    "stressed_reliability": float(reliability),
                }
            )
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(
        RESULTS / "dependence_family_candidate_importance.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        RESULTS / "dependence_family_summary.csv", index=False
    )

    reference = candidate_frame[candidate_frame["family"] == "gaussian"][
        ["candidate_id", "form", "direct_empirical_importance"]
    ].rename(columns={"direct_empirical_importance": "reference_importance"})
    stability_rows = []
    tolerance = 0.5 / GRID_SCENARIO_COUNT
    for (family, form), group in candidate_frame.groupby(["family", "form"]):
        merged = group.merge(reference, on=["candidate_id", "form"])
        current_sign = sign_class(
            merged["direct_empirical_importance"], tolerance
        )
        reference_sign = sign_class(merged["reference_importance"], tolerance)
        stability_rows.append(
            {
                "family": family,
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
                "sign_agreement": float(np.mean(current_sign == reference_sign)),
            }
        )
    pd.DataFrame(stability_rows).to_csv(
        RESULTS / "dependence_family_stability.csv", index=False
    )
    logging.info("Completed non-Gaussian dependence experiment")


def run_stress_decomposition_and_fairness() -> None:
    logging.info("Starting stress decomposition and equal-total-stress analysis")
    archive = np.load(HERE / "results" / "grid_experiment_arrays.npz")
    network = grid_network(5, 5, 20260716)
    candidates = grid_candidates(network, 5, 5)
    scenarios = archive["scenarios"]
    alpha_bar = archive["alpha_bar"]
    g = archive["g"]
    q = archive["q"]
    decomposition_rows = []
    for candidate in candidates:
        stressed = stress_scenarios(
            scenarios, candidate.edge_ids, 0.35, alpha_bar
        )
        row = expected_quadratic_loss_decomposition(scenarios, stressed, g, q)
        row.update(
            {
                "candidate_id": candidate.candidate_id,
                "form": candidate.scale,
                "size": candidate.size,
            }
        )
        decomposition_rows.append(row)
    decomposition_frame = pd.DataFrame(decomposition_rows)
    decomposition_frame.to_csv(
        RESULTS / "stress_moment_decomposition.csv", index=False
    )

    mean_headroom = alpha_bar - np.mean(scenarios, axis=0)
    target_expected_increment = 0.35 * float(np.mean(mean_headroom))
    stresses = {
        candidate.candidate_id: target_expected_increment
        / float(np.sum(mean_headroom[list(candidate.edge_ids)]))
        for candidate in candidates
    }
    fair_frame, fair_metadata = direct_importance_table(
        network,
        candidates,
        scenarios,
        alpha_bar,
        1.8,
        0.9625,
        stresses,
        n_time=48,
        source_weights=None,
        checkpoint_stem="grid_equal_total_expected_stress",
        metric="dpc",
    )
    fair_frame["total_expected_added_disturbance"] = [
        stresses[candidate.candidate_id]
        * float(np.sum(mean_headroom[list(candidate.edge_ids)]))
        for candidate in candidates
    ]
    fair_frame.to_csv(
        RESULTS / "equal_total_expected_stress_importance.csv", index=False
    )
    write_json(
        RESULTS / "equal_total_expected_stress_metadata.json",
        fair_metadata,
    )
    logging.info("Completed stress decomposition and equal-total-stress analysis")


def interval_status(lower: float, upper: float) -> int:
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def certificate_rate_row(
    network: WeightedNetwork,
    candidates: Sequence[Candidate],
    exact_baseline_losses: np.ndarray,
    approximate_baseline_losses: np.ndarray,
    exact_stressed: np.ndarray,
    approximate_stressed: np.ndarray,
    theta: float,
    epsilon: float,
    allowance_type: str,
) -> dict[str, object]:
    baseline_exact_reliability = exact_reliability(exact_baseline_losses, theta)
    baseline_interval = moment_interval(
        approximate_baseline_losses, theta, epsilon
    )
    exact_importance = np.empty(len(candidates), dtype=float)
    lower = np.empty(len(candidates), dtype=float)
    upper = np.empty(len(candidates), dtype=float)
    coverage = np.empty(len(candidates), dtype=bool)
    for index in range(len(candidates)):
        exact_importance[index] = baseline_exact_reliability - exact_reliability(
            exact_stressed[index], theta
        )
        stressed_interval = moment_interval(
            approximate_stressed[index], theta, epsilon
        )
        lower[index] = baseline_interval[0] - stressed_interval[1]
        upper[index] = baseline_interval[1] - stressed_interval[0]
        coverage[index] = (
            lower[index] - 1.0e-12
            <= exact_importance[index]
            <= upper[index] + 1.0e-12
        )
    statuses = np.asarray(
        [interval_status(lo, hi) for lo, hi in zip(lower, upper, strict=True)]
    )
    truth = sign_class(exact_importance, 1.0e-12)
    false_signs = int(np.sum((statuses != 0) & (statuses != truth)))

    strict_pair_count = 0
    certified_pair_count = 0
    false_pair_count = 0
    for first in range(len(candidates)):
        for second in range(len(candidates)):
            if first == second or exact_importance[first] <= exact_importance[second] + 1e-12:
                continue
            strict_pair_count += 1
            if lower[first] > upper[second]:
                certified_pair_count += 1
            if lower[second] > upper[first]:
                false_pair_count += 1
    eta = float(np.mean(approximate_baseline_losses))
    nu = float(np.std(approximate_baseline_losses, ddof=0))
    return {
        "allowance_type": allowance_type,
        "theta": theta,
        "gamma": (float(dpc(network, np.zeros(network.m), 1.8, n_time=64)) - theta)
        / float(dpc(network, np.zeros(network.m), 1.8, n_time=64)),
        "epsilon": epsilon,
        "baseline_loss_mean": eta,
        "baseline_loss_standard_deviation": nu,
        "candidate_count": len(candidates),
        "sign_certificate_count": int(np.sum(statuses != 0)),
        "sign_certificate_rate": float(np.mean(statuses != 0)),
        "positive_certificate_count": int(np.sum(statuses == 1)),
        "negative_certificate_count": int(np.sum(statuses == -1)),
        "false_sign_certificate_count": false_signs,
        "interval_coverage": float(np.mean(coverage)),
        "strict_pair_count": strict_pair_count,
        "ranking_certificate_count": certified_pair_count,
        "ranking_certificate_rate": certified_pair_count
        / max(strict_pair_count, 1),
        "false_ranking_certificate_count": false_pair_count,
    }


def run_certificate_map() -> None:
    logging.info("Starting analytical-certificate map")
    networks = motif_networks()
    case_names = ("cycle5", "diamond4")
    alpha_caps = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25)
    rows: list[dict[str, object]] = []
    for case_index, case_name in enumerate(case_names):
        network = networks[case_name]
        candidates = motif_candidates(network)
        d0, g, q, _ = rank_aware_coefficients(
            network, 1.8, n_time=24, n_inner=6
        )
        for cap in alpha_caps:
            alpha_bar = np.full(network.m, cap, dtype=float)
            scenarios, _ = exact_binary_scenarios(network.m, amplitude=0.06 * cap)
            exact_baseline = d0 - dpc_batch(network, scenarios, 1.8, n_time=64)
            approximate_baseline = np.asarray(quadratic_loss(scenarios, g, q))
            exact_stressed = np.empty(
                (len(candidates), len(scenarios)), dtype=float
            )
            approximate_stressed = np.empty_like(exact_stressed)
            support_epsilon = float(
                np.max(np.abs(exact_baseline - approximate_baseline))
            )
            for candidate_index, candidate in enumerate(candidates):
                stressed_scenarios = stress_scenarios(
                    scenarios, candidate.edge_ids, 0.8, alpha_bar
                )
                exact_stressed[candidate_index] = d0 - dpc_batch(
                    network, stressed_scenarios, 1.8, n_time=64
                )
                approximate_stressed[candidate_index] = quadratic_loss(
                    stressed_scenarios, g, q
                )
                support_epsilon = max(
                    support_epsilon,
                    float(
                        np.max(
                            np.abs(
                                exact_stressed[candidate_index]
                                - approximate_stressed[candidate_index]
                            )
                        )
                    ),
                )
            combined = np.concatenate(
                [exact_baseline.ravel(), exact_stressed.ravel()]
            )
            lower_quantile, upper_quantile = np.quantile(combined, [0.02, 0.98])
            theta_lower = max(0.0, float(lower_quantile))
            theta_upper = min(float(d0), float(upper_quantile))
            if theta_upper < theta_lower:
                theta_upper = theta_lower
            if np.isclose(theta_lower, theta_upper):
                theta_values = np.asarray([theta_lower])
            else:
                theta_values = np.linspace(theta_lower, theta_upper, 21)
            global_epsilon = theoretical_epsilon(
                network, 1.8, alpha_bar
            )
            for theta_index, theta in enumerate(theta_values):
                for allowance_type, epsilon in (
                    ("analytical_global", global_epsilon),
                    ("finite_support_diagnostic", support_epsilon),
                ):
                    row = certificate_rate_row(
                        network,
                        candidates,
                        exact_baseline,
                        approximate_baseline,
                        exact_stressed,
                        approximate_stressed,
                        float(theta),
                        epsilon,
                        allowance_type,
                    )
                    row.update(
                        {
                            "network": case_name,
                            "alpha_cap": cap,
                            "total_coupling_radius": float(
                                np.dot(alpha_bar, network.weights)
                            ),
                            "binary_amplitude": 0.06 * cap,
                            "stress": 0.8,
                            "threshold_index": theta_index,
                            "threshold_rule": "nonnegative_loss_quantile_grid",
                        }
                    )
                    rows.append(row)
            logging.info("Certificate map completed for %s cap %.4g", case_name, cap)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "certificate_rate_map.csv", index=False)
    summary = (
        frame.groupby(["network", "alpha_cap", "allowance_type"], as_index=False)
        .agg(
            maximum_sign_certificate_rate=("sign_certificate_rate", "max"),
            maximum_ranking_certificate_rate=("ranking_certificate_rate", "max"),
            minimum_interval_coverage=("interval_coverage", "min"),
            false_sign_certificates=("false_sign_certificate_count", "sum"),
            false_ranking_certificates=("false_ranking_certificate_count", "sum"),
            epsilon=("epsilon", "first"),
            total_coupling_radius=("total_coupling_radius", "first"),
        )
    )
    summary.to_csv(RESULTS / "certificate_rate_summary.csv", index=False)
    logging.info("Completed analytical-certificate map")


def run_runtime_expansion() -> None:
    logging.info("Starting benchmark interaction-matrix runtime expansion")
    network, _, source_weights, _ = parse_sioux_falls()
    coefficient_archive = np.load(RESULTS / "sioux_coefficients_weighted.npz")
    q = coefficient_archive["q"]
    candidates = sioux_candidates(network)
    query_candidates = candidates[:20]
    rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        for query_count in (1, 2, 5, 10, 20):
            matrix_seconds = 0.0
            matrix_values = []
            directional_seconds = 0.0
            directional_values = []
            for candidate in query_candidates[:query_count]:
                direction = np.zeros(network.m, dtype=float)
                direction[list(candidate.edge_ids)] = 1.0
                start = perf_counter()
                matrix_values.append(float(direction @ q @ direction))
                matrix_seconds += perf_counter() - start
                start = perf_counter()
                _, second = directional_loss_derivatives(
                    network,
                    direction,
                    SIOUX_HORIZON,
                    n_time=48,
                    step=6.0e-4,
                    source_weights=source_weights,
                )
                directional_seconds += perf_counter() - start
                directional_values.append(second)
            relative_error = float(
                np.linalg.norm(
                    np.asarray(directional_values) - np.asarray(matrix_values)
                )
                / max(np.linalg.norm(matrix_values), 1.0e-15)
            )
            rows.append(
                {
                    "query_count": query_count,
                    "stored_q_query_seconds": matrix_seconds,
                    "matrix_free_seconds": directional_seconds,
                    "relative_directional_error": relative_error,
                    "n": network.n,
                    "m": network.m,
                }
            )
    pd.DataFrame(rows).to_csv(
        RESULTS / "sioux_matrix_free_runtime.csv", index=False
    )

    scenarios = np.load(RESULTS / "sioux_scenarios.npz")["scenarios"]
    alpha_bar = np.full(network.m, SIOUX_ALPHA_MAX)
    g = coefficient_archive["g"]
    baseline = np.asarray(quadratic_loss(scenarios, g, q))
    qx = scenarios @ q.T
    scoring_start = perf_counter()
    checksum = 0.0
    for candidate in candidates:
        stressed = incremental_quadratic_loss(
            scenarios,
            candidate.edge_ids,
            SIOUX_STRESS,
            alpha_bar,
            g,
            q,
            baseline_losses=baseline,
            qx=qx,
        )
        checksum += float(np.sum(stressed))
    scoring_seconds = perf_counter() - scoring_start
    write_json(
        RESULTS / "sioux_scoring_runtime.json",
        {
            "candidate_count": len(candidates),
            "scenario_count": len(scenarios),
            "quadratic_candidate_scoring_seconds": scoring_seconds,
            "checksum": checksum,
        },
    )
    logging.info("Completed benchmark interaction-matrix runtime expansion")


def run_r1_audit() -> None:
    logging.info("Starting Sioux Falls and sensitivity result audit")
    pilot = json.loads((RESULTS / "r1_pilot_audit.json").read_text(encoding="utf-8"))
    sioux_candidates_frame = pd.read_csv(RESULTS / "sioux_candidates.csv")
    sioux_dpc = pd.read_csv(RESULTS / "sioux_weighted_dpc_importance.csv")
    sioux_efficiency = pd.read_csv(
        RESULTS / "sioux_weighted_global_efficiency_importance.csv"
    )
    sensitivity = pd.read_csv(RESULTS / "full_parameter_sensitivity.csv")
    sensitivity_metrics = pd.read_csv(
        RESULTS / "full_parameter_stability_metrics.csv"
    )
    dependence = pd.read_csv(
        RESULTS / "dependence_family_candidate_importance.csv"
    )
    dependence_summary = pd.read_csv(
        RESULTS / "dependence_family_summary.csv"
    )
    decomposition = pd.read_csv(RESULTS / "stress_moment_decomposition.csv")
    fair = pd.read_csv(RESULTS / "equal_total_expected_stress_importance.csv")
    certificate = pd.read_csv(RESULTS / "certificate_rate_map.csv")
    quadrature = pd.read_csv(RESULTS / "sioux_dpc_quadrature_check.csv")
    matrix_free = pd.read_csv(RESULTS / "sioux_matrix_free_runtime.csv")

    numeric_frames = (
        sioux_dpc,
        sioux_efficiency,
        sensitivity,
        dependence,
        decomposition,
        fair,
        certificate,
        quadrature,
        matrix_free,
    )
    finite = all(
        np.isfinite(
            frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all()
        for frame in numeric_frames
    )
    sensitivity_nonrank = sensitivity_metrics.drop(
        columns=["rank_spearman"]
    ).select_dtypes(include=[np.number])
    sensitivity_defined_ranks = sensitivity_metrics["rank_spearman"].dropna()
    reference_sensitivity = sensitivity[
        np.isclose(sensitivity["horizon"], 1.8)
        & np.isclose(sensitivity["stress"], 0.35)
        & np.isclose(sensitivity["gamma"], 0.9625)
    ][["candidate_id", "form", "direct_empirical_importance"]].rename(
        columns={"direct_empirical_importance": "reference_importance"}
    )
    undefined_ranks_valid = True
    for row in sensitivity_metrics[
        sensitivity_metrics["rank_spearman"].isna()
    ].itertuples(index=False):
        current = sensitivity[
            np.isclose(sensitivity["horizon"], row.horizon)
            & np.isclose(sensitivity["stress"], row.stress)
            & np.isclose(sensitivity["gamma"], row.gamma)
            & (sensitivity["form"] == row.form)
        ][["candidate_id", "form", "direct_empirical_importance"]]
        paired = current.merge(reference_sensitivity, on=["candidate_id", "form"])
        if (
            paired["direct_empirical_importance"].nunique() > 1
            and paired["reference_importance"].nunique() > 1
        ):
            undefined_ranks_valid = False
            break
    finite = bool(
        finite
        and np.isfinite(sensitivity_nonrank.to_numpy(dtype=float)).all()
        and np.isfinite(sensitivity_defined_ranks.to_numpy(dtype=float)).all()
    )
    fair_budget_spread = float(
        fair["total_expected_added_disturbance"].max()
        - fair["total_expected_added_disturbance"].min()
    )
    checks = {
        "pilot_pass": bool(pilot["pass"]),
        "sioux_candidate_count": int(len(sioux_candidates_frame)),
        "sioux_form_counts": sioux_candidates_frame.groupby("form").size().to_dict(),
        "sioux_dpc_result_count": int(len(sioux_dpc)),
        "sioux_efficiency_result_count": int(len(sioux_efficiency)),
        "sensitivity_result_count": int(len(sensitivity)),
        "sensitivity_configuration_count": int(
            sensitivity[["horizon", "stress", "gamma"]].drop_duplicates().shape[0]
        ),
        "sensitivity_candidate_count": int(sensitivity["candidate_id"].nunique()),
        "sensitivity_undefined_rank_count": int(
            sensitivity_metrics["rank_spearman"].isna().sum()
        ),
        "sensitivity_undefined_ranks_valid": bool(undefined_ranks_valid),
        "dependence_result_count": int(len(dependence)),
        "dependence_family_count": int(dependence["family"].nunique()),
        "dependence_marginal_mean_spread": float(
            dependence_summary["marginal_mean"].max()
            - dependence_summary["marginal_mean"].min()
        ),
        "dependence_marginal_sd_spread": float(
            dependence_summary["marginal_standard_deviation"].max()
            - dependence_summary["marginal_standard_deviation"].min()
        ),
        "decomposition_max_closure_error": float(
            np.max(np.abs(decomposition["closure_error"]))
        ),
        "equal_total_stress_budget_spread": fair_budget_spread,
        "equal_total_stress_minimum_lambda": float(fair["stress"].min()),
        "equal_total_stress_maximum_lambda": float(fair["stress"].max()),
        "certificate_minimum_gamma": float(certificate["gamma"].min()),
        "certificate_maximum_gamma": float(certificate["gamma"].max()),
        "certificate_nonzero_analytical_condition_count": int(
            np.sum(
                (certificate["allowance_type"] == "analytical_global")
                & (
                    (certificate["sign_certificate_rate"] > 0.0)
                    | (certificate["ranking_certificate_rate"] > 0.0)
                )
            )
        ),
        "certificate_minimum_analytical_coverage": float(
            certificate.loc[
                certificate["allowance_type"] == "analytical_global",
                "interval_coverage",
            ].min()
        ),
        "certificate_false_analytical_sign_count": int(
            certificate.loc[
                certificate["allowance_type"] == "analytical_global",
                "false_sign_certificate_count",
            ].sum()
        ),
        "certificate_false_analytical_ranking_count": int(
            certificate.loc[
                certificate["allowance_type"] == "analytical_global",
                "false_ranking_certificate_count",
            ].sum()
        ),
        "sioux_quadrature_max_absolute_difference": float(
            quadrature["absolute_difference"].max()
        ),
        "sioux_matrix_free_max_relative_error": float(
            matrix_free["relative_directional_error"].max()
        ),
        "all_numeric_outputs_finite": finite,
    }
    checks["all_required_checks_pass"] = bool(
        checks["pilot_pass"]
        and checks["sioux_candidate_count"] == 797
        and checks["sioux_form_counts"]
        == {"corridor": 739, "edge": 38, "region": 20}
        and checks["sioux_dpc_result_count"] == 797
        and checks["sioux_efficiency_result_count"] == 797
        and checks["sensitivity_result_count"] == 104 * 27
        and checks["sensitivity_configuration_count"] == 27
        and checks["sensitivity_candidate_count"] == 104
        and checks["sensitivity_undefined_ranks_valid"]
        and checks["dependence_result_count"] == 104 * 4
        and checks["dependence_family_count"] == 4
        and checks["dependence_marginal_mean_spread"] < 1.0e-12
        and checks["dependence_marginal_sd_spread"] < 1.0e-12
        and checks["decomposition_max_closure_error"] < 1.0e-12
        and checks["equal_total_stress_budget_spread"] < 1.0e-12
        and checks["equal_total_stress_minimum_lambda"] >= 0.0
        and checks["equal_total_stress_maximum_lambda"] <= 1.0
        and checks["certificate_minimum_gamma"] > 0.0
        and checks["certificate_maximum_gamma"] <= 1.0 + 1.0e-12
        and checks["certificate_nonzero_analytical_condition_count"] > 0
        and checks["certificate_minimum_analytical_coverage"] >= 1.0 - 1.0e-12
        and checks["certificate_false_analytical_sign_count"] == 0
        and checks["certificate_false_analytical_ranking_count"] == 0
        and checks["sioux_quadrature_max_absolute_difference"] < 1.0e-6
        and checks["sioux_matrix_free_max_relative_error"] < 0.02
        and checks["all_numeric_outputs_finite"]
    )
    write_json(RESULTS / "sioux_and_sensitivity_audit.json", checks)
    if not checks["all_required_checks_pass"]:
        raise RuntimeError(f"Sioux Falls and sensitivity audit failed: {checks}")
    logging.info("Sioux Falls and sensitivity result audit passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=(
            "pilot",
            "sioux",
            "sensitivity",
            "dependence",
            "decomposition",
            "certificates",
            "runtime",
            "audit",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()
    configure_logging()
    write_json(RESULTS / "r1_environment.json", environment_payload())
    start = perf_counter()
    if args.phase in {"pilot", "all"}:
        run_pilot()
    if args.phase in {"sioux", "all"}:
        run_sioux_benchmark()
    if args.phase in {"sensitivity", "all"}:
        run_full_sensitivity()
    if args.phase in {"dependence", "all"}:
        run_dependence_families()
    if args.phase in {"decomposition", "all"}:
        run_stress_decomposition_and_fairness()
    if args.phase in {"certificates", "all"}:
        run_certificate_map()
    if args.phase in {"runtime", "all"}:
        run_runtime_expansion()
    if args.phase in {"audit", "all"}:
        run_r1_audit()
    logging.info("Phase %s completed in %.3f seconds", args.phase, perf_counter() - start)


if __name__ == "__main__":
    main()
