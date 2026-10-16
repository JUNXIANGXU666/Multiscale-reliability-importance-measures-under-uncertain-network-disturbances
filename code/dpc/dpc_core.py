"""Dpc core."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.stats import beta, norm, rankdata, spearmanr


DEFAULT_SEED = 20260716


@dataclass(frozen=True)
class WeightedNetwork:
    """Finite weighted undirected network with zero-based node indices."""

    name: str
    n: int
    edges: np.ndarray
    weights: np.ndarray
    positions: dict[int, tuple[float, float]]

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=int)
        weights = np.asarray(self.weights, dtype=float)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (m, 2)")
        if len(weights) != len(edges):
            raise ValueError("weights and edges must have the same length")
        if np.any(weights <= 0.0) or np.any(weights > 1.0):
            raise ValueError("weights must lie in (0, 1]")
        if np.any(edges < 0) or np.any(edges >= self.n):
            raise ValueError("edge endpoints must be valid node indices")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "weights", weights)

    @property
    def m(self) -> int:
        return int(len(self.edges))

    @property
    def h0(self) -> np.ndarray:
        h = np.zeros((self.n, self.n), dtype=float)
        for (u, v), weight in zip(self.edges, self.weights, strict=True):
            h[u, v] += weight
            h[v, u] += weight
        return h

    @property
    def edge_midpoints(self) -> np.ndarray:
        return np.asarray(
            [
                (
                    0.5 * (self.positions[int(u)][0] + self.positions[int(v)][0]),
                    0.5 * (self.positions[int(u)][1] + self.positions[int(v)][1]),
                )
                for u, v in self.edges
            ],
            dtype=float,
        )

    def hamiltonian(self, alpha: np.ndarray | None = None) -> np.ndarray:
        h = self.h0
        if alpha is None:
            return h
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.m,):
            raise ValueError(f"alpha must have shape ({self.m},)")
        for value, (u, v), weight in zip(alpha, self.edges, self.weights, strict=True):
            decrement = value * weight
            h[u, v] -= decrement
            h[v, u] -= decrement
        return h

    def as_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n))
        for edge_id, ((u, v), weight) in enumerate(
            zip(self.edges, self.weights, strict=True)
        ):
            graph.add_edge(
                int(u),
                int(v),
                weight=float(weight),
                distance=float(1.0 / weight),
                edge_id=edge_id,
            )
        return graph


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    scale: str
    edge_ids: tuple[int, ...]
    label: str

    @property
    def size(self) -> int:
        return len(self.edge_ids)


def gauss_rule(order: int, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(int(order))
    scale = 0.5 * (upper - lower)
    return scale * (nodes + 1.0) + lower, scale * weights


def propagators_from_eigendecomposition(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    times: np.ndarray | float,
) -> np.ndarray:
    values = np.atleast_1d(np.asarray(times, dtype=float))
    phase = np.exp(-1j * values[:, None] * eigenvalues[None, :])
    result = np.einsum(
        "ia,ta,ja->tij", eigenvectors, phase, eigenvectors, optimize=True
    )
    return result


def normalise_source_weights(
    source_weights: np.ndarray | None,
    node_count: int,
) -> np.ndarray:
    """Return a nonnegative unit-sum source-weight vector."""

    if source_weights is None:
        return np.full(node_count, 1.0 / node_count, dtype=float)
    values = np.asarray(source_weights, dtype=float)
    if values.shape != (node_count,):
        raise ValueError(f"source_weights must have shape ({node_count},)")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("source_weights must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("source_weights must have positive sum")
    return values / total


def dpc_from_hamiltonian(
    h: np.ndarray,
    horizon: float,
    n_time: int = 32,
    source_weights: np.ndarray | None = None,
) -> float:
    """Evaluate finite-horizon DPC by Gauss-Legendre quadrature."""

    n = h.shape[0]
    times, weights = gauss_rule(n_time, 0.0, float(horizon))
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    propagators = propagators_from_eigendecomposition(eigenvalues, eigenvectors, times)
    occupation = np.abs(propagators) ** 2
    concentration = np.sum(occupation * occupation, axis=1)
    participation = 1.0 / concentration
    source_mass = normalise_source_weights(source_weights, n)
    return float(
        np.sum(weights[:, None] * participation * source_mass[None, :]) / horizon
    )


def dpc(
    network: WeightedNetwork,
    alpha: np.ndarray,
    horizon: float,
    n_time: int = 32,
    source_weights: np.ndarray | None = None,
) -> float:
    return dpc_from_hamiltonian(
        network.hamiltonian(alpha),
        horizon,
        n_time,
        source_weights=source_weights,
    )


def dpc_batch(
    network: WeightedNetwork,
    alphas: np.ndarray,
    horizon: float,
    n_time: int = 32,
    source_weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(alphas, dtype=float)
    if values.ndim != 2 or values.shape[1] != network.m:
        raise ValueError(f"alphas must have shape (M, {network.m})")
    return np.fromiter(
        (
            dpc(
                network,
                row,
                horizon,
                n_time,
                source_weights=source_weights,
            )
            for row in values
        ),
        dtype=float,
        count=len(values),
    )


def quadratic_loss(alphas: np.ndarray, g: np.ndarray, q: np.ndarray) -> np.ndarray:
    values = np.asarray(alphas, dtype=float)
    one_dimensional = values.ndim == 1
    values = np.atleast_2d(values)
    linear = values @ g
    quadratic = 0.5 * np.einsum("bi,ij,bj->b", values, q, values, optimize=True)
    result = linear + quadratic
    return result[0] if one_dimensional else result


def incremental_quadratic_loss(
    alphas: np.ndarray,
    candidate: Sequence[int],
    stress: float,
    alpha_bar: np.ndarray,
    g: np.ndarray,
    q: np.ndarray,
    baseline_losses: np.ndarray | None = None,
    qx: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(alphas, dtype=float)
    indices = np.asarray(candidate, dtype=int)
    if baseline_losses is None:
        baseline_losses = np.asarray(quadratic_loss(values, g, q))
    if qx is None:
        qx = values @ q.T
    increments = stress * (alpha_bar[indices][None, :] - values[:, indices])
    linear_update = np.sum((g[indices][None, :] + qx[:, indices]) * increments, axis=1)
    q_sub = q[np.ix_(indices, indices)]
    quadratic_update = 0.5 * np.einsum(
        "bi,ij,bj->b", increments, q_sub, increments, optimize=True
    )
    return baseline_losses + linear_update + quadratic_update


def stress_scenarios(
    alphas: np.ndarray,
    candidate: Sequence[int],
    stress: float,
    alpha_bar: np.ndarray,
) -> np.ndarray:
    stressed = np.array(alphas, dtype=float, copy=True)
    indices = np.asarray(candidate, dtype=int)
    stressed[:, indices] += stress * (
        alpha_bar[indices][None, :] - stressed[:, indices]
    )
    return stressed


def finite_difference_coefficients(
    network: WeightedNetwork,
    horizon: float,
    n_time: int = 48,
    step: float = 1.0e-3,
    fourth_order_edges: bool = True,
    richardson_mixed: bool = False,
    source_weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Independent central-difference reference for g and Q."""

    start = perf_counter()
    m = network.m
    zero = np.zeros(m, dtype=float)
    cache: dict[tuple[float, ...], float] = {}

    def evaluate(alpha: np.ndarray) -> float:
        key = tuple(np.round(alpha, 14))
        if key not in cache:
            cache[key] = dpc(
                network,
                alpha,
                horizon,
                n_time,
                source_weights=source_weights,
            )
        return cache[key]

    d0 = evaluate(zero)
    g = np.zeros(m, dtype=float)
    q = np.zeros((m, m), dtype=float)

    for edge in range(m):
        direction = np.zeros(m, dtype=float)
        direction[edge] = 1.0
        if fourth_order_edges:
            dm2 = evaluate(-2.0 * step * direction)
            dm1 = evaluate(-step * direction)
            dp1 = evaluate(step * direction)
            dp2 = evaluate(2.0 * step * direction)
            d_first = (dm2 - 8.0 * dm1 + 8.0 * dp1 - dp2) / (12.0 * step)
            d_second = (
                -dp2 + 16.0 * dp1 - 30.0 * d0 + 16.0 * dm1 - dm2
            ) / (12.0 * step * step)
        else:
            dm1 = evaluate(-step * direction)
            dp1 = evaluate(step * direction)
            d_first = (dp1 - dm1) / (2.0 * step)
            d_second = (dp1 - 2.0 * d0 + dm1) / (step * step)
        g[edge] = -d_first
        q[edge, edge] = -d_second

    def mixed_derivative(edge: int, other: int, local_step: float) -> float:
        first = np.zeros(m, dtype=float)
        second = np.zeros(m, dtype=float)
        first[edge] = local_step
        second[other] = local_step
        return (
            evaluate(first + second)
            - evaluate(first - second)
            - evaluate(-first + second)
            + evaluate(-first - second)
        ) / (4.0 * local_step * local_step)

    for edge, other in combinations(range(m), 2):
        mixed = mixed_derivative(edge, other, step)
        if richardson_mixed:
            refined = mixed_derivative(edge, other, 0.5 * step)
            mixed = (4.0 * refined - mixed) / 3.0
        q[edge, other] = -mixed
        q[other, edge] = -mixed

    elapsed = perf_counter() - start
    timing = {
        "coefficient_seconds": elapsed,
        "dpc_evaluations": float(len(cache)),
        "n_time": float(n_time),
        "step": float(step),
    }
    return d0, g, q, timing


def rank_aware_coefficients(
    network: WeightedNetwork,
    horizon: float,
    n_time: int = 16,
    n_inner: int = 4,
    source_weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Evaluate g and Q from the ordered Fréchet integrals.

    Edge rank two is used explicitly. The implementation forms all pair
    interactions together at each quadrature configuration.
    """

    start = perf_counter()
    n = network.n
    m = network.m
    source_mass = normalise_source_weights(source_weights, n)
    edges = network.edges
    edge_weights = network.weights
    oriented_from = np.column_stack((edges[:, 0], edges[:, 1]))
    oriented_to = np.column_stack((edges[:, 1], edges[:, 0]))

    eigenvalues, eigenvectors = np.linalg.eigh(network.h0)

    def prop(times: np.ndarray | float) -> np.ndarray:
        return propagators_from_eigendecomposition(eigenvalues, eigenvectors, times)

    t_nodes, t_weights = gauss_rule(n_time, 0.0, horizon)
    s_nodes, s_weights = gauss_rule(n_inner, 0.0, 1.0)
    s1 = np.repeat(s_nodes, n_inner)
    y = np.tile(s_nodes, n_inner)
    s2 = s1 * y
    triangle_weights = np.repeat(s_weights, n_inner) * np.tile(s_weights, n_inner) * s1

    d0 = 0.0
    g = np.zeros(m, dtype=float)
    q_matrix = np.zeros((m, m), dtype=float)
    first_seconds = 0.0
    second_seconds = 0.0
    response_seconds = 0.0

    for t, outer_weight in zip(t_nodes, t_weights, strict=True):
        response_start = perf_counter()
        u0 = prop(float(t))[0]
        occupation = np.abs(u0) ** 2
        concentration = np.sum(occupation * occupation, axis=0)
        participation = 1.0 / concentration
        d0 += outer_weight * float(source_mass @ participation) / horizon
        response_seconds += perf_counter() - response_start

        first_start = perf_counter()
        left = prop((1.0 - s_nodes) * t)
        right = prop(s_nodes * t)
        u1_integral = np.zeros((m, n, n), dtype=complex)
        for orientation in (0, 1):
            left_columns = np.take(left, oriented_from[:, orientation], axis=2)
            right_rows = np.take(right, oriented_to[:, orientation], axis=1)
            u1_integral += np.einsum(
                "sie,sej,s->eij",
                left_columns,
                right_rows,
                s_weights,
                optimize=True,
            )
        u1 = 1j * t * edge_weights[:, None, None] * u1_integral
        first_seconds += perf_counter() - first_start

        response_start = perf_counter()
        p1 = 2.0 * np.real(np.conj(u0)[None, :, :] * u1)
        q1 = 2.0 * np.sum(occupation[None, :, :] * p1, axis=1)
        pi1 = -q1 / concentration[None, :] ** 2
        g -= outer_weight * (pi1 @ source_mass) / horizon
        response_seconds += perf_counter() - response_start

        second_start = perf_counter()
        left2 = prop((1.0 - s1) * t)
        middle2 = prop((s1 - s2) * t)
        right2 = prop(s2 * t)
        ordered = np.zeros((m, m, n, n), dtype=complex)
        for first_orientation in (0, 1):
            left_columns = np.take(
                left2, oriented_from[:, first_orientation], axis=2
            )
            left_columns = left_columns * edge_weights[None, None, :]
            middle_rows = np.take(
                middle2, oriented_to[:, first_orientation], axis=1
            )
            for second_orientation in (0, 1):
                middle_scalars = np.take(
                    middle_rows,
                    oriented_from[:, second_orientation],
                    axis=2,
                )
                right_rows = np.take(
                    right2, oriented_to[:, second_orientation], axis=1
                )
                right_rows = right_rows * edge_weights[None, :, None]
                ordered += np.einsum(
                    "kie,kef,kfj,k->efij",
                    left_columns,
                    middle_scalars,
                    right_rows,
                    triangle_weights,
                    optimize=True,
                )
        u2 = -t * t * (ordered + np.swapaxes(ordered, 0, 1))
        second_seconds += perf_counter() - second_start

        response_start = perf_counter()
        cross = np.einsum("eir,fir->efir", np.conj(u1), u1, optimize=True)
        p2 = 2.0 * np.real(np.conj(u0)[None, None, :, :] * u2 + cross)
        p1_product = p1[:, None, :, :] * p1[None, :, :, :]
        q2 = 2.0 * np.sum(
            p1_product + occupation[None, None, :, :] * p2,
            axis=2,
        )
        pi2 = (
            2.0
            * q1[:, None, :]
            * q1[None, :, :]
            / concentration[None, None, :] ** 3
            - q2 / concentration[None, None, :] ** 2
        )
        q_matrix -= outer_weight * np.einsum(
            "efr,r->ef", pi2, source_mass, optimize=True
        ) / horizon
        response_seconds += perf_counter() - response_start

    q_matrix = 0.5 * (q_matrix + q_matrix.T)
    elapsed = perf_counter() - start
    timing = {
        "coefficient_seconds": elapsed,
        "first_integral_seconds": first_seconds,
        "second_integral_seconds": second_seconds,
        "response_seconds": response_seconds,
        "n_time": float(n_time),
        "n_inner": float(n_inner),
        "n_quad": float(n_time * n_inner * n_inner),
    }
    return float(d0), g, q_matrix, timing


def directional_loss_derivatives(
    network: WeightedNetwork,
    direction: np.ndarray,
    horizon: float,
    n_time: int = 32,
    step: float = 1.0e-3,
    source_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Matrix-free directional first and second loss derivatives."""

    vector = np.asarray(direction, dtype=float)
    zero = np.zeros(network.m, dtype=float)
    d0 = dpc(
        network,
        zero,
        horizon,
        n_time,
        source_weights=source_weights,
    )
    plus = dpc(
        network,
        step * vector,
        horizon,
        n_time,
        source_weights=source_weights,
    )
    minus = dpc(
        network,
        -step * vector,
        horizon,
        n_time,
        source_weights=source_weights,
    )
    first = -(plus - minus) / (2.0 * step)
    second = -(plus - 2.0 * d0 + minus) / (step * step)
    return float(first), float(second)


def moment_interval(
    losses: np.ndarray,
    theta: float,
    epsilon: float,
) -> tuple[float, float, float, float]:
    values = np.asarray(losses, dtype=float)
    eta = float(np.mean(values))
    nu = float(np.std(values, ddof=0))
    lower_margin = theta - epsilon - eta
    if lower_margin > 0.0:
        lower = lower_margin * lower_margin / (nu * nu + lower_margin * lower_margin)
    else:
        lower = 0.0
    upper_margin = eta - theta - epsilon
    if upper_margin > 0.0:
        upper = nu * nu / (nu * nu + upper_margin * upper_margin)
    else:
        upper = 1.0
    return float(lower), float(upper), eta, nu


def reliability_importance_interval(
    baseline_losses: np.ndarray,
    stressed_losses: np.ndarray,
    theta: float,
    epsilon: float,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    baseline = moment_interval(baseline_losses, theta, epsilon)
    stressed = moment_interval(stressed_losses, theta, epsilon)
    lower = baseline[0] - stressed[1]
    upper = baseline[1] - stressed[0]
    return float(lower), float(upper), baseline[:2], stressed[:2]


def exact_reliability(losses: np.ndarray, theta: float) -> float:
    return float(np.mean(np.asarray(losses) <= theta))


def theoretical_epsilon(
    network: WeightedNetwork,
    horizon: float,
    alpha_bar: np.ndarray,
) -> float:
    kappa = float(np.dot(alpha_bar, network.weights))
    constant = 16.0 * network.n**2 * (1.0 + 6.0 * network.n + 6.0 * network.n**2) * horizon**3
    return float((constant / 6.0) * kappa**3)


def gaussian_copula_beta_scenarios(
    m: int,
    count: int,
    alpha_bar: np.ndarray,
    rho: float,
    seed: int,
    beta_a: float = 2.0,
    beta_b: float = 8.0,
) -> np.ndarray:
    """Bounded disturbances with a common-factor Gaussian copula."""

    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((count, 1))
    idiosyncratic = rng.standard_normal((count, m))
    latent = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idiosyncratic
    uniforms = np.clip(norm.cdf(latent), 1.0e-10, 1.0 - 1.0e-10)
    proportions = beta.ppf(uniforms, beta_a, beta_b)
    return proportions * np.asarray(alpha_bar, dtype=float)[None, :]


def exact_binary_scenarios(
    m: int,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the equally weighted binary support for a small graph."""

    integers = np.arange(2**m, dtype=np.uint64)
    bits = ((integers[:, None] >> np.arange(m, dtype=np.uint64)) & 1).astype(float)
    scenarios = amplitude * bits
    probabilities = np.full(len(scenarios), 1.0 / len(scenarios), dtype=float)
    return scenarios, probabilities


def motif_networks() -> dict[str, WeightedNetwork]:
    path = WeightedNetwork(
        "path4",
        4,
        np.array([[0, 1], [1, 2], [2, 3]]),
        np.array([0.88, 0.63, 0.92]),
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)},
    )
    cycle = WeightedNetwork(
        "cycle5",
        5,
        np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]),
        np.array([0.91, 0.66, 0.84, 0.58, 0.95]),
        {
            node: (
                float(np.cos(2.0 * np.pi * node / 5.0)),
                float(np.sin(2.0 * np.pi * node / 5.0)),
            )
            for node in range(5)
        },
    )
    diamond = WeightedNetwork(
        "diamond4",
        4,
        np.array([[0, 1], [1, 3], [0, 2], [2, 3], [1, 2]]),
        np.array([0.93, 0.72, 0.64, 0.96, 0.51]),
        {0: (0.0, 0.0), 1: (1.0, 0.8), 2: (1.0, -0.8), 3: (2.0, 0.0)},
    )
    bridge = WeightedNetwork(
        "bridge6",
        6,
        np.array(
            [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 3]]
        ),
        np.array([0.82, 0.61, 0.94, 0.47, 0.88, 0.69, 0.91]),
        {
            0: (0.0, 0.8),
            1: (0.0, -0.8),
            2: (1.0, 0.0),
            3: (2.0, 0.0),
            4: (3.0, 0.8),
            5: (3.0, -0.8),
        },
    )
    return {network.name: network for network in (path, cycle, diamond, bridge)}


def grid_network(rows: int = 5, columns: int = 5, seed: int = DEFAULT_SEED) -> WeightedNetwork:
    rng = np.random.default_rng(seed)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    positions: dict[int, tuple[float, float]] = {}

    def node(row: int, column: int) -> int:
        return row * columns + column

    for row in range(rows):
        for column in range(columns):
            positions[node(row, column)] = (float(column), float(rows - 1 - row))
    for row in range(rows):
        for column in range(columns - 1):
            edges.append((node(row, column), node(row, column + 1)))
            central = 0.13 if row == rows // 2 else 0.0
            weights.append(float(np.clip(0.62 + central + rng.normal(0.0, 0.11), 0.30, 0.98)))
    for row in range(rows - 1):
        for column in range(columns):
            edges.append((node(row, column), node(row + 1, column)))
            central = 0.15 if column == columns // 2 else 0.0
            weights.append(float(np.clip(0.60 + central + rng.normal(0.0, 0.11), 0.30, 0.98)))
    return WeightedNetwork(
        f"grid{rows}x{columns}",
        rows * columns,
        np.asarray(edges, dtype=int),
        np.asarray(weights, dtype=float),
        positions,
    )


def random_connected_network(
    n: int,
    edge_probability: float,
    seed: int,
    name: str | None = None,
) -> WeightedNetwork:
    rng = np.random.default_rng(seed)
    attempt = 0
    while True:
        graph = nx.gnp_random_graph(n, edge_probability, seed=seed + attempt)
        if nx.is_connected(graph):
            break
        attempt += 1
    edges = np.asarray(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype=int)
    weights = rng.uniform(0.35, 1.0, size=len(edges))
    layout = nx.spring_layout(graph, seed=seed, weight=None)
    positions = {node: (float(value[0]), float(value[1])) for node, value in layout.items()}
    return WeightedNetwork(name or f"er{n}", n, edges, weights, positions)


def synthetic_topology_network(
    family: str,
    n: int,
    seed: int,
    name: str | None = None,
) -> WeightedNetwork:
    """Create a connected weighted network with mean degree close to four."""

    if n < 6:
        raise ValueError("n must be at least 6")
    family_key = str(family).strip().lower()
    rng = np.random.default_rng(seed)
    if family_key == "erdos_renyi":
        probability = min(0.90, 4.0 / (n - 1))
        attempt = 0
        while True:
            graph = nx.gnp_random_graph(n, probability, seed=seed + attempt)
            if nx.is_connected(graph):
                break
            attempt += 1
    elif family_key == "small_world":
        graph = nx.connected_watts_strogatz_graph(
            n,
            k=4,
            p=0.25,
            tries=200,
            seed=seed,
        )
    elif family_key == "scale_free":
        graph = nx.barabasi_albert_graph(n, m=2, seed=seed)
    else:
        raise ValueError(
            "family must be 'erdos_renyi', 'small_world', or 'scale_free'"
        )
    edges = np.asarray(
        sorted((min(u, v), max(u, v)) for u, v in graph.edges()),
        dtype=int,
    )
    weights = np.clip(
        0.66 + rng.normal(0.0, 0.12, size=len(edges)),
        0.35,
        1.0,
    )
    layout = nx.spring_layout(graph, seed=seed, weight=None)
    positions = {
        node: (float(value[0]), float(value[1]))
        for node, value in layout.items()
    }
    return WeightedNetwork(
        name or f"{family_key}_{n}_{seed}",
        n,
        edges,
        np.asarray(weights, dtype=float),
        positions,
    )


def edge_id_lookup(network: WeightedNetwork) -> dict[tuple[int, int], int]:
    return {
        (min(int(u), int(v)), max(int(u), int(v))): edge_id
        for edge_id, (u, v) in enumerate(network.edges)
    }


def grid_candidates(network: WeightedNetwork, rows: int = 5, columns: int = 5) -> list[Candidate]:
    lookup = edge_id_lookup(network)
    candidates: list[Candidate] = []
    for edge_id in range(network.m):
        candidates.append(Candidate(f"E{edge_id + 1:02d}", "edge", (edge_id,), f"Edge {edge_id + 1}"))

    corridor_index = 1
    for row in range(rows):
        row_edges = [
            lookup[(row * columns + column, row * columns + column + 1)]
            for column in range(columns - 1)
        ]
        for length in (2, 3, 4):
            for start in range(len(row_edges) - length + 1):
                selected = tuple(row_edges[start : start + length])
                candidates.append(
                    Candidate(f"C{corridor_index:02d}", "corridor", selected, f"Corridor {corridor_index}")
                )
                corridor_index += 1
    for column in range(columns):
        column_edges = [
            lookup[(row * columns + column, (row + 1) * columns + column)]
            for row in range(rows - 1)
        ]
        for length in (2, 3, 4):
            for start in range(len(column_edges) - length + 1):
                selected = tuple(column_edges[start : start + length])
                candidates.append(
                    Candidate(f"C{corridor_index:02d}", "corridor", selected, f"Corridor {corridor_index}")
                )
                corridor_index += 1

    midpoints = network.edge_midpoints
    centre_x = 0.5 * (columns - 1)
    centre_y = 0.5 * (rows - 1)
    region_masks = [
        (midpoints[:, 0] <= centre_x) & (midpoints[:, 1] >= centre_y),
        (midpoints[:, 0] > centre_x) & (midpoints[:, 1] >= centre_y),
        (midpoints[:, 0] <= centre_x) & (midpoints[:, 1] < centre_y),
        (midpoints[:, 0] > centre_x) & (midpoints[:, 1] < centre_y),
    ]
    for region_index, mask in enumerate(region_masks, start=1):
        selected = tuple(np.flatnonzero(mask).astype(int).tolist())
        candidates.append(Candidate(f"R{region_index:02d}", "region", selected, f"Region {region_index}"))
    return candidates


def motif_candidates(network: WeightedNetwork) -> list[Candidate]:
    candidates = [
        Candidate(f"E{edge + 1:02d}", "edge", (edge,), f"Edge {edge + 1}")
        for edge in range(network.m)
    ]
    graph = network.as_networkx()
    lookup = edge_id_lookup(network)
    corridor_index = 1
    seen: set[tuple[int, ...]] = set()
    for source, target in combinations(range(network.n), 2):
        for path in nx.all_simple_paths(graph, source, target, cutoff=min(3, network.n - 1)):
            if len(path) < 3:
                continue
            edge_ids = tuple(
                lookup[(min(path[index], path[index + 1]), max(path[index], path[index + 1]))]
                for index in range(len(path) - 1)
            )
            key = tuple(sorted(edge_ids))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Candidate(f"C{corridor_index:02d}", "corridor", edge_ids, f"Corridor {corridor_index}")
            )
            corridor_index += 1
    return candidates


def classical_edge_measures(
    network: WeightedNetwork,
    horizon: float,
    n_time: int = 32,
) -> pd.DataFrame:
    graph = network.as_networkx()
    betweenness_map = nx.edge_betweenness_centrality(
        graph, normalized=True, weight="distance"
    )
    laplacian = nx.laplacian_matrix(graph, weight="weight").toarray().astype(float)
    laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    baseline_algebraic_connectivity = float(
        max(np.linalg.eigvalsh(laplacian)[1], 0.0)
    )
    bridges = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in nx.bridges(graph)
    }
    d0 = dpc(network, np.zeros(network.m), horizon, n_time)
    rows = []
    for edge_id, ((u, v), weight) in enumerate(
        zip(network.edges, network.weights, strict=True)
    ):
        key = (int(u), int(v)) if graph.has_edge(int(u), int(v)) else (int(v), int(u))
        resistance = (
            laplacian_pinv[u, u]
            + laplacian_pinv[v, v]
            - 2.0 * laplacian_pinv[u, v]
        )
        alpha = np.zeros(network.m)
        alpha[edge_id] = 1.0
        deletion_loss = d0 - dpc(network, alpha, horizon, n_time)
        reduced = graph.copy()
        reduced.remove_edge(int(u), int(v))
        if nx.is_connected(reduced):
            reduced_laplacian = (
                nx.laplacian_matrix(reduced, weight="weight")
                .toarray()
                .astype(float)
            )
            reduced_connectivity = float(
                max(np.linalg.eigvalsh(reduced_laplacian)[1], 0.0)
            )
        else:
            reduced_connectivity = 0.0
        rows.append(
            {
                "edge_id": edge_id,
                "candidate_id": f"E{edge_id + 1:02d}",
                "betweenness": float(betweenness_map[key]),
                "resistance_leverage": float(weight * resistance),
                "deletion_dpc_loss": float(deletion_loss),
                "algebraic_connectivity_loss": float(
                    baseline_algebraic_connectivity - reduced_connectivity
                ),
                "is_bridge": int(
                    (min(int(u), int(v)), max(int(u), int(v))) in bridges
                ),
            }
        )
    return pd.DataFrame(rows)


def score_candidates_quadratic(
    scenarios: np.ndarray,
    candidates: Sequence[Candidate],
    stress: float,
    alpha_bar: np.ndarray,
    g: np.ndarray,
    q: np.ndarray,
    theta: float,
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    timing: dict[str, float] = {}
    start = perf_counter()
    baseline_losses = np.asarray(quadratic_loss(scenarios, g, q))
    qx = scenarios @ q.T
    timing["scenario_preprocessing_seconds"] = perf_counter() - start
    baseline_reliability = exact_reliability(baseline_losses, theta)
    baseline_interval = moment_interval(baseline_losses, theta, epsilon)

    start = perf_counter()
    rows = []
    for candidate in candidates:
        stressed_losses = incremental_quadratic_loss(
            scenarios,
            candidate.edge_ids,
            stress,
            alpha_bar,
            g,
            q,
            baseline_losses=baseline_losses,
            qx=qx,
        )
        stressed_reliability = exact_reliability(stressed_losses, theta)
        stressed_interval = moment_interval(stressed_losses, theta, epsilon)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "scale": candidate.scale,
                "label": candidate.label,
                "size": candidate.size,
                "quadratic_importance": baseline_reliability - stressed_reliability,
                "importance_lower": baseline_interval[0] - stressed_interval[1],
                "importance_upper": baseline_interval[1] - stressed_interval[0],
                "stressed_quadratic_reliability": stressed_reliability,
            }
        )
    timing["candidate_scoring_seconds"] = perf_counter() - start
    start = perf_counter()
    frame = pd.DataFrame(rows)
    frame["screening_rank"] = frame.groupby("scale")["importance_lower"].rank(
        method="min", ascending=False
    )
    timing["sorting_seconds"] = perf_counter() - start
    timing["baseline_quadratic_reliability"] = baseline_reliability
    timing["baseline_interval_lower"] = baseline_interval[0]
    timing["baseline_interval_upper"] = baseline_interval[1]
    return frame, timing


def rank_metrics(exact: np.ndarray, approximate: np.ndarray) -> dict[str, float]:
    exact_values = np.asarray(exact, dtype=float)
    approximate_values = np.asarray(approximate, dtype=float)
    correlation = spearmanr(exact_values, approximate_values).statistic
    exact_rank = rankdata(-exact_values, method="average")
    approximate_rank = rankdata(-approximate_values, method="average")
    return {
        "spearman": float(correlation),
        "mean_absolute_rank_error": float(np.mean(np.abs(exact_rank - approximate_rank))),
    }


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def sparse_scoring_kernel(
    scenarios: np.ndarray,
    candidates: Sequence[np.ndarray],
    stress: float,
    alpha_bar: np.ndarray,
    g: np.ndarray,
    q: np.ndarray | sparse.spmatrix,
) -> dict[str, float]:
    """Time the scenario preprocessing and candidate update stages."""

    start = perf_counter()
    if sparse.issparse(q):
        qx = np.asarray(q @ scenarios.T).T
        q_dense_access = q.tocsr()
        baseline_quadratic = 0.5 * np.sum(scenarios * qx, axis=1)
    else:
        qx = scenarios @ np.asarray(q).T
        q_dense_access = np.asarray(q)
        baseline_quadratic = 0.5 * np.sum(scenarios * qx, axis=1)
    baseline = scenarios @ g + baseline_quadratic
    preprocess = perf_counter() - start

    start = perf_counter()
    means = np.empty(len(candidates), dtype=float)
    variances = np.empty(len(candidates), dtype=float)
    for index, candidate in enumerate(candidates):
        selected = np.asarray(candidate, dtype=int)
        increments = stress * (
            alpha_bar[selected][None, :] - scenarios[:, selected]
        )
        linear_update = np.sum(
            (g[selected][None, :] + qx[:, selected]) * increments,
            axis=1,
        )
        if sparse.issparse(q):
            q_sub = q_dense_access[selected][:, selected]
            quadratic_update = 0.5 * np.sum(
                increments * np.asarray(q_sub @ increments.T).T,
                axis=1,
            )
        else:
            q_sub = q_dense_access[np.ix_(selected, selected)]
            quadratic_update = 0.5 * np.einsum(
                "bi,ij,bj->b", increments, q_sub, increments, optimize=True
            )
        losses = baseline + linear_update + quadratic_update
        means[index] = np.mean(losses)
        variances[index] = np.var(losses)
    candidate_seconds = perf_counter() - start

    start = perf_counter()
    order = np.argsort(-means)
    sorting_seconds = perf_counter() - start
    return {
        "scenario_preprocessing_seconds": preprocess,
        "candidate_scoring_seconds": candidate_seconds,
        "sorting_seconds": sorting_seconds,
        "checksum": float(np.sum(means[order[: min(10, len(order))]]) + np.sum(variances)),
    }
