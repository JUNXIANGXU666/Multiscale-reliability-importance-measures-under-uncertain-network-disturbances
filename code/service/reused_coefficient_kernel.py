"""Ordered-integral coefficient kernel reused from the frozen R2 implementation."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass
class WeightedNetwork:
    n: int
    edges: np.ndarray
    weights: np.ndarray

    @property
    def m(self):
        return len(self.edges)

    @property
    def h0(self):
        return self.hamiltonian(np.zeros(self.m))

    def hamiltonian(self, alpha):
        h = np.zeros((self.n, self.n))
        weights = self.weights*(1-np.asarray(alpha))
        h[self.edges[:, 0], self.edges[:, 1]] = weights
        h[self.edges[:, 1], self.edges[:, 0]] = weights
        return h


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

