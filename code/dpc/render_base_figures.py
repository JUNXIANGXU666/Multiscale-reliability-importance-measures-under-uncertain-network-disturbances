"""Render numerical diagnostics from saved results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

BLUE = "#3C5488"
CYAN = "#4DBBD5"
GREEN = "#00A087"
RED = "#E64B35"
ORANGE = "#F39B7F"
PURPLE = "#8491B4"
DARK = "#303030"
MID_GREY = "#777777"
LIGHT_GREY = "#D9D9D9"

DIVERGING = LinearSegmentedColormap.from_list(
    "dpc_diverging", [BLUE, "#F7F7F7", RED]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "dpc_sequential", ["#F4F7FA", CYAN, BLUE]
)

SCALE_COLOURS = {"edge": BLUE, "corridor": GREEN, "region": RED}
NETWORK_COLOURS = {
    "path4": BLUE,
    "cycle5": GREEN,
    "diamond4": RED,
    "bridge6": PURPLE,
}
NETWORK_TITLES = {
    "path4": "Path",
    "cycle5": "Cycle",
    "diamond4": "Diamond",
    "bridge6": "Bridge",
}


def configure_style() -> None:
    """Apply one restrained, Arial-based style to every output."""

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.7,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def panel_label(axis: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    axis.text(
        x,
        y,
        f"({label})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=DARK,
    )


def clean_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis:
        axis.grid(True, axis=grid_axis, color=LIGHT_GREY, linewidth=0.55, alpha=0.7)
        axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, stem: str) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / f"{stem}.png"
    pdf_path = FIGURES / f"{stem}.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return [str(png_path), str(pdf_path)]


def load_archive(name: str) -> dict[str, np.ndarray]:
    with np.load(RESULTS / name) as archive:
        return {key: archive[key].copy() for key in archive.files}


def parse_edge_ids(value: object) -> list[int]:
    return [int(item) for item in str(value).split() if item]


def draw_network(
    axis: plt.Axes,
    edges: np.ndarray,
    positions: np.ndarray,
    edge_colours: list[object] | np.ndarray,
    edge_widths: float | list[float] = 3.0,
    node_labels: bool = True,
) -> None:
    if np.isscalar(edge_widths):
        widths = [float(edge_widths)] * len(edges)
    else:
        widths = list(edge_widths)
    for edge_index, (u, v) in enumerate(edges.astype(int)):
        axis.plot(
            [positions[u, 0], positions[v, 0]],
            [positions[u, 1], positions[v, 1]],
            color=edge_colours[edge_index],
            linewidth=widths[edge_index],
            solid_capstyle="round",
            zorder=1,
        )
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=34,
        facecolor="white",
        edgecolor=DARK,
        linewidth=0.75,
        zorder=3,
    )
    if node_labels:
        for node, (x_value, y_value) in enumerate(positions):
            axis.text(
                x_value,
                y_value,
                str(node + 1),
                ha="center",
                va="center",
                fontsize=6.5,
                color=DARK,
                zorder=4,
            )
    axis.set_aspect("equal")
    axis.margins(0.22)
    axis.set_axis_off()


def coefficient_validation_figure() -> dict[str, object]:
    quadrature = pd.read_csv(RESULTS / "coefficient_quadrature_convergence.csv")
    finite_difference = pd.read_csv(
        RESULTS / "coefficient_finite_difference_validation.csv"
    )
    quadrature = quadrature.sort_values("n_quad")
    finite_difference = finite_difference.sort_values("step")

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), layout="constrained")

    axes[0].loglog(
        quadrature["n_quad"],
        quadrature["g_relative_error"],
        marker="o",
        color=BLUE,
        label=r"Direct term, $\boldsymbol{g}$",
    )
    axes[0].loglog(
        quadrature["n_quad"],
        quadrature["q_relative_error"],
        marker="s",
        color=RED,
        label=r"Interaction term, $\boldsymbol{Q}$",
    )
    axes[0].set_xlabel(r"Quadrature work, $n_{\mathrm{quad}}$")
    axes[0].set_ylabel("Relative coefficient error")
    axes[0].set_title("Rank-aware quadrature convergence")
    axes[0].legend(loc="best")
    clean_axis(axes[0], grid_axis="both")
    panel_label(axes[0], "a")

    axes[1].loglog(
        finite_difference["step"],
        finite_difference["g_relative_error"],
        marker="o",
        color=BLUE,
        label=r"Direct term, $\boldsymbol{g}$",
    )
    axes[1].loglog(
        finite_difference["step"],
        finite_difference["q_relative_error"],
        marker="s",
        color=RED,
        label=r"Interaction term, $\boldsymbol{Q}$",
    )
    axes[1].set_xlabel(r"Finite-difference step, $h$")
    axes[1].set_ylabel("Relative coefficient error")
    axes[1].set_title("Independent finite-difference check")
    axes[1].legend(loc="best")
    clean_axis(axes[1], grid_axis="both")
    panel_label(axes[1], "b")

    axes[2].scatter(
        quadrature["seconds"],
        quadrature["q_relative_error"],
        marker="o",
        s=28,
        color=GREEN,
        label="Rank-aware quadrature",
    )
    axes[2].scatter(
        finite_difference["seconds"],
        finite_difference["q_relative_error"],
        marker="s",
        s=28,
        color=PURPLE,
        label="Finite differences",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Coefficient time (s)")
    axes[2].set_ylabel(r"Relative error in $\boldsymbol{Q}$")
    axes[2].set_title("Accuracy and computation time")
    axes[2].legend(loc="best")
    clean_axis(axes[2], grid_axis="both")
    panel_label(axes[2], "c")

    outputs = save_figure(figure, "coefficient_validation")
    return {
        "stem": "coefficient_validation",
        "panels": 3,
        "sources": [
            "coefficient_quadrature_convergence.csv",
            "coefficient_finite_difference_validation.csv",
        ],
        "outputs": outputs,
    }


def motif_interaction_figure() -> dict[str, object]:
    names = ["path4", "cycle5", "diamond4", "bridge6"]
    archives = {
        name: load_archive(f"{name}_coefficients.npz") for name in names
    }
    all_g = np.concatenate([archives[name]["g"] for name in names])
    g_limit = float(np.max(np.abs(all_g)))
    g_norm = TwoSlopeNorm(vmin=-g_limit, vcenter=0.0, vmax=g_limit)

    figure = plt.figure(figsize=(11.6, 7.1), layout="constrained")
    grid = figure.add_gridspec(2, 4, height_ratios=[0.95, 1.25])
    network_axes = [figure.add_subplot(grid[0, index]) for index in range(4)]

    for index, (name, axis) in enumerate(zip(names, network_axes)):
        archive = archives[name]
        edge_colours = [DIVERGING(g_norm(value)) for value in archive["g"]]
        draw_network(
            axis,
            archive["edges"],
            archive["positions"],
            edge_colours,
            edge_widths=3.4,
        )
        positive = int(np.sum(archive["g"] > 0.0))
        negative = int(np.sum(archive["g"] < 0.0))
        axis.set_title(
            f"{NETWORK_TITLES[name]} network\n{positive} positive, {negative} negative"
        )
        panel_label(axis, chr(ord("a") + index), x=0.01, y=0.96)

    g_map = ScalarMappable(norm=g_norm, cmap=DIVERGING)
    g_map.set_array([])
    colour_bar = figure.colorbar(
        g_map,
        ax=network_axes,
        location="right",
        shrink=0.82,
        pad=0.015,
        aspect=25,
    )
    colour_bar.set_label(r"Direct coefficient, $g_e$")

    heat_axis = figure.add_subplot(grid[1, 0:2])
    diamond_q = archives["diamond4"]["q"]
    q_limit = float(np.max(np.abs(diamond_q)))
    q_norm = TwoSlopeNorm(vmin=-q_limit, vcenter=0.0, vmax=q_limit)
    heat_image = heat_axis.imshow(
        diamond_q,
        cmap=DIVERGING,
        norm=q_norm,
        origin="upper",
        interpolation="nearest",
    )
    tick_labels = [rf"$e_{{{index + 1}}}$" for index in range(diamond_q.shape[0])]
    heat_axis.set_xticks(np.arange(diamond_q.shape[0]), tick_labels)
    heat_axis.set_yticks(np.arange(diamond_q.shape[0]), tick_labels)
    heat_axis.set_xlabel("Second disturbed edge")
    heat_axis.set_ylabel("First disturbed edge")
    heat_axis.set_title("Diamond-network interaction matrix")
    heat_bar = figure.colorbar(heat_image, ax=heat_axis, shrink=0.84, pad=0.03)
    heat_bar.set_label(r"Interaction coefficient, $Q_{ef}$")
    panel_label(heat_axis, "e", x=-0.10, y=1.06)

    interaction_axis = figure.add_subplot(grid[1, 2:4])
    motif = pd.read_csv(RESULTS / "motif_candidate_losses.csv")
    corridor = motif[motif["scale"] == "corridor"].copy()
    exact_nonadditivity: list[float] = []
    for row in corridor.itertuples(index=False):
        edge_rows = motif[
            (motif["network"] == row.network) & (motif["scale"] == "edge")
        ].set_index("candidate_id")
        edge_sum = sum(
            float(edge_rows.loc[f"E{edge_id + 1:02d}", "exact_loss"])
            for edge_id in parse_edge_ids(row.edge_ids)
        )
        exact_nonadditivity.append(float(row.exact_loss) - edge_sum)
    corridor["exact_nonadditivity"] = exact_nonadditivity

    for name in names:
        subset = corridor[corridor["network"] == name]
        interaction_axis.scatter(
            subset["pair_interaction"],
            subset["exact_nonadditivity"],
            s=26,
            alpha=0.78,
            color=NETWORK_COLOURS[name],
            edgecolor="white",
            linewidth=0.35,
            label=NETWORK_TITLES[name],
        )
    combined = np.concatenate(
        [corridor["pair_interaction"].to_numpy(), corridor["exact_nonadditivity"].to_numpy()]
    )
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    padding = 0.06 * max(upper - lower, 1.0e-6)
    interaction_axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color=DARK,
        linestyle="--",
        linewidth=0.9,
        label="One-to-one line",
    )
    rank_value = float(
        spearmanr(corridor["pair_interaction"], corridor["exact_nonadditivity"]).statistic
    )
    interaction_axis.text(
        0.03,
        0.96,
        rf"$\rho_{{\mathrm{{s}}}}={rank_value:.3f}$",
        transform=interaction_axis.transAxes,
        ha="left",
        va="top",
        color=DARK,
    )
    interaction_axis.set_xlabel("Predicted pair interaction")
    interaction_axis.set_ylabel("Exact non-additivity")
    interaction_axis.set_title(r"Set interaction at $\delta=0.22$")
    interaction_axis.legend(loc="lower right", ncol=2)
    clean_axis(interaction_axis, grid_axis="both")
    panel_label(interaction_axis, "f", x=-0.10, y=1.06)

    outputs = save_figure(figure, "small_network_signs_and_interactions")
    return {
        "stem": "small_network_signs_and_interactions",
        "panels": 6,
        "sources": [
            "path4_coefficients.npz",
            "cycle5_coefficients.npz",
            "diamond4_coefficients.npz",
            "bridge6_coefficients.npz",
            "motif_candidate_losses.csv",
        ],
        "outputs": outputs,
        "interaction_spearman": rank_value,
    }


def grid_multiscale_figure() -> dict[str, object]:
    archive = load_archive("grid_experiment_arrays.npz")
    candidates = pd.read_csv(RESULTS / "grid_candidate_results.csv")
    edges = archive["edges"].astype(int)
    positions = archive["positions"]

    figure, axes = plt.subplots(1, 3, figsize=(11.6, 4.05), layout="constrained")

    edge_rows = candidates[candidates["scale"] == "edge"].copy()
    edge_values = np.zeros(len(edges))
    for row in edge_rows.itertuples(index=False):
        edge_values[int(str(row.candidate_id)[1:]) - 1] = float(row.exact_importance)
    edge_limit = float(np.max(np.abs(edge_values)))
    edge_norm = TwoSlopeNorm(vmin=-edge_limit, vcenter=0.0, vmax=edge_limit)
    draw_network(
        axes[0],
        edges,
        positions,
        [DIVERGING(edge_norm(value)) for value in edge_values],
        edge_widths=3.4,
        node_labels=False,
    )
    axes[0].set_title("Edge reliability importance")
    edge_map = ScalarMappable(norm=edge_norm, cmap=DIVERGING)
    edge_map.set_array([])
    edge_bar = figure.colorbar(edge_map, ax=axes[0], shrink=0.78, pad=0.025)
    edge_bar.set_label(r"Exact importance, $\mathcal{I}_{L}$")
    panel_label(axes[0], "a", x=-0.03, y=1.06)

    top_corridor = (
        candidates[candidates["scale"] == "corridor"]
        .nlargest(1, "exact_importance")
        .iloc[0]
    )
    selected_corridor = set(parse_edge_ids(top_corridor["edge_ids"]))
    base_colours = [LIGHT_GREY] * len(edges)
    base_widths = [1.6] * len(edges)
    draw_network(
        axes[1],
        edges,
        positions,
        base_colours,
        edge_widths=base_widths,
        node_labels=False,
    )
    for edge_id, (u, v) in enumerate(edges):
        if edge_id in selected_corridor:
            axes[1].plot(
                [positions[u, 0], positions[v, 0]],
                [positions[u, 1], positions[v, 1]],
                color=RED,
                linewidth=4.6,
                solid_capstyle="round",
                zorder=2,
            )
    axes[1].set_title(f"Highest-ranked corridor, {top_corridor['candidate_id']}")
    axes[1].text(
        0.50,
        -0.03,
        rf"Exact $\mathcal{{I}}_L={float(top_corridor['exact_importance']):.3f}$",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        color=DARK,
    )
    panel_label(axes[1], "b", x=-0.03, y=1.06)

    region_rows = candidates[candidates["scale"] == "region"].copy()
    region_values = region_rows.set_index("candidate_id")["exact_importance"].to_dict()
    region_edge_values = np.zeros(len(edges))
    for row in region_rows.itertuples(index=False):
        for edge_id in parse_edge_ids(row.edge_ids):
            region_edge_values[edge_id] = float(row.exact_importance)
    region_norm = Normalize(vmin=0.0, vmax=float(region_rows["exact_importance"].max()))
    draw_network(
        axes[2],
        edges,
        positions,
        [SEQUENTIAL(region_norm(value)) for value in region_edge_values],
        edge_widths=3.6,
        node_labels=False,
    )
    label_positions = {
        "R01": (1.0, 3.25),
        "R02": (3.0, 3.25),
        "R03": (1.0, 0.75),
        "R04": (3.0, 0.75),
    }
    for region_id, (x_value, y_value) in label_positions.items():
        axes[2].text(
            x_value,
            y_value,
            f"{region_id}\n{region_values[region_id]:.3f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=DARK,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": LIGHT_GREY,
                "linewidth": 0.6,
                "alpha": 0.92,
            },
            zorder=5,
        )
    axes[2].set_title("Regional reliability importance")
    region_map = ScalarMappable(norm=region_norm, cmap=SEQUENTIAL)
    region_map.set_array([])
    region_bar = figure.colorbar(region_map, ax=axes[2], shrink=0.78, pad=0.025)
    region_bar.set_label(r"Exact importance, $\mathcal{I}_{L}$")
    panel_label(axes[2], "c", x=-0.03, y=1.06)

    outputs = save_figure(figure, "grid_multiscale_identification")
    return {
        "stem": "grid_multiscale_identification",
        "panels": 3,
        "sources": ["grid_experiment_arrays.npz", "grid_candidate_results.csv"],
        "outputs": outputs,
        "top_corridor": str(top_corridor["candidate_id"]),
    }


def robustness_ablation_figure() -> dict[str, object]:
    ablation = pd.read_csv(RESULTS / "grid_ablation_metrics.csv")
    topology = pd.read_csv(RESULTS / "topology_rank_metrics.csv")
    networks = pd.read_csv(RESULTS / "topology_network_summary.csv")
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.15), layout="constrained")

    scale_order = ["edge", "corridor", "region"]
    method_order = [
        "first_order",
        "diagonal_second_order",
        "full_second_order",
    ]
    method_labels = {
        "first_order": "First order",
        "diagonal_second_order": "Diagonal second order",
        "full_second_order": "Full second order",
    }
    method_colours = {
        "first_order": CYAN,
        "diagonal_second_order": GREEN,
        "full_second_order": BLUE,
    }
    positions = np.arange(len(scale_order), dtype=float)
    width = 0.24
    for method_index, method in enumerate(method_order):
        subset = (
            ablation[ablation["method"] == method]
            .set_index("scale")
            .reindex(scale_order)
        )
        offset = (method_index - 1) * width
        axes[0, 0].bar(
            positions + offset,
            subset["mean_absolute_importance_error"],
            width=width * 0.88,
            color=method_colours[method],
            edgecolor="white",
            linewidth=0.5,
            label=method_labels[method],
        )
        axes[0, 1].bar(
            positions + offset,
            subset["mean_absolute_rank_error"],
            width=width * 0.88,
            color=method_colours[method],
            edgecolor="white",
            linewidth=0.5,
            label=method_labels[method],
        )
    for axis in axes[0]:
        axis.set_xticks(positions, ["Edge", "Corridor", "Region"])
        clean_axis(axis, grid_axis="y")
    axes[0, 0].set_ylabel("Mean absolute importance error")
    axes[0, 0].set_title("Interaction ablation: importance accuracy")
    axes[0, 0].legend(loc="upper left", ncol=1)
    axes[0, 1].set_ylabel("Mean absolute rank error")
    axes[0, 1].set_title("Interaction ablation: ranking accuracy")
    for method_index in range(len(method_order)):
        offset = (method_index - 1) * width
        axes[0, 1].text(
            positions[2] + offset,
            0.035,
            "0",
            ha="center",
            va="bottom",
            fontsize=7,
            color=DARK,
        )
    panel_label(axes[0, 0], "a")
    panel_label(axes[0, 1], "b")

    measure_order = [
        "full_second_order_importance",
        "first_order_importance",
        "direct_coefficient",
        "deletion_dpc_loss",
        "betweenness",
        "resistance_leverage",
        "algebraic_connectivity_loss",
    ]
    measure_labels = [
        "Full\nsecond order",
        "First\norder",
        "Direct\ncoefficient",
        "Deletion\nDPC",
        "Edge\nbetweenness",
        "Resistance\nleverage",
        "Connectivity\nloss",
    ]
    measure_colours = [BLUE, CYAN, PURPLE, ORANGE, RED, GREEN, MID_GREY]
    for axis, column, title, ylabel in [
        (
            axes[1, 0],
            "spearman",
            "Rank stability across synthetic topologies",
            "Spearman rank correlation",
        ),
        (
            axes[1, 1],
            "top5_overlap",
            "Top-five edge recovery",
            "Top-five overlap",
        ),
    ]:
        data = [
            topology[topology["measure"] == measure][column]
            .dropna()
            .to_numpy(dtype=float)
            for measure in measure_order
        ]
        boxes = axis.boxplot(
            data,
            positions=np.arange(len(measure_order)),
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": DARK, "linewidth": 1.2},
            whiskerprops={"color": MID_GREY, "linewidth": 0.8},
            capprops={"color": MID_GREY, "linewidth": 0.8},
        )
        for patch, colour in zip(boxes["boxes"], measure_colours, strict=True):
            patch.set_facecolor(colour)
            patch.set_alpha(0.72)
            patch.set_edgecolor("white")
        for measure_index, values in enumerate(data):
            jitter = (
                np.linspace(-0.14, 0.14, len(values))
                if len(values) > 1
                else np.zeros(len(values))
            )
            axis.scatter(
                measure_index + jitter,
                values,
                s=9,
                color=DARK,
                alpha=0.42,
                linewidths=0.0,
                zorder=3,
            )
        axis.set_xticks(np.arange(len(measure_order)), measure_labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        clean_axis(axis, grid_axis="y")
        saturated_count = int((~networks["ranking_defined"].astype(bool)).sum())
        axis.text(
            0.99,
            0.04,
            f"{saturated_count} saturated networks omitted",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=MID_GREY,
        )
    axes[1, 0].axhline(0.0, color=LIGHT_GREY, linewidth=0.8, zorder=0)
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 1].set_ylim(-0.03, 1.05)
    panel_label(axes[1, 0], "c")
    panel_label(axes[1, 1], "d")

    outputs = save_figure(figure, "topology_robustness_and_ablation")
    full_metrics = topology[
        topology["measure"] == "full_second_order_importance"
    ]
    return {
        "stem": "topology_robustness_and_ablation",
        "panels": 4,
        "sources": [
            "grid_ablation_metrics.csv",
            "topology_rank_metrics.csv",
            "topology_network_summary.csv",
        ],
        "outputs": outputs,
        "network_count": int(networks["network_id"].nunique()),
        "full_second_order_median_spearman": float(
            full_metrics["spearman"].median()
        ),
    }


def calibration_figure() -> dict[str, object]:
    calibration = pd.read_csv(RESULTS / "grid_scenario_calibration.csv")
    audit = json.loads((RESULTS / "numerical_audit.json").read_text(encoding="utf-8"))

    figure = plt.figure(figsize=(11.4, 6.55), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[1.12, 1.0])
    waveform_axis = figure.add_subplot(grid[0, :])
    residual_axis = figure.add_subplot(grid[1, 0])
    scatter_axis = figure.add_subplot(grid[1, 1])

    waveform_axis.plot(
        calibration["scenario_index"],
        calibration["exact_loss"],
        color=BLUE,
        linewidth=1.25,
        label="Exact loss",
        zorder=2,
    )
    waveform_axis.plot(
        calibration["scenario_index"],
        calibration["quadratic_loss"],
        color=ORANGE,
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label="Quadratic approximation",
        zorder=3,
    )
    waveform_axis.set_xlabel("Scenario index")
    waveform_axis.set_ylabel(r"Signed DPC loss, $\ell_L$")
    waveform_axis.set_title("Scenario-level exact and approximated losses")
    waveform_axis.legend(loc="upper right", ncol=2)
    clean_axis(waveform_axis, grid_axis="y")
    panel_label(waveform_axis, "a", x=-0.06, y=1.05)

    residual = calibration["residual"].to_numpy()
    scenario_index = calibration["scenario_index"].to_numpy()
    residual_axis.plot(scenario_index, residual, color=MID_GREY, linewidth=0.75)
    residual_axis.fill_between(
        scenario_index,
        0.0,
        residual,
        where=residual >= 0.0,
        color=RED,
        alpha=0.36,
        linewidth=0.0,
    )
    residual_axis.fill_between(
        scenario_index,
        0.0,
        residual,
        where=residual < 0.0,
        color=BLUE,
        alpha=0.36,
        linewidth=0.0,
    )
    residual_axis.axhline(0.0, color=DARK, linewidth=0.7)
    residual_axis.set_xlabel("Scenario index")
    residual_axis.set_ylabel(r"Residual, $\ell_L-\widehat{\ell}_L$")
    residual_axis.set_title("Approximation residual")
    clean_axis(residual_axis, grid_axis="y")
    panel_label(residual_axis, "b", x=-0.12, y=1.06)

    magnitude = calibration["disturbance_l2"].to_numpy()
    scatter = scatter_axis.scatter(
        calibration["exact_loss"],
        calibration["quadratic_loss"],
        c=magnitude,
        cmap=SEQUENTIAL,
        s=25,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    combined = np.concatenate(
        [calibration["exact_loss"].to_numpy(), calibration["quadratic_loss"].to_numpy()]
    )
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    padding = 0.04 * max(upper - lower, 1.0e-6)
    scatter_axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color=DARK,
        linestyle="--",
        linewidth=0.9,
    )
    scatter_axis.set_xlim(lower - padding, upper + padding)
    scatter_axis.set_ylim(lower - padding, upper + padding)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_xlabel(r"Exact loss, $\ell_L$")
    scatter_axis.set_ylabel(r"Quadratic loss, $\widehat{\ell}_L$")
    scatter_axis.set_title("Exact-versus-approximated calibration")
    scatter_axis.text(
        0.04,
        0.96,
        "RMSE = "
        + f"{float(audit['grid_loss_rmse']):.4f}\n"
        + rf"$\rho_{{\mathrm{{s}}}}={float(audit['grid_loss_spearman']):.4f}$",
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        color=DARK,
    )
    clean_axis(scatter_axis, grid_axis="both")
    panel_label(scatter_axis, "c", x=-0.12, y=1.06)
    magnitude_bar = figure.colorbar(scatter, ax=scatter_axis, shrink=0.79, pad=0.025)
    magnitude_bar.set_label(r"Disturbance magnitude, $\|\boldsymbol{\alpha}\|_2$")

    outputs = save_figure(figure, "quadratic_approximation_calibration")
    return {
        "stem": "quadratic_approximation_calibration",
        "panels": 3,
        "sources": ["grid_scenario_calibration.csv", "numerical_audit.json"],
        "outputs": outputs,
        "rmse": float(audit["grid_loss_rmse"]),
        "spearman": float(audit["grid_loss_spearman"]),
    }


def uncertainty_sensitivity_figure() -> dict[str, object]:
    reliability = pd.read_csv(RESULTS / "correlation_reliability_summary.csv")
    rank_stability = pd.read_csv(RESULTS / "correlation_rank_stability.csv")
    sensitivity = pd.read_csv(RESULTS / "sensitivity_results.csv")

    figure = plt.figure(figsize=(11.6, 6.9), layout="constrained")
    grid = figure.add_gridspec(2, 6, height_ratios=[0.9, 1.12])
    reliability_axis = figure.add_subplot(grid[0, 0:3])
    rank_axis = figure.add_subplot(grid[0, 3:6])
    heat_axes = [figure.add_subplot(grid[1, 2 * index : 2 * index + 2]) for index in range(3)]

    reliability_axis.plot(
        reliability["rho"],
        reliability["exact_reliability"],
        marker="o",
        color=BLUE,
        label="Exact reliability",
    )
    reliability_axis.plot(
        reliability["rho"],
        reliability["quadratic_reliability"],
        marker="s",
        linestyle=(0, (4, 2)),
        color=ORANGE,
        label="Quadratic reliability",
    )
    reliability_axis.set_ylim(0.0, 1.0)
    reliability_axis.set_xlabel(r"Correlation parameter, $\rho$")
    reliability_axis.set_ylabel(r"Reliability, $R_L(\tau)$")
    reliability_axis.set_title("Reliability under correlated disturbances")
    reliability_axis.legend(loc="lower left")
    clean_axis(reliability_axis, grid_axis="y")
    panel_label(reliability_axis, "a", x=-0.10, y=1.06)

    for scale in ["edge", "corridor", "region"]:
        subset = rank_stability[rank_stability["scale"] == scale].sort_values("rho")
        rank_axis.plot(
            subset["rho"],
            subset["rank_spearman"],
            marker={"edge": "o", "corridor": "s", "region": "^"}[scale],
            color=SCALE_COLOURS[scale],
            label=scale.capitalize(),
        )
    rank_axis.set_ylim(0.98, 1.001)
    rank_axis.set_xlabel(r"Correlation parameter, $\rho$")
    rank_axis.set_ylabel(r"Rank correlation, $\rho_{\mathrm{s}}$ (zoomed)")
    rank_axis.set_title("Within-scale ranking stability")
    rank_axis.legend(loc="lower left", ncol=3)
    clean_axis(rank_axis, grid_axis="y")
    panel_label(rank_axis, "b", x=-0.10, y=1.06)

    heat_norm = Normalize(vmin=0.0, vmax=float(sensitivity["exact_importance"].max()))
    heat_image = None
    for index, (scale, axis) in enumerate(zip(["edge", "corridor", "region"], heat_axes)):
        subset = sensitivity[sensitivity["scale"] == scale]
        pivot = (
            subset.pivot(index="gamma", columns="stress", values="exact_importance")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        heat_image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap=SEQUENTIAL,
            norm=heat_norm,
            interpolation="nearest",
        )
        axis.set_xticks(
            np.arange(len(pivot.columns)),
            [f"{value:.1f}" for value in pivot.columns],
        )
        axis.set_yticks(
            np.arange(len(pivot.index)),
            [f"{value:.2f}" for value in pivot.index],
        )
        axis.set_xlabel(r"Stress intensity, $\lambda$")
        if index == 0:
            axis.set_ylabel(r"Threshold quantile, $\gamma$")
        else:
            axis.set_yticklabels([])
        axis.set_title(f"{scale.capitalize()} importance")
        panel_label(axis, chr(ord("c") + index), x=-0.12, y=1.06)

    if heat_image is not None:
        heat_bar = figure.colorbar(
            heat_image,
            ax=heat_axes,
            location="right",
            shrink=0.86,
            pad=0.018,
            aspect=28,
        )
        heat_bar.set_label(r"Exact importance, $\mathcal{I}_{L}$")

    outputs = save_figure(figure, "correlation_and_sensitivity")
    return {
        "stem": "correlation_and_sensitivity",
        "panels": 5,
        "sources": [
            "correlation_reliability_summary.csv",
            "correlation_rank_stability.csv",
            "sensitivity_results.csv",
        ],
        "outputs": outputs,
    }


def runtime_scaling_figure() -> dict[str, object]:
    coefficient = pd.read_csv(RESULTS / "runtime_coefficient_scaling.csv")
    scoring = pd.read_csv(RESULTS / "runtime_scoring_scaling.csv")
    crossover = pd.read_csv(RESULTS / "runtime_matrix_free_crossover.csv")

    figure, axes = plt.subplots(2, 3, figsize=(11.6, 7.0), layout="constrained")
    axes = axes.ravel()

    coefficient_summary = coefficient.groupby("n")["coefficient_seconds"].agg(
        ["median", "min", "max"]
    )
    axes[0].errorbar(
        coefficient_summary.index,
        coefficient_summary["median"],
        yerr=np.vstack(
            [
                coefficient_summary["median"] - coefficient_summary["min"],
                coefficient_summary["max"] - coefficient_summary["median"],
            ]
        ),
        marker="o",
        color=BLUE,
        capsize=2.5,
        elinewidth=0.8,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Nodes, $n$")
    axes[0].set_ylabel("Coefficient time (s)")
    axes[0].set_title("Coefficient construction")
    clean_axis(axes[0], grid_axis="y")
    panel_label(axes[0], "a")

    panel_specs = [
        ("scenario_count", r"Scenarios, $M$", "b"),
        ("candidate_count", r"Candidates, $|\mathfrak{S}|$", "c"),
        ("candidate_size", r"Candidate size, $|S|$", "d"),
        ("edge_count", r"Disturbed edges, $m$", "e"),
    ]
    for axis, (experiment_axis, x_label, label) in zip(axes[1:5], panel_specs):
        for route, colour, marker in [
            ("dense", RED, "o"),
            ("sparse", GREEN, "s"),
        ]:
            subset = scoring[
                (scoring["experiment_axis"] == experiment_axis)
                & (scoring["route"] == route)
            ].sort_values("axis_value")
            axis.plot(
                subset["axis_value"],
                subset["candidate_scoring_seconds"],
                marker=marker,
                color=colour,
                label=route.capitalize(),
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Candidate-scoring time (s)")
        axis.set_title(
            {
                "scenario_count": "Scenario scaling",
                "candidate_count": "Candidate-count scaling",
                "candidate_size": "Candidate-size scaling",
                "edge_count": "Network-dimension scaling",
            }[experiment_axis]
        )
        clean_axis(axis, grid_axis="both")
        panel_label(axis, label)

    axes[1].legend(loc="best")

    axes[5].plot(
        crossover["query_count"],
        crossover["full_interaction_seconds"],
        marker="o",
        color=RED,
        label="Full interaction matrix",
    )
    axes[5].plot(
        crossover["query_count"],
        crossover["matrix_free_seconds"],
        marker="s",
        color=GREEN,
        label="Matrix-free directions",
    )
    axes[5].set_xscale("log")
    axes[5].set_yscale("log")
    axes[5].set_xlabel("Directional queries")
    axes[5].set_ylabel("Cumulative time (s)")
    axes[5].set_title("Full versus matrix-free routes")
    maximum_error = float(crossover["relative_directional_error"].max())
    axes[5].text(
        0.04,
        0.82,
        f"Maximum relative error = {100.0 * maximum_error:.2f}%",
        transform=axes[5].transAxes,
        ha="left",
        va="top",
        color=DARK,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    axes[5].legend(loc="lower right")
    clean_axis(axes[5], grid_axis="both")
    panel_label(axes[5], "f")

    outputs = save_figure(figure, "runtime_and_complexity_scaling")
    return {
        "stem": "runtime_and_complexity_scaling",
        "panels": 6,
        "sources": [
            "runtime_coefficient_scaling.csv",
            "runtime_scoring_scaling.csv",
            "runtime_matrix_free_crossover.csv",
        ],
        "outputs": outputs,
        "matrix_free_max_relative_error": maximum_error,
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOGS / "figure_generation.log", mode="a", encoding="utf-8"
            ),
        ],
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    builders = [
        coefficient_validation_figure,
        motif_interaction_figure,
        grid_multiscale_figure,
        robustness_ablation_figure,
        calibration_figure,
        uncertainty_sensitivity_figure,
        runtime_scaling_figure,
    ]
    figure_records: list[dict[str, object]] = []
    for builder in builders:
        logging.info("Generating %s", builder.__name__)
        figure_records.append(builder())

    audit = json.loads((RESULTS / "numerical_audit.json").read_text(encoding="utf-8"))
    source_paths = [
        RESULTS / source
        for record in figure_records
        for source in record["sources"]
    ]
    output_paths = [
        Path(output)
        for record in figure_records
        for output in record["outputs"]
    ]
    output_checks = {
        "all_source_files_exist": all(path.exists() for path in source_paths),
        "all_output_files_exist": all(path.exists() for path in output_paths),
        "all_output_files_nonempty": all(path.stat().st_size > 0 for path in output_paths),
        "figure_set_count": len(figure_records),
        "png_count": sum(path.suffix.lower() == ".png" for path in output_paths),
        "pdf_count": sum(path.suffix.lower() == ".pdf" for path in output_paths),
        "minimum_output_bytes": min(path.stat().st_size for path in output_paths),
    }
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "font_family": "Arial",
        "png_dpi": 600,
        "vector_format": "PDF",
        "palette": {
            "blue": BLUE,
            "cyan": CYAN,
            "green": GREEN,
            "red": RED,
            "orange": ORANGE,
            "purple": PURPLE,
        },
        "numerical_audit_pass": bool(audit["all_required_checks_pass"]),
        "output_checks": output_checks,
        "figures": figure_records,
    }
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not all(
        output_checks[key]
        for key in [
            "all_source_files_exist",
            "all_output_files_exist",
            "all_output_files_nonempty",
        ]
    ):
        raise RuntimeError("Figure output audit failed")
    logging.info(
        "Generated %d figure sets in %s",
        len(figure_records),
        Path("experiments") / FIGURES.name,
    )


if __name__ == "__main__":
    main()
