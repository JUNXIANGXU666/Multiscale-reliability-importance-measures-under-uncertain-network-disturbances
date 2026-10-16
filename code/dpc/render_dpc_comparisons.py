from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
R1 = ROOT / "sioux_outputs" / "results"
R2 = ROOT / "robustness_outputs" / "results"
OUT = ROOT / "robustness_outputs" / "figures"
MANUSCRIPT = OUT
BASE_RESULTS = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLOURS = {
    "blue": "#2C6EBA",
    "orange": "#E69F00",
    "green": "#2A9D8F",
    "grey": "#6B7280",
    "red": "#C44E52",
}

LEGACY_BLUE = "#3C5488"
LEGACY_CYAN = "#4DBBD5"
LEGACY_RED = "#E64B35"
LEGACY_ORANGE = "#F39B7F"
LEGACY_DARK = "#303030"
LEGACY_GREY = "#777777"
LEGACY_LIGHT = "#D9D9D9"
LEGACY_DIVERGING = LinearSegmentedColormap.from_list(
    "dpc_diverging_r2", [LEGACY_BLUE, "#F7F7F7", LEGACY_RED]
)
LEGACY_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "dpc_sequential_r2", ["#F4F7FA", LEGACY_CYAN, LEGACY_BLUE]
)


def save(fig: plt.Figure, name: str) -> None:
    for target in (OUT / name, MANUSCRIPT / name):
        fig.savefig(target, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_with_audit_png(fig: plt.Figure, name: str) -> None:
    """Save a vector manuscript figure and a high-resolution audit copy."""

    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / Path(name).with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(MANUSCRIPT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float, y: float) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=LEGACY_DARK,
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=LEGACY_LIGHT, linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)


def draw_network(
    ax: plt.Axes,
    edges: np.ndarray,
    positions: np.ndarray,
    edge_colours: list[object] | np.ndarray,
    edge_widths: float | list[float],
) -> None:
    widths = (
        [float(edge_widths)] * len(edges)
        if np.isscalar(edge_widths)
        else list(edge_widths)
    )
    for edge_index, (u, v) in enumerate(edges.astype(int)):
        ax.plot(
            [positions[u, 0], positions[v, 0]],
            [positions[u, 1], positions[v, 1]],
            color=edge_colours[edge_index],
            linewidth=widths[edge_index],
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=30,
        facecolor="white",
        edgecolor=LEGACY_DARK,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_aspect("equal")
    ax.margins(0.22)
    ax.set_axis_off()


def parse_edge_ids(value: object) -> list[int]:
    return [int(item) for item in str(value).split() if item]


def half_up_3(value: float) -> str:
    """Format manuscript labels to three decimals using conventional half-up rounding."""

    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def grid_multiscale_identification() -> None:
    """Regenerate Figure 4 from archived outputs with consistent numerical rounding."""

    with np.load(BASE_RESULTS / "grid_experiment_arrays.npz") as archive:
        edges = archive["edges"].astype(int).copy()
        positions = archive["positions"].copy()
    candidates = pd.read_csv(BASE_RESULTS / "grid_candidate_results.csv")

    with mpl.rc_context({"font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.5}):
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.05), layout="constrained")

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
            [LEGACY_DIVERGING(edge_norm(value)) for value in edge_values],
            3.4,
        )
        axes[0].set_title("Edge reliability importance")
        edge_map = ScalarMappable(norm=edge_norm, cmap=LEGACY_DIVERGING)
        edge_map.set_array([])
        edge_bar = fig.colorbar(edge_map, ax=axes[0], shrink=0.78, pad=0.025)
        edge_bar.set_label(r"Exact importance, $\mathcal{I}_{L}$")
        panel_label(axes[0], "a", -0.03, 1.06)

        top_corridor = (
            candidates[candidates["scale"] == "corridor"]
            .nlargest(1, "exact_importance")
            .iloc[0]
        )
        selected_corridor = set(parse_edge_ids(top_corridor["edge_ids"]))
        draw_network(
            axes[1],
            edges,
            positions,
            [LEGACY_LIGHT] * len(edges),
            1.6,
        )
        for edge_id, (u, v) in enumerate(edges):
            if edge_id in selected_corridor:
                axes[1].plot(
                    [positions[u, 0], positions[v, 0]],
                    [positions[u, 1], positions[v, 1]],
                    color=LEGACY_RED,
                    linewidth=4.6,
                    solid_capstyle="round",
                    zorder=2,
                )
        axes[1].set_title(f"Highest-ranked corridor, {top_corridor['candidate_id']}")
        axes[1].text(
            0.50,
            -0.03,
            rf"Exact $\mathcal{{I}}_L={half_up_3(float(top_corridor['exact_importance']))}$",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            color=LEGACY_DARK,
        )
        panel_label(axes[1], "b", -0.03, 1.06)

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
            [LEGACY_SEQUENTIAL(region_norm(value)) for value in region_edge_values],
            3.6,
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
                f"{region_id}\n{half_up_3(float(region_values[region_id]))}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=LEGACY_DARK,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": LEGACY_LIGHT,
                    "linewidth": 0.6,
                    "alpha": 0.92,
                },
                zorder=5,
            )
        axes[2].set_title("Regional reliability importance")
        region_map = ScalarMappable(norm=region_norm, cmap=LEGACY_SEQUENTIAL)
        region_map.set_array([])
        region_bar = fig.colorbar(region_map, ax=axes[2], shrink=0.78, pad=0.025)
        region_bar.set_label(r"Exact importance, $\mathcal{I}_{L}$")
        panel_label(axes[2], "c", -0.03, 1.06)

        save_with_audit_png(fig, "grid_multiscale_identification.pdf")


def quadratic_approximation_calibration() -> None:
    """Regenerate Figure 5 with the manuscript's Spearman notation."""

    calibration = pd.read_csv(BASE_RESULTS / "grid_scenario_calibration.csv")
    import json

    audit = json.loads((BASE_RESULTS / "numerical_audit.json").read_text(encoding="utf-8"))
    with mpl.rc_context({"font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.5}):
        fig = plt.figure(figsize=(11.4, 6.55), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0])
        waveform_axis = fig.add_subplot(grid[0, :])
        residual_axis = fig.add_subplot(grid[1, 0])
        scatter_axis = fig.add_subplot(grid[1, 1])

        waveform_axis.plot(calibration["scenario_index"], calibration["exact_loss"], color=LEGACY_BLUE, linewidth=1.25, label="Exact loss", zorder=2)
        waveform_axis.plot(calibration["scenario_index"], calibration["quadratic_loss"], color=LEGACY_ORANGE, linewidth=1.0, linestyle=(0, (4, 2)), label="Quadratic approximation", zorder=3)
        waveform_axis.set_xlabel("Scenario index")
        waveform_axis.set_ylabel(r"Signed DPC loss, $\ell_L$")
        waveform_axis.set_title("Scenario-level exact and approximated losses", loc="left")
        waveform_axis.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
        )
        clean_axis(waveform_axis, "y")
        panel_label(waveform_axis, "a", -0.06, 1.05)

        residual = calibration["residual"].to_numpy()
        scenario_index = calibration["scenario_index"].to_numpy()
        residual_axis.plot(scenario_index, residual, color=LEGACY_GREY, linewidth=0.75)
        residual_axis.fill_between(scenario_index, 0.0, residual, where=residual >= 0.0, color=LEGACY_RED, alpha=0.36, linewidth=0.0)
        residual_axis.fill_between(scenario_index, 0.0, residual, where=residual < 0.0, color=LEGACY_BLUE, alpha=0.36, linewidth=0.0)
        residual_axis.axhline(0.0, color=LEGACY_DARK, linewidth=0.7)
        residual_axis.set_xlabel("Scenario index")
        residual_axis.set_ylabel(r"Residual, $\ell_L-\widehat{\ell}_L$")
        residual_axis.set_title("Approximation residual")
        clean_axis(residual_axis, "y")
        panel_label(residual_axis, "b", -0.12, 1.06)

        magnitude = calibration["disturbance_l2"].to_numpy()
        scatter = scatter_axis.scatter(
            calibration["exact_loss"],
            calibration["quadratic_loss"],
            c=magnitude,
            cmap=LEGACY_SEQUENTIAL,
            s=20,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3,
        )
        combined = np.concatenate([calibration["exact_loss"].to_numpy(), calibration["quadratic_loss"].to_numpy()])
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        padding = 0.04 * max(upper - lower, 1.0e-6)
        scatter_axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color=LEGACY_DARK, linestyle="--", linewidth=0.9)
        scatter_axis.set_xlim(lower - padding, upper + padding)
        scatter_axis.set_ylim(lower - padding, upper + padding)
        scatter_axis.set_aspect("equal", adjustable="box")
        scatter_axis.set_xlabel(r"Exact loss, $\ell_L$")
        scatter_axis.set_ylabel(r"Quadratic loss, $\widehat{\ell}_L$")
        scatter_axis.set_title("Exact-versus-approximated calibration")
        scatter_axis.text(
            0.04,
            0.96,
            "RMSE = " + f"{float(audit['grid_loss_rmse']):.4f}\n" + rf"$r_{{\mathrm{{S}}}}={float(audit['grid_loss_spearman']):.4f}$",
            transform=scatter_axis.transAxes,
            ha="left",
            va="top",
            color=LEGACY_DARK,
        )
        clean_axis(scatter_axis, "both")
        panel_label(scatter_axis, "c", -0.12, 1.06)
        magnitude_bar = fig.colorbar(scatter, ax=scatter_axis, shrink=0.79, pad=0.025)
        magnitude_bar.set_label(r"Disturbance magnitude, $\|\boldsymbol{\alpha}\|_2$")

        save_with_audit_png(fig, "quadratic_approximation_calibration.pdf")


def sioux_agreement() -> None:
    data = pd.read_csv(R1 / "sioux_rank_comparisons.csv")
    dpc_eff = data[data["comparison"] == "weighted_dpc_vs_weighted_global_efficiency"].copy()
    dpc_eff["form"] = pd.Categorical(dpc_eff["form"], ["edge", "corridor", "region"], ordered=True)
    dpc_eff = dpc_eff.sort_values("form")
    weighted = data[data["comparison"] == "od_weighted_vs_equal_source_dpc"].iloc[0]

    labels = ["Edge", "Corridor", "Region", "OD vs equal\nsource (edge)"]
    spearman = list(dpc_eff["spearman"]) + [float(weighted["spearman"])]
    overlap = list(dpc_eff["top5_overlap"]) + [float(weighted["top5_overlap"])]
    colours = [COLOURS["blue"]] * 3 + [COLOURS["orange"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.35), constrained_layout=True)
    for ax, values, title, ylabel in (
        (axes[0], spearman, "(a) Rank association", "Spearman correlation"),
        (axes[1], overlap, "(b) Leading-set agreement", "Top-five overlap"),
    ):
        bars = ax.bar(x, values, width=0.64, color=colours, edgecolor="white", linewidth=0.7)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "sioux_benchmark_comparison_r2.pdf")


def kernel_comparison() -> None:
    data = pd.read_csv(R2 / "kernel_rank_comparisons.csv")
    forms = ["edge", "corridor", "region"]
    cases = ["grid", "sioux_falls"]
    labels = ["Grid\nedge", "Grid\ncorridor", "Grid\nregion", "Sioux Falls\nedge", "Sioux Falls\ncorridor", "Sioux Falls\nregion"]
    specs = [
        ("adjacency_quantum", "laplacian_quantum", "(a) Adjacency CTQW versus Laplacian CTQW"),
        ("laplacian_quantum", "classical_diffusion", "(b) Laplacian CTQW versus classical diffusion"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.55), constrained_layout=True)
    for ax, (reference, comparison, title) in zip(axes, specs):
        rows = []
        for case in cases:
            for form in forms:
                row = data[
                    (data["case"] == case)
                    & (data["form"] == form)
                    & (data["reference_kernel"] == reference)
                    & (data["comparison_kernel"] == comparison)
                ].iloc[0]
                rows.append(row)
        x = np.arange(len(rows))
        rho = np.array([float(row["spearman"]) for row in rows])
        sign = np.array([float(row["sign_agreement"]) for row in rows])
        ax.bar(x, rho, width=0.62, color=COLOURS["blue"], alpha=0.9, label="Spearman correlation")
        ax.scatter(x, sign, s=26, marker="D", color=COLOURS["orange"], edgecolor="white", linewidth=0.45, zorder=3, label="Sign agreement")
        ax.axhline(0, color="#374151", linewidth=0.7)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylim(-0.08, 1.05)
        ax.set_ylabel("Agreement")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=2, frameon=False)
    save(fig, "kernel_comparison.pdf")


def runtime_scaling() -> None:
    data = pd.read_csv(R2 / "runtime_coefficient_scaling_extended.csv")
    grouped = data.groupby("m", as_index=False).agg(
        time_median=("coefficient_seconds", "median"),
        time_min=("coefficient_seconds", "min"),
        time_max=("coefficient_seconds", "max"),
    ).sort_values("m")
    measured = data[data["source"] == "r2_measured_after_warmup"].copy()
    memory = measured.groupby("m", as_index=False)["peak_working_set_bytes"].median().sort_values("m")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.34, wspace=0.30)
    ax = axes[0]
    yerr = np.vstack((grouped["time_median"] - grouped["time_min"], grouped["time_max"] - grouped["time_median"]))
    ax.errorbar(grouped["m"], grouped["time_median"], yerr=yerr, fmt="o", markersize=3.5, color=COLOURS["blue"], ecolor=COLOURS["grey"], capsize=2.5, label="Measured median and range")
    ax.set_yscale("log")
    ax.set_xlabel("Number of edges, $m$")
    ax.set_ylabel("Coefficient time (s)")
    ax.set_title("(a) Measured coefficient construction", loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#D1D5DB", linewidth=0.55, alpha=0.65)
    ax.grid(True, which="minor", color="#D1D5DB", linewidth=0.40, alpha=0.30)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.29), borderaxespad=0)

    ax = axes[1]
    ax.scatter(memory["m"], memory["peak_working_set_bytes"] / 2**30, color=COLOURS["red"], s=16, label="Measured peak process memory")
    m_proj = np.logspace(2, np.log10(50000), 120)
    full_gib = 8 * m_proj**2 / 2**30
    packed_gib = 4 * m_proj * (m_proj + 1) / 2**30
    ax.plot(m_proj, full_gib, color=COLOURS["blue"], linewidth=1.4, label="Projected full $Q$")
    ax.plot(m_proj, packed_gib, color=COLOURS["green"], linewidth=1.4, linestyle="--", label="Projected packed $Q$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of edges, $m$")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title("(b) Measured and projected memory", loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#D1D5DB", linewidth=0.55, alpha=0.65)
    ax.grid(True, which="minor", color="#D1D5DB", linewidth=0.40, alpha=0.30)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.29), borderaxespad=0)
    fig.canvas.draw()
    for panel in axes:
        assert not panel.get_legend().get_window_extent().overlaps(panel.get_window_extent())
    save_with_audit_png(fig, "interaction_cost_and_storage_r2.pdf")


if __name__ == "__main__":
    grid_multiscale_identification()
    quadratic_approximation_calibration()
    sioux_agreement()
    kernel_comparison()
    runtime_scaling()
