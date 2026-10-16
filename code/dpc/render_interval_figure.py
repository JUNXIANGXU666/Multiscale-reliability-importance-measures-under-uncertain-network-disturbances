from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "sioux_outputs" / "results"
INPUTS = ROOT / "sioux_inputs" / "SiouxFalls"
OUTPUT = ROOT / "figures_r1"
OUTPUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
NAVY = "#33445C"
GREY = "#D3DAE3"
LIGHT_GREY = "#EEF1F5"

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E3E8EF",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def read_sioux_nodes() -> pd.DataFrame:
    frame = pd.read_csv(INPUTS / "SiouxFalls_node.tntp", sep="\t")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns={"Node": "node", "X": "x", "Y": "y"})
    frame["node"] = frame["node"].astype(int) - 1
    return frame[["node", "x", "y"]]


def sioux_benchmark_figure() -> None:
    nodes = read_sioux_nodes()
    edges = pd.read_csv(RESULTS / "sioux_physical_edges.csv")
    candidates = pd.read_csv(RESULTS / "sioux_candidates.csv")
    importance = pd.read_csv(RESULTS / "sioux_weighted_dpc_importance.csv")
    comparison = pd.read_csv(RESULTS / "sioux_rank_comparisons.csv")

    edge_candidates = candidates[candidates["form"] == "edge"][
        ["candidate_id", "edge_ids"]
    ].copy()
    edge_candidates["edge_id"] = edge_candidates["edge_ids"].astype(int)
    edge_values = edge_candidates.merge(
        importance[["candidate_id", "direct_empirical_importance"]],
        on="candidate_id",
        how="left",
    )
    edges = edges.merge(edge_values[["edge_id", "direct_empirical_importance"]], on="edge_id")
    coordinates = nodes.set_index("node")[["x", "y"]].to_dict("index")
    segments = [
        [
            (coordinates[int(row.u)]["x"], coordinates[int(row.u)]["y"]),
            (coordinates[int(row.v)]["x"], coordinates[int(row.v)]["y"]),
        ]
        for row in edges.itertuples()
    ]
    values = edges["direct_empirical_importance"].to_numpy(float)
    limit = float(np.max(np.abs(values)))

    fig = plt.figure(figsize=(10.2, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_map = fig.add_subplot(grid[0, 0])
    ax_rank = fig.add_subplot(grid[0, 1])

    collection = LineCollection(
        segments,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        linewidths=2.7,
        capstyle="round",
        zorder=2,
    )
    collection.set_array(values)
    ax_map.add_collection(collection)
    ax_map.scatter(
        nodes["x"],
        nodes["y"],
        s=18,
        facecolor="white",
        edgecolor=NAVY,
        linewidth=0.7,
        zorder=3,
    )
    ax_map.autoscale()
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title("OD-weighted edge reliability importance on Sioux Falls", pad=8)
    colour_bar = fig.colorbar(collection, ax=ax_map, orientation="horizontal", fraction=0.075, pad=0.03)
    colour_bar.set_label(r"Direct empirical importance, $\mathcal{I}_{L}$")

    rows = [
        ("DPC vs efficiency: edge", "weighted_dpc_vs_weighted_global_efficiency", "edge"),
        ("DPC vs efficiency: corridor", "weighted_dpc_vs_weighted_global_efficiency", "corridor"),
        ("DPC vs efficiency: region", "weighted_dpc_vs_weighted_global_efficiency", "region"),
        ("OD-weighted vs equal-source DPC: edge", "od_weighted_vs_equal_source_dpc", "edge"),
    ]
    labels: list[str] = []
    spearman: list[float] = []
    overlap: list[float] = []
    for label, name, form in rows:
        selected = comparison[(comparison["comparison"] == name) & (comparison["form"] == form)].iloc[0]
        labels.append(label)
        spearman.append(float(selected["spearman"]))
        overlap.append(float(selected["top5_overlap"]))
    y = np.arange(len(labels))[::-1]
    for index in range(len(labels)):
        ax_rank.plot(
            [overlap[index], spearman[index]],
            [y[index], y[index]],
            color=GREY,
            linewidth=1.4,
            zorder=1,
        )
    ax_rank.scatter(spearman, y, s=38, color=BLUE, marker="o", label=r"Spearman $r_{\mathrm{S}}$", zorder=3)
    ax_rank.scatter(overlap, y, s=34, color=ORANGE, marker="s", label="Top-five overlap", zorder=3)
    ax_rank.set_yticks(y, labels)
    ax_rank.set_xlim(-0.02, 1.03)
    ax_rank.set_xlabel("Agreement")
    ax_rank.set_title("The conventional comparator changes the ranking", pad=8)
    ax_rank.legend(loc="upper right", frameon=False)
    ax_rank.spines["top"].set_visible(False)
    ax_rank.spines["right"].set_visible(False)
    save_figure(fig, "sioux_benchmark_comparison")


def certificate_map_figure() -> None:
    frame = pd.read_csv(RESULTS / "certificate_rate_summary.csv")
    networks = [("cycle5", "Five-node cycle"), ("diamond4", "Four-node diamond")]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.75), constrained_layout=True, sharey=True)
    encodings = [
        ("analytical_global", "maximum_sign_certificate_rate", BLUE, "o", "-", "Analytical: sign"),
        ("analytical_global", "maximum_ranking_certificate_rate", BLUE, "s", "--", "Analytical: ranking"),
        ("finite_support_diagnostic", "maximum_sign_certificate_rate", ORANGE, "o", "-", "Finite support: sign"),
        ("finite_support_diagnostic", "maximum_ranking_certificate_rate", ORANGE, "s", "--", "Finite support: ranking"),
    ]
    for ax, (network, title) in zip(axes, networks, strict=True):
        selected = frame[frame["network"] == network].sort_values("total_coupling_radius")
        for allowance, metric, colour, marker, linestyle, label in encodings:
            group = selected[selected["allowance_type"] == allowance]
            ax.plot(
                group["total_coupling_radius"],
                group[metric],
                color=colour,
                marker=marker,
                markersize=4.0,
                linewidth=1.4,
                linestyle=linestyle,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title)
        ax.set_xlabel(r"Total coupling radius, $\kappa_{\mathcal{A}}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Maximum resolved fraction")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    save_figure(fig, "certificate_operating_map")


def computational_scope_figure() -> None:
    summary = json.loads((RESULTS / "r1_verified_result_summary.json").read_text(encoding="utf-8"))
    computation = summary["computation"]
    labels = [
        "Construct coefficients",
        "Direct DPC: 797 candidates",
        "Global efficiency: 797 candidates",
        r"Quadratic: $797\times128$ scores",
        "Matrix-free: 20 directions",
        r"Stored $\mathbf{Q}$: 20 queries",
    ]
    times = np.asarray(
        [
            computation["coefficient_construction_seconds"],
            computation["direct_candidate_seconds"],
            computation["conventional_candidate_seconds"],
            computation["quadratic_candidate_scoring_seconds"],
            computation["matrix_free_query_seconds_at_20"],
            computation["stored_matrix_query_seconds_at_20"],
        ]
    )
    storage = pd.read_csv(RESULTS / "interaction_matrix_storage_projection.csv")

    fig, (ax_time, ax_storage) = plt.subplots(1, 2, figsize=(10.0, 4.15), constrained_layout=True)
    y = np.arange(len(labels))[::-1]
    ax_time.barh(y, times, color=[NAVY, BLUE, "#56B4E9", GREEN, ORANGE, VERMILLION], height=0.58)
    ax_time.set_yticks(y, labels)
    ax_time.set_xscale("log")
    ax_time.set_xlabel("Measured wall time (s)")
    ax_time.set_title("Full coefficients are a reusable one-off cost")
    for yy, value in zip(y, times, strict=True):
        ax_time.text(value * 1.12, yy, f"{value:.3g}", va="center", fontsize=7.5)
    ax_time.set_xlim(times.min() / 2.0, times.max() * 4.0)
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)

    edge_count = storage["edge_count"].to_numpy(float)
    dense = storage["dense_gigabytes"].to_numpy(float)
    triangular = storage["upper_triangular_bytes"].to_numpy(float) / 1024.0**3
    ax_storage.plot(edge_count, dense, color=BLUE, marker="o", markersize=4.0, linewidth=1.5, label="Full dense matrix")
    ax_storage.plot(edge_count, triangular, color=ORANGE, marker="s", markersize=4.0, linewidth=1.5, linestyle="--", label="Packed upper triangle")
    ax_storage.set_xscale("log")
    ax_storage.set_yscale("log")
    ax_storage.set_xlabel("Potential edges, $m$")
    ax_storage.set_ylabel("Interaction storage (GiB)")
    ax_storage.set_title(r"Explicit $\mathbf{Q}$ storage grows quadratically")
    ax_storage.legend(loc="upper left", frameon=False)
    ax_storage.spines["top"].set_visible(False)
    ax_storage.spines["right"].set_visible(False)
    save_figure(fig, "interaction_cost_and_storage")


def main() -> None:
    sioux_benchmark_figure()
    certificate_map_figure()
    computational_scope_figure()
    manifest = {
        "figures": [
            "sioux_benchmark_comparison",
            "certificate_operating_map",
            "interaction_cost_and_storage",
        ],
        "source_directory": str(RESULTS),
    }
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
