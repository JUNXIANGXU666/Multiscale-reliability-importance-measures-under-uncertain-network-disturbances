"""Check horizon sensitivity and interval results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_RESULTS = HERE / "results"
R1_RESULTS = HERE / "sioux_outputs" / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    baseline = pd.read_csv(R1_RESULTS / "baseline_propagation_scale_audit.csv")
    horizon_candidates = pd.read_csv(
        R1_RESULTS / "horizon_scale_candidate_importance.csv"
    )
    horizon_summary = pd.read_csv(R1_RESULTS / "horizon_scale_sensitivity.csv")
    certificate = pd.read_csv(R1_RESULTS / "certificate_rate_map.csv")
    motifs = pd.read_csv(BASE_RESULTS / "motif_candidate_losses.csv")

    require(len(baseline) == 33, "The baseline propagation audit must contain 33 networks.")
    require(
        set(horizon_candidates["case"]) == {"grid", "sioux_falls"},
        "The horizon audit must contain both declared cases.",
    )
    counts = (
        horizon_candidates.groupby(["case", "horizon"])["candidate_id"]
        .nunique()
        .to_dict()
    )
    require(set(v for (case, _), v in counts.items() if case == "grid") == {104},
            "Every grid horizon must contain all 104 candidates.")
    require(set(v for (case, _), v in counts.items() if case == "sioux_falls") == {797},
            "Every Sioux Falls horizon must contain all 797 candidates.")
    require(len(horizon_summary) == 18,
            "The horizon summary must contain three forms at three horizons for two cases.")

    analytical = certificate[certificate["allowance_type"] == "analytical_global"]
    require(len(analytical) == 420,
            "The analytical certificate map must contain 420 physical conditions.")
    require(float(analytical["interval_coverage"].min()) == 1.0,
            "Every analytical interval must cover its direct empirical target.")
    require(int(analytical["false_sign_certificate_count"].sum()) == 0,
            "No analytical sign certificate may be false.")
    require(int(analytical["false_ranking_certificate_count"].sum()) == 0,
            "No analytical ranking certificate may be false.")
    resolved = analytical[
        (analytical["sign_certificate_count"] > 0)
        | (analytical["ranking_certificate_count"] > 0)
    ]
    require(len(resolved) == 22,
            "Exactly 22 analytical conditions must resolve a sign or ranking.")

    bridge_edges = motifs[(motifs["network"] == "bridge6") & (motifs["scale"] == "edge")]
    bridge_edges = bridge_edges.sort_values("exact_loss", ascending=False)
    require(bridge_edges.iloc[0]["candidate_id"] == "E04",
            "The unique inter-community bridge must rank first in the bridge motif.")
    bridge_ratio = float(
        bridge_edges.iloc[0]["exact_loss"] / bridge_edges.iloc[1]["exact_loss"]
    )

    numeric_frames = [baseline, horizon_candidates, horizon_summary, certificate, motifs]
    finite = all(
        np.isfinite(frame.select_dtypes(include=[np.number]).dropna().to_numpy()).all()
        for frame in numeric_frames
    )
    require(finite, "Every reported numeric output must be finite when defined.")

    baseline_index = baseline.set_index("network")
    report = {
        "all_checks_pass": True,
        "baseline_network_count": int(len(baseline)),
        "grid_primary_dpc": float(baseline_index.loc["grid5x5", "dpc_declared_sources"]),
        "grid_primary_horizon_to_traversal_ratio": float(
            baseline_index.loc["grid5x5", "horizon_to_traversal_ratio"]
        ),
        "sioux_primary_dpc": float(
            baseline_index.loc["sioux_falls_tntp", "dpc_declared_sources"]
        ),
        "sioux_primary_horizon_to_traversal_ratio": float(
            baseline_index.loc["sioux_falls_tntp", "horizon_to_traversal_ratio"]
        ),
        "grid_candidate_count_per_horizon": 104,
        "sioux_candidate_count_per_horizon": 797,
        "horizon_candidate_result_count": int(len(horizon_candidates)),
        "analytical_condition_count": int(len(analytical)),
        "analytical_resolved_condition_count": int(len(resolved)),
        "analytical_minimum_coverage": float(analytical["interval_coverage"].min()),
        "analytical_false_sign_count": int(
            analytical["false_sign_certificate_count"].sum()
        ),
        "analytical_false_ranking_count": int(
            analytical["false_ranking_certificate_count"].sum()
        ),
        "bridge_candidate": str(bridge_edges.iloc[0]["candidate_id"]),
        "bridge_loss_to_second_edge_ratio": bridge_ratio,
    }
    (R1_RESULTS / "final_extension_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
