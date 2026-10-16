from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "robustness_outputs" / "results"
SOURCE = RESULTS / "kernel_candidate_importance.csv"
OUT = RESULTS / "kernel_evolution_contrasts.csv"
AUDIT = RESULTS / "kernel_evolution_contrasts_audit.json"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    subset = frame[
        (frame["weight_scale"] == 1.0)
        & (frame["stress_mode"] == "affine")
        & frame["kernel"].isin(["laplacian_quantum", "classical_diffusion"])
    ].copy()
    key = ["case", "candidate_id", "form", "size"]
    pivot = subset.pivot(index=key, columns="kernel", values="direct_empirical_importance").reset_index()
    if pivot[["laplacian_quantum", "classical_diffusion"]].isna().any().any():
        raise RuntimeError("The matched kernel table contains a missing candidate value.")

    rows: list[dict[str, object]] = []
    tol = 1.0e-12
    for (case, form), group in pivot.groupby(["case", "form"], sort=True):
        quantum = group["laplacian_quantum"].to_numpy(float)
        classical = group["classical_diffusion"].to_numpy(float)
        abs_delta = np.abs(quantum) - np.abs(classical)
        signed_delta = quantum - classical
        rows.append(
            {
                "case": case,
                "form": form,
                "candidate_count": len(group),
                "quantum_magnitude_larger": int(np.sum(abs_delta > tol)),
                "classical_magnitude_larger": int(np.sum(abs_delta < -tol)),
                "equal_magnitude": int(np.sum(np.abs(abs_delta) <= tol)),
                "median_absolute_quantum_minus_classical": float(np.median(abs_delta)),
                "median_signed_quantum_minus_classical": float(np.median(signed_delta)),
                "sign_agreement": float(
                    np.mean(np.sign(quantum) == np.sign(classical))
                ),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    audit = {
        "source": str(SOURCE.relative_to(ROOT)),
        "matched_candidate_rows": int(len(pivot)),
        "group_count": int(len(result)),
        "all_counts_close": bool(
            np.all(
                result["quantum_magnitude_larger"]
                + result["classical_magnitude_larger"]
                + result["equal_magnitude"]
                == result["candidate_count"]
            )
        ),
        "pass": True,
    }
    audit["pass"] = audit["all_counts_close"] and audit["matched_candidate_rows"] > 0
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(result.to_csv(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
