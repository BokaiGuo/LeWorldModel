"""Summarize paired PointMaze rows from a held-out RGB planning report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def summarize(report_path: Path, baseline_method: str,
              intervention_method: str) -> dict:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{report_path}: missing rows")
    by_method: dict[str, dict[int, dict]] = {}
    for row in rows:
        method = str(row["method"])
        by_method.setdefault(method, {})[int(row["episode_id"])] = row
    if baseline_method not in by_method or intervention_method not in by_method:
        raise ValueError("requested methods are absent from the report")
    common = sorted(set(by_method[baseline_method]) & set(by_method[intervention_method]))
    if not common:
        raise ValueError("methods have no common episode roots")
    baseline = np.asarray(
        [bool(by_method[baseline_method][episode]["success"]) for episode in common],
        dtype=bool,
    )
    intervention = np.asarray(
        [bool(by_method[intervention_method][episode]["success"]) for episode in common],
        dtype=bool,
    )
    baseline_only = int(np.sum(baseline & ~intervention))
    intervention_only = int(np.sum(intervention & ~baseline))
    discordant = baseline_only + intervention_only
    base_distance = np.asarray(
        [by_method[baseline_method][episode]["final_distance"] for episode in common],
        dtype=float,
    )
    intervention_distance = np.asarray(
        [by_method[intervention_method][episode]["final_distance"] for episode in common],
        dtype=float,
    )
    return {
        "schema": "gdd_wm_paired_pointmaze_consequence.v0.1",
        "protocol": {
            "report": str(report_path.resolve()),
            "baseline_method": baseline_method,
            "intervention_method": intervention_method,
            "same_episode_panel": True,
            "outcome_used_for_selection": False,
            "device": payload.get("protocol", {}).get("device"),
        },
        "episodes": common,
        "success": {
            "n": len(common),
            "baseline": int(baseline.sum()),
            "intervention": int(intervention.sum()),
            "baseline_only": baseline_only,
            "intervention_only": intervention_only,
            "paired_delta_count": int(intervention.sum() - baseline.sum()),
            "paired_delta_rate": float((intervention.astype(int) - baseline.astype(int)).mean()),
            "mcnemar_exact_p": float(
                binomtest(min(baseline_only, intervention_only), discordant, p=0.5).pvalue
                if discordant else 1.0
            ),
        },
        "final_distance": {
            "baseline_mean": float(base_distance.mean()),
            "intervention_mean": float(intervention_distance.mean()),
            "paired_delta_mean": float((intervention_distance - base_distance).mean()),
        },
        "claim_boundary": (
            "PointMaze held-out consequence only; a saturated baseline can show "
            "transfer safety without identifying incremental gain"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--intervention-method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.report, args.baseline_method, args.intervention_method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
