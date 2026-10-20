"""Summarize paired Cube RGB macro-transfer panels with uncertainty.

The panel runner stores paired root rows but intentionally keeps the runner
lightweight.  This read-only adjudicator computes deterministic paired
bootstrap intervals and an exact sign test without looking at any unpaired
test outcome for method selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap(values: np.ndarray, *, seed: int, draws: int = 20_000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired bootstrap requires at least two rows")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def _sign_test(wins: int, losses: int) -> float | None:
    n = int(wins + losses)
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def _arm_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    # The panel stores reduction as baseline - intervention.  Report the
    # conventional intervention - baseline delta for direct interpretation.
    distance = -np.asarray(
        [float(row["final_distance_reduction_vs_fullscale"]) for row in rows],
        dtype=np.float64,
    )
    wins = int(np.sum(distance < 0.0))
    losses = int(np.sum(distance > 0.0))
    success_delta = np.asarray(
        [float(row["success_delta_vs_fullscale"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "n": int(len(distance)),
        "intervention_minus_fullscale_mean_final_distance": float(distance.mean()),
        "bootstrap_ci95": _bootstrap(distance, seed=seed),
        "distance_wins": wins,
        "distance_losses": losses,
        "distance_ties": int(len(distance) - wins - losses),
        "distance_non_tie_improvement_rate": float(wins / max(wins + losses, 1)),
        "distance_exact_two_sided_sign_p": _sign_test(wins, losses),
        "success_intervention_only": int(np.sum(success_delta > 0.0)),
        "success_fullscale_only": int(np.sum(success_delta < 0.0)),
        "success_delta_sum": float(success_delta.sum()),
    }


def summarize(path: Path, *, seed: int = 20263099) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "live_cube_rgb_macro_paired_panel.v1":
        raise ValueError(f"unexpected panel schema in {path}")
    protocol = payload.get("protocol", {})
    if not protocol.get("root_disjoint") or protocol.get("outcome_used_for_arm_selection"):
        raise ValueError("panel does not declare root-disjoint, selection-safe protocol")
    paired = payload.get("paired_vs_fullscale_reclose", {})
    result = {
        "schema": "live_cube_rgb_macro_paired_panel_summary.v1",
        "input_panel": str(path.resolve()),
        "protocol": protocol,
        "arms": {},
        "claim_boundary": (
            "RGB-successor-only calibration and paired closed-loop consequence; "
            "not frozen LeWM teacher evidence, causal latent dynamics, or universal transfer."
        ),
    }
    for index, (arm, report) in enumerate(paired.items()):
        if arm == "fullscale_reclose":
            continue
        rows = report.get("root_rows", [])
        if not rows:
            raise ValueError(f"missing paired rows for arm {arm}")
        result["arms"][arm] = _arm_summary(rows, seed=seed + index)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20263099)
    args = parser.parse_args()
    report = summarize(args.input, seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
