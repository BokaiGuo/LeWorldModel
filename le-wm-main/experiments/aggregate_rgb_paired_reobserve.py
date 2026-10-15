"""Aggregate root-disjoint RGB paired finite-difference audits.

The estimand is fixed before reading the reports:

    paired_delta = MAE(action-successor - zero-successor)
                   - MAE(action-successor - source)

Negative values mean that the matched zero-action successor reduced the
effect-estimation error.  The script only aggregates already completed runs;
it has no policy-selection or test-set tuning authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    return parser.parse_args()


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    args = parse_args()
    reports: list[dict[str, Any]] = []
    for path in args.inputs:
        payload = json.loads(path.read_text())
        reports.append(payload)
    if not reports:
        raise ValueError("at least one report is required")
    protocol = reports[0]["protocol"]
    for report in reports[1:]:
        other = report["protocol"]
        for key in ("fan_size", "pulse_magnitude", "action_repeat", "paired_zero_action", "device"):
            if other.get(key) != protocol.get(key):
                raise ValueError(
                    f"protocol mismatch for {key}: {other.get(key)!r} != {protocol.get(key)!r}"
                )
    deltas = np.asarray([
        float(report["effect"]["paired_effect_mae_delta_vs_source"])
        for report in reports
    ], dtype=np.float64)
    paired_mae = np.asarray([
        float(report["effect"]["visual_angle_effect_mae_rad"])
        for report in reports
    ], dtype=np.float64)
    source_mae = np.asarray([
        float(report["effect"]["source_to_action_angle_effect_mae_rad"])
        for report in reports
    ], dtype=np.float64)
    top1 = np.asarray([
        float(report["ranking"]["visual_reobserved_top1"])
        for report in reports
    ], dtype=np.float64)
    delta_ci = bootstrap_mean(deltas, args.bootstrap, 20261015)
    report: dict[str, Any] = {
        "estimand": "paired_zero_successor_effect_mae_minus_source_effect_mae",
        "claim_boundary": (
            "paired RGB finite-difference effect observability only; not online planning "
            "or universal world-model evidence"
        ),
        "protocol": {
            key: protocol[key]
            for key in ("fan_size", "pulse_magnitude", "action_repeat", "paired_zero_action", "device")
        },
        "n_seeds": int(len(reports)),
        "seeds": [int(report["protocol"]["seed"]) for report in reports],
        "paired_effect_mae_rad": paired_mae.tolist(),
        "source_effect_mae_rad": source_mae.tolist(),
        "paired_delta_rad": deltas.tolist(),
        "paired_delta_mean_rad": float(deltas.mean()),
        "paired_delta_bootstrap_95ci_rad": list(delta_ci),
        "paired_improves_seed_count": int(np.sum(deltas < 0.0)),
        "top1": top1.tolist(),
        "top1_mean": float(top1.mean()),
        "top1_random_baseline": float(1.0 / protocol["fan_size"]),
        "inputs": [str(path) for path in args.inputs],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rgb_paired_reobserve_panel.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# RGB paired finite-difference panel", "",
        "| metric | value |", "|---|---:|",
        f"| seeds | {report['n_seeds']} |",
        f"| pulse magnitude | {protocol['pulse_magnitude']} |",
        f"| action repeat | {protocol['action_repeat']} |",
        f"| paired effect MAE mean (rad) | {paired_mae.mean():.6f} |",
        f"| source effect MAE mean (rad) | {source_mae.mean():.6f} |",
        f"| paired minus source delta mean (rad) | {deltas.mean():.6f} |",
        f"| paired delta bootstrap 95% CI (rad) | [{delta_ci[0]:.6f}, {delta_ci[1]:.6f}] |",
        f"| seeds improved | {int(np.sum(deltas < 0.0))}/{len(deltas)} |",
        f"| RGB successor top-1 mean | {top1.mean():.4f} |",
        f"| random top-1 | {1.0 / protocol['fan_size']:.4f} |",
        "", report["claim_boundary"], "",
    ]
    (args.output / "rgb_paired_reobserve_panel.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
