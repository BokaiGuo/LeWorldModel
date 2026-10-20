"""Summarize the phase-level paired orientation repair on shared roots.

The intervention is deliberately named by mechanism rather than by model:
ordered paired orientation contrast + reverse phase gate + XY shield.  It is
still a privileged state-chart experiment and is not frozen-RGB evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE = "safe_fallback"
REPAIR = "orientation_pair_reverse_phase_gate_shielded_two_pulse_macro"


def _paired(report: dict[str, Any], metric: str) -> dict[str, Any]:
    left = report["closed_loop"][BASELINE]["rows"]
    right = report["closed_loop"][REPAIR]["rows"]
    if len(left) != len(right):
        raise ValueError("arms do not contain the same number of roots")
    values = [float(b[metric]) - float(r[metric]) for b, r in zip(left, right)]
    # Positive means the repair is better for lower-is-better metrics.
    return {
        "repair_minus_baseline_raw": float(-sum(values) / len(values)),
        "repair_wins": int(sum(v > 0.0 for v in values)),
        "baseline_wins": int(sum(v < 0.0 for v in values)),
        "ties": int(sum(v == 0.0 for v in values)),
        "n": len(values),
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    base = report["closed_loop"][BASELINE]["metrics"]
    repair = report["closed_loop"][REPAIR]["metrics"]
    metrics = {}
    for name in ("pose_success", "final_xy", "final_angle", "contact_loss_rate",
                 "predicted_observed_effect_error"):
        metrics[name] = {
            "baseline": base.get(name),
            "repair": repair.get(name),
        }
    paired = {name: _paired(report, name) for name in ("final_xy", "final_angle")}
    return {
        "schema": "pusht_phase_repair_summary.v1",
        "arms": {"baseline": BASELINE, "repair": REPAIR},
        "metrics": metrics,
        "paired": paired,
        "decision": {
            "phase_repair_positive_point_estimate": bool(
                repair["pose_success"]["estimate"] > base["pose_success"]["estimate"]
                and repair["final_xy"]["estimate"] < base["final_xy"]["estimate"]
            ),
            "claim_boundary": (
                "Privileged state-chart evidence for a phase-level paired-contrast "
                "and XY-shield controller. It does not establish frozen RGB/LeWM "
                "predictor gain, causal latent physics, or universal transfer."
            ),
        },
        "source_protocol": report.get("protocol", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
