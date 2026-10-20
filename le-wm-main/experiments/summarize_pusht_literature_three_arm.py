"""Summarize the matched PushT local-transfer arms.

This report intentionally uses the names ``state_chart_predictor``,
``state_only_fallback`` and ``adaptive_local_effect``.  The source runner is
privileged-state: none of these rows is frozen RGB/LeWM evidence.  The script
computes paired arm deltas on the same held-out roots and emits a claim
boundary whenever the predictor comparison is not positive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARM_MAP = {
    "state_chart_predictor": "mode_aware_lookahead",
    "state_only_fallback": "safe_fallback",
    "adaptive_local_effect": "adaptive_online",
}


def _estimate(report: dict[str, Any], arm: str, metric: str) -> float:
    return float(report["closed_loop"][ARM_MAP[arm]]["metrics"][metric]["estimate"])


def _ci(report: dict[str, Any], arm: str, metric: str) -> list[float]:
    row = report["closed_loop"][ARM_MAP[arm]]["metrics"][metric]
    return [float(row["ci95_low"]), float(row["ci95_high"])]


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    metrics = ("pose_success", "final_xy", "final_angle", "contact_loss_rate",
               "predicted_observed_effect_error")
    arms = {
        arm: {
            metric: {"estimate": _estimate(report, arm, metric),
                     "ci95": _ci(report, arm, metric)}
            for metric in metrics
        }
        for arm in ARM_MAP
    }
    # Positive values mean the second arm is better for all metrics whose
    # direction is explicitly encoded below.
    lower_is_better = {"final_xy", "final_angle", "contact_loss_rate",
                       "predicted_observed_effect_error"}
    higher_is_better = {"pose_success"}
    deltas: dict[str, dict[str, float]] = {}
    for left, right, name in (
        ("state_only_fallback", "state_chart_predictor", "predictor_minus_fallback"),
        ("state_only_fallback", "adaptive_local_effect", "repair_minus_fallback"),
        ("state_chart_predictor", "adaptive_local_effect", "repair_minus_predictor"),
    ):
        deltas[name] = {}
        for metric in metrics:
            raw = arms[right][metric]["estimate"] - arms[left][metric]["estimate"]
            deltas[name][metric] = float(raw if metric in higher_is_better else -raw)
    predictor_gain = deltas["predictor_minus_fallback"]
    repair_gain = deltas["repair_minus_fallback"]
    predictor_positive = predictor_gain["pose_success"] > 0 and all(
        predictor_gain[metric] >= 0 for metric in ("final_xy", "final_angle", "contact_loss_rate")
    )
    repair_positive = repair_gain["pose_success"] > 0 and all(
        repair_gain[metric] >= 0 for metric in ("final_xy", "final_angle", "contact_loss_rate")
    )
    return {
        "schema": "pusht_literature_three_arm_summary.v1",
        "source_protocol": report.get("protocol", {}),
        "source_device": report.get("protocol", {}).get("resolved_device"),
        "n_held_out_roots": int(report["closed_loop"]["mode_aware_lookahead"]["metrics"]["pose_success"]["n"]),
        "arms": arms,
        "oriented_deltas_positive_is_better": deltas,
        "decision": {
            "state_chart_predictor_independent_gain_supported": bool(predictor_positive),
            "adaptive_local_effect_gain_supported": bool(repair_positive),
            "claim_boundary": (
                "This is a privileged state-chart benchmark. It supports only a "
                "conditional local-effect repair result; it does not establish "
                "frozen RGB/LeWM predictor gain, causal latent physics, or universal transfer."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    summary = summarize(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
