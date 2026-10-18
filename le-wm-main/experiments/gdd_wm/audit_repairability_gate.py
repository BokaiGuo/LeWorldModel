"""Authorize or reject a train-only learned-dynamics repair.

This is a provenance gate, not a planning benchmark.  It consumes capacity
audit artifacts and separates three questions that were previously conflated:

1. did a fixed readout beat the action-only baseline on calibration data?
2. did predicted-dynamics terms add value under root-disjoint train OOF?
3. is the held-out consequence attributable to dynamics rather than capacity?

Only (1) and (2) can authorize a deployment readout.  Held-out outcome
numbers are reported for audit and never used to authorize the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _delta(report: dict[str, Any], family: str) -> dict[str, Any]:
    metrics = report.get("metrics", {}).get(family, {})
    value = metrics.get("vs_action_only", {}).get("regret_delta")
    if not isinstance(value, dict):
        return {"mean": None, "ci95": None}
    return {"mean": value.get("mean"), "ci95": value.get("ci95")}


def audit(path: Path) -> dict[str, Any]:
    report = _load(path)
    calibration = report.get("calibration_selection", {})
    safe_gate = calibration.get("safe_gate", {})
    crossfit = report.get("nonlinear_calibration_selection", {}).get(
        "bilinear_train_crossfit_gate", {}
    )
    full_dynamics = bool(crossfit.get("full_dynamics_gate_passed", False))
    calibration_passed = bool(safe_gate.get("passed", False))
    selected = str(crossfit.get("selected", "missing"))
    if selected == "bilinear_full_predicted_dynamics" and full_dynamics:
        authorization = "permitted_predicted_dynamics"
    else:
        authorization = "fallback_action_only"

    # These are descriptive only: they use the held-out outcome and therefore
    # cannot affect authorization.
    selected_family = "calibration_safe_gate"
    held_out = _delta(report, selected_family)
    shuffled = _delta(report, "listwise_shuffled_prediction")
    ambiguity = None
    if held_out["mean"] is not None and shuffled["mean"] is not None:
        ambiguity = bool(float(shuffled["mean"]) < float(held_out["mean"]))

    return {
        "schema": "gdd_wm_repairability_gate.v0.1",
        "source": str(path),
        "status": authorization,
        "authorization": {
            "calibration_non_degradation_gate": calibration_passed,
            "predicted_dynamics_train_crossfit_gate": full_dynamics,
            "selected_crossfit_family": selected,
            "rule": "authorize only calibration gate AND predicted-dynamics train-OOF gate",
        },
        "held_out_diagnostic_only": {
            "selected_family": selected_family,
            "selected_vs_action_only": held_out,
            "shuffled_prediction_vs_action_only": shuffled,
            "shuffled_control_better_than_selected": ambiguity,
            "interpretation": (
                "capacity_or_action-template ambiguity; do not claim learned-dynamics attribution"
                if ambiguity is True
                else "no held-out shuffled-control ambiguity detected"
                if ambiguity is False
                else "held-out attribution unavailable"
            ),
        },
        "claim_boundary": (
            "This gate authorizes a train-only diagnostic readout only; it does not "
            "establish D-JEPA correctness, RGB online planning, or universal transfer."
        ),
    }


def run(inputs: list[Path], output: Path) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one capacity audit is required")
    audits = [audit(path) for path in inputs]
    permitted = sum(item["status"] == "permitted_predicted_dynamics" for item in audits)
    result = {
        "schema": "gdd_wm_repairability_gate_panel.v0.1",
        "artifact_count": len(audits),
        "permitted_predicted_dynamics_count": int(permitted),
        "audits": audits,
        "cross_environment_rule": "report each environment separately; never pool authorization across environments",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-audit", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.capacity_audit, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
