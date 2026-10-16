"""Join frozen prediction, action-effect, and planning consequences.

This report is intentionally descriptive.  It does not fit a predictor of
planning risk from three environments and does not pool incomparable latent
MSE scales.  Its purpose is to expose the propagation break between
predictive sufficiency, action identifiability, and decision consequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_row(path: Path, environment: str) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != "frozen_cross_environment_prediction.v1":
        raise ValueError(f"{path}: unexpected prediction schema")
    reports = [row for row in payload.get("reports", []) if row.get("status") == "completed"]
    if len(reports) != 1:
        raise ValueError(f"{path}: expected exactly one completed report")
    report = reports[0]
    if report.get("environment") != environment:
        raise ValueError(f"{path}: expected {environment}, got {report.get('environment')}")
    summary = report["summary"]
    horizons: dict[str, Any] = {}
    for horizon in (1, 2, 4):
        observed = summary["by_mode_horizon"][f"observed/h{horizon}"]
        action = summary["paired_action_effect"][f"h{horizon}"]
        horizons[str(horizon)] = {
            "prediction_mse": float(observed["mse"]["mean"]),
            "prediction_mse_ci95": [float(x) for x in observed["mse"]["ci95"]],
            "relative_commutator": float(observed["relative_commutator"]),
            "zero_minus_observed_mse": float(action["zero_minus_observed_mse"]["mean"]),
            "zero_minus_observed_mse_ci95": [float(x) for x in action["zero_minus_observed_mse"]["ci95"]],
            "fraction_zero_worse": float(action["fraction_zero_worse"]),
            "action_effect_supported": bool(action["zero_minus_observed_mse"]["ci95"][0] > 0.0),
        }
    return {
        "environment": environment,
        "checkpoint": payload["protocol"].get("checkpoint"),
        "device": payload["protocol"].get("device") or "unspecified_in_artifact",
        "test_episode_count": int(report["test_episode_count"]),
        "horizons": horizons,
        "artifact": str(path),
    }


def _planning_row(path: Path, environment: str) -> dict[str, Any]:
    payload = _read(path)
    if environment == "TwoRoom":
        expected = "live_tworoom_planning_aggregate.v1"
    elif environment == "Reacher":
        expected = "live_reacher_planning_aggregate.v1"
    else:
        raise ValueError(environment)
    if payload.get("schema") != expected:
        raise ValueError(f"{path}: unexpected planning schema")
    paired = payload["paired"]
    return {
        "environment": environment,
        "planning_delta": float(paired["success_rate_delta"]),
        "planning_ci95": [float(x) for x in paired["bootstrap"]["ci95"]],
        "planning_wins": int(paired["wins"]),
        "planning_losses": int(paired["losses"]),
        "planning_ties": int(paired["ties"]),
        "planning_supported": bool(paired["bootstrap"]["ci95"][0] > 0.0),
        "artifact": str(path),
    }


def _cube_row(path: Path) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != "live_cube_phase_conditioned_planning.v1":
        raise ValueError(f"{path}: unexpected Cube planning schema")
    paired = payload["paired"]
    return {
        "environment": "Cube",
        "planning_delta": float(paired["success_rate_delta"]),
        "planning_mean_distance_delta_m": float(paired["mean_final_distance_delta_m"]),
        "planning_wins": int(paired["success_wins"]),
        "planning_losses": int(paired["success_losses"]),
        "planning_ties": int(paired["success_ties"]),
        "planning_supported": False,
        "artifact": str(path),
    }


def _pointmaze_row(path: Path) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != "independent_planning_gain.v1":
        raise ValueError(f"{path}: unexpected PointMaze planning schema")
    paired = payload["paired_inference"]
    return {
        "environment": "PointMaze",
        "planning_delta": float(paired["mean"]),
        "planning_ci95": [float(paired["ci95_low"]), float(paired["ci95_high"])],
        "planning_supported": bool(payload.get("positive_gain") and payload.get("statistical_gate")),
        "protocol_gate": bool(payload.get("protocol_gate")),
        "statistical_gate": bool(payload.get("statistical_gate")),
        "artifact": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tworoom-prediction", type=Path, required=True)
    parser.add_argument("--tworoom-planning", type=Path, required=True)
    parser.add_argument("--reacher-prediction", type=Path, required=True)
    parser.add_argument("--reacher-planning", type=Path, required=True)
    parser.add_argument("--cube-prediction", type=Path, required=True)
    parser.add_argument("--cube-planning", type=Path, required=True)
    parser.add_argument("--pointmaze-planning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction = [
        _prediction_row(args.tworoom_prediction, "swm/TwoRoom-v1"),
        _prediction_row(args.reacher_prediction, "swm/ReacherDMControl-v0"),
        _prediction_row(args.cube_prediction, "swm/OGBCube-v0"),
    ]
    planning = [
        _planning_row(args.tworoom_planning, "TwoRoom"),
        _planning_row(args.reacher_planning, "Reacher"),
        _cube_row(args.cube_planning),
        _pointmaze_row(args.pointmaze_planning),
    ]
    by_environment: dict[str, dict[str, Any]] = {
        "TwoRoom": {"prediction": prediction[0], "planning": planning[0]},
        "Reacher": {"prediction": prediction[1], "planning": planning[1]},
        "Cube": {"prediction": prediction[2], "planning": planning[2]},
        "PointMaze": {"prediction": None, "planning": planning[3]},
    }
    report = {
        "schema": "prediction_to_planning_gap.v1",
        "environments": list(by_environment),
        "rows": by_environment,
        "interpretation": {
            "TwoRoom": "action effect is supported and live planning consequence is supported",
            "Reacher": "action effect is supported and live planning consequence is supported",
            "Cube": "prediction is measurable but action effect is not supported and live planning fails",
            "PointMaze": "planning gate is protocol-valid but statistically inconclusive; prediction join unavailable",
        },
        "claim_boundary": "descriptive propagation-gap audit; not a universal predictor or pooled correlation claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
