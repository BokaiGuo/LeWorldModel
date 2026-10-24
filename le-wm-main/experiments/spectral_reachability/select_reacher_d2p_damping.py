"""Select a frozen Reacher D2P damping value from calibration-only receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Use only calibration-root D2P physical final distance; lower wins."""

    if len(reports) < 2:
        raise ValueError("need at least two calibration receipts")
    failures: list[str] = []
    common: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        protocol, inputs, rows = report.get("protocol", {}), report.get("inputs", {}), report.get("rows", [])
        if report.get("schema") != "d2p_reacher_p0_goal_progress.v1" or report.get("status") != "calibration_only" or not protocol.get("calibration_only"):
            failures.append(f"report {index} is not calibration-only")
            continue
        if not all(report.get("gates", {}).values()) or not rows or any(row.get("split") != "calibration" for row in rows):
            failures.append(f"report {index} has a failed gate or non-calibration row")
            continue
        fingerprint = {
            "checkpoint": inputs.get("weights_sha256"),
            "adapter": inputs.get("adapter_sha256"),
            "runner": inputs.get("runner_sha256"),
            "action_parameterization": protocol.get("action_parameterization"),
            "environment_action_box_raw": protocol.get("environment_action_box_raw"),
            "model_action_box_normalized": protocol.get("model_action_box_normalized"),
            "future_horizon": protocol.get("future_horizon"),
            "history_len": protocol.get("history_len"),
            "action_block": protocol.get("action_block"),
            "checkpoint_action_block": protocol.get("checkpoint_action_block"),
            "action_layout": protocol.get("action_layout"),
            "relinearization_steps": protocol.get("relinearization_steps"),
            "d2p_restarts": protocol.get("d2p_restarts"),
            "d2p_initialization": protocol.get("d2p_initialization"),
            "d2p_proposal_samples": protocol.get("d2p_proposal_samples"),
            "d2p_proposal_elites": protocol.get("d2p_proposal_elites"),
            "context_mode": protocol.get("context_mode"),
            "rollout_root_conditioning": protocol.get("rollout_root_conditioning"),
            "execution_root_reconstruction": protocol.get("execution_root_reconstruction"),
            "warmup_policy": protocol.get("warmup_policy"),
            "warmup_steps": protocol.get("warmup_steps"),
            "seeds": [row.get("seed") for row in rows],
        }
        if common is None:
            common = fingerprint
        elif common != fingerprint:
            failures.append(f"report {index} changes a calibration contract")
            continue
        values = np.asarray([row["methods"]["d2p_goal_progress"]["physical"]["final_qpos_distance"] for row in rows], dtype=np.float64)
        dampings = {float(row["methods"]["d2p_goal_progress"]["receipt"]["damping"]) for row in rows}
        if len(dampings) != 1 or not np.isfinite(values).all():
            failures.append(f"report {index} has invalid damping or physical values")
            continue
        candidates.append({"damping": dampings.pop(), "calibration_root_mean_final_qpos_distance": float(values.mean()), "calibration_root_values": values.tolist()})
    if len({item["damping"] for item in candidates}) != len(candidates):
        failures.append("duplicate damping receipts")
    ordered = sorted(candidates, key=lambda item: (item["calibration_root_mean_final_qpos_distance"], item["damping"]))
    return {
        "schema": "d2p_reacher_damping_selection.v1",
        "status": "pass" if not failures else "invalid",
        "selection_rule": "smallest mean D2P final qpos distance on calibration-only roots; ties choose smaller damping",
        "selected": ordered[0] if ordered and not failures else None,
        "candidates": ordered,
        "calibration_root_seeds": common["seeds"] if common is not None else [],
        "calibration_contract": {key: value for key, value in (common or {}).items() if key != "seeds"},
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    result = select([_load(path) for path in args.reports])
    result["calibration_reports"] = [str(path) for path in args.reports]
    result["calibration_report_sha256"] = [_sha(path) for path in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise RuntimeError("damping calibration selection failed")


if __name__ == "__main__":
    main()
