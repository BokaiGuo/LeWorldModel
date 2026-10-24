"""Fail-closed integrity audit for a newly generated PointMaze SRA archive.

This validates only the archive transition contract.  It is deliberately
separate from model scoring and is not P4 efficacy evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability.pointmaze_physical_evaluator import (
    FRAME_SKIP,
    VELOCITY_LIMIT,
    build_model,
    replay_recorded_actions,
)


SCHEMA = "pointmaze_sra_p4_high_level_archive_transition_audit.v1"
TOLERANCE = 1.0e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_checks(manifest: dict[str, object]) -> dict[str, bool]:
    contract = manifest.get("rollout_transition_contract")
    if not isinstance(contract, dict):
        return {"declared_high_level_contract": False, "evaluator_hash_matches": False}
    evaluator = Path(__file__).with_name("pointmaze_physical_evaluator.py")
    return {
        "declared_high_level_contract": (
            contract.get("name") == "mazeenv_high_level_action_v1"
            and contract.get("state_clock") == "states[0] is reset; states[t+1] follows actions[t]"
            and contract.get("velocity_clip_before_action") == float(VELOCITY_LIMIT)
            and contract.get("raw_action_clip") == [-1.0, 1.0]
            and contract.get("mujoco_steps_per_action") == int(FRAME_SKIP)
            and manifest.get("state_action_dtype") == "float64"
        ),
        "evaluator_hash_matches": contract.get("physical_evaluator_sha256") == _sha256(evaluator),
    }


def audit(archive: Path) -> dict[str, object]:
    archive = archive.resolve()
    manifest_path = archive / "extension_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    states = torch.load(archive / "states.pth", map_location="cpu", weights_only=True).detach().cpu().numpy()
    actions = torch.load(archive / "actions.pth", map_location="cpu", weights_only=True).detach().cpu().numpy()
    start, stop = int(manifest["base_root_count"]), int(manifest["extended_root_count"])
    if states.shape != actions.shape[:2] + (4,) or actions.shape[2:] != (2,):
        raise ValueError("archive must contain states [roots,time,4] and actions [roots,time,2]")
    if not (0 <= start < stop <= len(states)):
        raise ValueError("manifest fresh root interval is invalid")
    errors: list[float] = []
    for root in range(start, stop):
        replayed = replay_recorded_actions(build_model(), states[root, 0], actions[root, : states.shape[1] - 1])
        errors.append(float(np.max(np.abs(replayed - states[root, states.shape[1] - 1]))))
    checks = {
        **_contract_checks(manifest),
        "state_dtype_float64": states.dtype == np.float64,
        "action_dtype_float64": actions.dtype == np.float64,
        "finite_tensors": bool(np.isfinite(states[start:stop]).all() and np.isfinite(actions[start:stop]).all()),
        "replay_matches_stored_terminal_state": bool(max(errors, default=float("inf")) < TOLERANCE),
    }
    return {
        "schema": SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "claim_boundary": "High-level archive transition integrity only; not model scoring or P4 efficacy evidence.",
        "archive": str(archive),
        "archive_manifest_sha256": _sha256(manifest_path),
        "fresh_root_range": [start, stop - 1],
        "fresh_root_count": stop - start,
        "max_abs_terminal_state_error": max(errors, default=float("inf")),
        "median_abs_terminal_state_error": float(np.median(errors)) if errors else float("inf"),
        "tolerance": TOLERANCE,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit receipt: {args.output}")
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
