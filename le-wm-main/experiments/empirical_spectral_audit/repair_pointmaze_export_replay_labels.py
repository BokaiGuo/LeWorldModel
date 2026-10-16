"""Repair per-window PointMaze state labels in an existing enriched export.

The historical multi-window export wrote ``target_state_raw`` using a stale
batch-local source index.  This utility does not touch predictions, split
membership, or action blocks.  It only reconstructs the declared raw state
labels by replaying each row's observed four-block action sequence from its
recorded source state, and writes a new artifact for auditability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.empirical_spectral_audit.evaluate_pointmaze_support_matched_external_evidence import (
    _legacy_pointmaze_xml,
)

HORIZONS = (1, 2, 4)


def _replay_states(row: dict[str, Any]) -> np.ndarray:
    import mujoco

    source = np.asarray(row["source_state_raw"], dtype=np.float64)
    actions = np.asarray(row["action_blocks_raw"], dtype=np.float64)
    if source.shape != (4,):
        raise ValueError(f"source_state_raw must have shape (4,), got {source.shape}")
    if actions.shape != (4, 10):
        raise ValueError(f"action_blocks_raw must have shape (4, 10), got {actions.shape}")
    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    data.qpos[:] = source[:2]
    data.qvel[:] = source[2:]
    mujoco.mj_forward(model, data)
    states: list[np.ndarray] = []
    for block in actions.reshape(4, 5, 2):
        for action in block:
            data.ctrl[:] = action
            for _ in range(5):
                mujoco.mj_step(model, data)
        states.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
    return np.asarray(states, dtype=np.float64)


def repair(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text())
    rows = payload.get("enriched_records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("input export has no enriched_records")
    repaired = 0
    continuity_errors: list[float] = []
    for row in rows:
        replay = _replay_states(row)
        original = np.asarray([
            row["latent_by_horizon"][str(h)]["target_state_raw"]
            for h in HORIZONS
        ], dtype=np.float64)
        for index, horizon in enumerate(HORIZONS):
            row["latent_by_horizon"][str(horizon)]["target_state_raw"] = replay[
                horizon - 1
            ].tolist()
        repaired += 1
        continuity_errors.extend(np.linalg.norm(original - replay[[0, 1, 3]], axis=1).tolist())

    protocol = dict(payload.get("protocol", {}))
    protocol["replay_label_repair"] = {
        "source": str(input_path.resolve()),
        "method": "modern MuJoCo replay from each row source_state_raw and action_blocks_raw",
        "steps_per_action": 5,
        "horizons": list(HORIZONS),
        "predictions_unchanged": True,
        "split_membership_unchanged": True,
    }
    payload["protocol"] = protocol
    payload["schema"] = "enriched_learned_model.v1.repaired_replay_labels"
    report = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "records_repaired": repaired,
        "original_vs_replayed_target_l2": {
            "mean": float(np.mean(continuity_errors)),
            "median": float(np.median(continuity_errors)),
            "max": float(np.max(continuity_errors)),
        },
        "claim_boundary": "Label-repair audit only; no model or planning result is changed.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    (output_path.parent / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
