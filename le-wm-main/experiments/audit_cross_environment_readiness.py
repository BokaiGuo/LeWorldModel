"""Preflight the non-negotiable assets for frozen cross-environment transfer.

This is deliberately a readiness audit, not a proxy transfer result.  It
prevents an unavailable trajectory corpus or an environment-specific checkpoint
from being silently replaced by an unrelated replay artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ENVIRONMENTS = {
    "PushT": {"dataset": "pusht_expert_train.h5", "checkpoint": "pusht/lewm"},
    "TwoRoom": {"dataset": "tworoom.h5", "checkpoint": "tworoom/lewm"},
    "PointMaze": {"dataset": "pointmaze.h5", "checkpoint": "pointmaze/lewm"},
}


def checkpoint_ready(root: Path, relative: str) -> dict[str, object]:
    directory = root / "checkpoints" / relative
    return {
        "directory": str(directory),
        "weights": str(directory / "weights.pt"),
        "config": str(directory / "config.json"),
        "ready": (directory / "weights.pt").is_file() and (directory / "config.json").is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stablewm-root", type=Path, default=Path("/home/gbk/.stable_worldmodel"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, spec in ENVIRONMENTS.items():
        dataset = args.stablewm_root / "datasets" / spec["dataset"]
        checkpoint = checkpoint_ready(args.stablewm_root, spec["checkpoint"])
        rows.append({
            "environment": name,
            "dataset": str(dataset),
            "dataset_ready": dataset.is_file(),
            "checkpoint": checkpoint,
            "frozen_zero_shot_runnable": bool(dataset.is_file() and checkpoint["ready"]),
        })
    runnable = [row["environment"] for row in rows if row["frozen_zero_shot_runnable"]]
    report = {
        "protocol": {
            "required_evidence": "same fixed checkpoint must run zero-shot under legal RGB/action interfaces on at least two distinct environments",
            "forbidden_substitutions": ["per-environment adapter retraining", "privileged state at runtime", "dataset-free replay proxy"],
        },
        "rows": rows,
        "runnable_environments": runnable,
        "status": "ready" if len(runnable) >= 2 else "blocked_missing_assets",
        "claim_boundary": "No cross-environment transfer result exists unless at least two rows are runnable with the same compatible checkpoint. Environment-specific checkpoints cannot establish a universal world model.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cross_environment_readiness.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "cross_environment_readiness.md").write_text(
        "# Frozen cross-environment transfer readiness\n\n"
        f"Status: **{report['status']}**.\n\n"
        "| Environment | Dataset | Checkpoint | Zero-shot runnable |\n| --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {row['environment']} | {row['dataset_ready']} | {row['checkpoint']['ready']} | {row['frozen_zero_shot_runnable']} |"
            for row in rows
        ) + "\n"
    )
    print(json.dumps({"status": report["status"], "runnable_environments": runnable}, indent=2))


if __name__ == "__main__":
    main()
