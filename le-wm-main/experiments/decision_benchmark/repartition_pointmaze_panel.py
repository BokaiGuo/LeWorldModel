"""Create a frozen, root-disjoint split from a generated PointMaze panel.

This utility is used only before RGB verification and adapter fitting.  The
root lists are explicit command-line inputs so an outcome root cannot be
silently selected from task labels or post-hoc success.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _read_roots(value: str) -> set[str]:
    roots = {item.strip() for item in value.split(",") if item.strip()}
    if not roots:
        raise ValueError("each root list must contain at least one root")
    return roots


def repartition(
    source: Path,
    output: Path,
    train: set[str],
    calibration: set[str],
    outcome: set[str],
    exclude: set[str] | None = None,
) -> None:
    exclude = set() if exclude is None else set(exclude)
    overlap = (train & calibration) | (train & outcome) | (calibration & outcome)
    if overlap & exclude:
        raise ValueError(f"a root cannot be both assigned and excluded: {sorted(overlap & exclude)}")
    if overlap:
        raise ValueError(f"root split overlap: {sorted(overlap)}")
    payload = json.loads((source / "candidate_dataset.json").read_text(encoding="utf-8"))
    records = payload["records"]
    available = {str(row["instance_id"]) for row in records}
    requested = train | calibration | outcome
    missing = requested - available
    if missing:
        raise ValueError(f"requested roots are not in source panel: {sorted(missing)}")
    unexpected = available - requested - exclude
    if unexpected:
        raise ValueError(
            "every generated root must be assigned explicitly; unassigned roots: "
            + repr(sorted(unexpected))
        )
    split_by_root = {
        **{root: "train" for root in train},
        **{root: "calibration" for root in calibration},
        **{root: "outcome" for root in outcome},
    }
    records = [row for row in records if str(row["instance_id"]) not in exclude]
    for row in records:
        row["split"] = split_by_root[str(row["instance_id"])]
    payload["records"] = records
    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "root_split_source": "explicit_root_lists_frozen_before_outcome_evaluation",
        "root_split_train_count": len(train),
        "root_split_calibration_count": len(calibration),
        "root_split_outcome_count": len(outcome),
        "outcome_root_selection_from_task_labels": False,
        "outcome_root_selection_from_success": False,
        "excluded_root_count": len(exclude),
        "source_panel": str(source.resolve()),
    })
    payload["metadata"] = metadata
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate_dataset.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    shutil.copy2(source / "arrays.npz", output / "arrays.npz")
    manifest = {
        "schema": "pointmaze_goal_conditioned_candidate_bank.explicit_split.v1",
        "source_panel": str(source.resolve()),
        "candidate_dataset": str((output / "candidate_dataset.json").resolve()),
        "arrays": str((output / "arrays.npz").resolve()),
        "split_root_counts": {
            "train": len(train), "calibration": len(calibration), "outcome": len(outcome),
        },
        "root_disjoint": True,
        "outcome_root_selection_from_task_labels": False,
        "claim_boundary": "fresh root-disjoint outcome split for offline action-transfer intervention; not online planning evidence",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-roots", required=True)
    parser.add_argument("--calibration-roots", required=True)
    parser.add_argument("--outcome-roots", required=True)
    parser.add_argument("--exclude-roots", default="")
    args = parser.parse_args()
    repartition(
        args.source, args.output,
        _read_roots(args.train_roots),
        _read_roots(args.calibration_roots),
        _read_roots(args.outcome_roots),
        _read_roots(args.exclude_roots) if args.exclude_roots.strip() else set(),
    )
    print(json.dumps({"output": str(args.output.resolve()), "root_disjoint": True}, indent=2))


if __name__ == "__main__":
    main()
