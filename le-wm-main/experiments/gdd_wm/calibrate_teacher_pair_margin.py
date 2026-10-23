"""Freeze a D2 teacher-pair margin from train roots only."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(dataset: Path, output: Path, *, quantile: float = 0.25) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset requires records")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("split")) == "train":
            groups[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(float(row["teacher_score"]))
    deltas = [abs(left - right) for scores in groups.values() for i, left in enumerate(scores) for right in scores[i + 1:] if abs(left - right) > 0.0]
    if not deltas:
        raise ValueError("train roots contain no non-zero teacher pair differences")
    margin = float(np.quantile(np.asarray(deltas), quantile))
    report = {
        "schema": "gdd_wm_train_only_pair_margin.v0.1",
        "status": "calibrated",
        "dataset_sha256": _sha(dataset),
        "split": "train",
        "pair_count": len(deltas),
        "quantile": float(quantile),
        "margin": margin,
        "selection_uses_calibration_or_outcome": False,
        "claim_boundary": "train-scale calibration only; not teacher, D2, or planning evidence",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantile", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.dataset, args.output, quantile=args.quantile), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
