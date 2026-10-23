"""Freeze a calibration-only execution-alignment gate for a GDD teacher.

Teacher stability and non-leaking provenance are necessary but not sufficient:
a score can be stable while ordering realized candidate outcomes at chance. This
audit reads *only* calibration-root realized costs, estimates root-level pair
agreement with the fixed teacher, and uses a deterministic root bootstrap to
decide whether the lower confidence bound exceeds chance.  Outcome rows are
not read and cannot influence the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_teacher_execution_alignment.v0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sign(value: float, margin: float) -> int:
    return 0 if abs(float(value)) <= margin else (1 if value > 0.0 else -1)


def _root_agreement(rows: list[dict[str, Any]], margin: float) -> tuple[int, int]:
    """Return correct/comparable pair counts for one root across its contexts."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["teacher_set_id"])].append(row)
    correct = comparable = 0
    for group in groups.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                teacher = _sign(float(left["teacher_score"]) - float(right["teacher_score"]), margin)
                # Lower realized task cost is preferred.
                execution = _sign(float(right["real_task_cost"]) - float(left["real_task_cost"]), margin)
                if teacher == 0 or execution == 0:
                    continue
                comparable += 1
                correct += int(teacher == execution)
    return correct, comparable


def audit(
    dataset_path: Path,
    *,
    margin: float = 1.0e-3,
    alpha: float = 0.05,
    draws: int = 10_000,
    seed: int = 202610381,
    minimum_roots: int = 20,
) -> dict[str, Any]:
    """Create a calibration-only, root-resampled teacher-alignment decision."""

    if not 0.0 < alpha < 1.0 or draws < 100 or minimum_roots < 1 or margin < 0.0:
        raise ValueError("alpha, draws, minimum_roots, and margin violate the gate contract")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("candidate dataset must be an object containing records")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("candidate dataset must contain metadata")
    calibration = [dict(row) for row in payload["records"] if str(row.get("split")) == "calibration"]
    if not calibration:
        raise ValueError("candidate dataset has no calibration rows")
    required = {"instance_id", "teacher_set_id", "teacher_score", "real_task_cost", "teacher_model_id"}
    for index, row in enumerate(calibration):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"calibration row {index} lacks {missing}")
    expected_teacher = metadata.get("teacher_model_id")
    teacher_ids = {str(row["teacher_model_id"]) for row in calibration}
    if len(teacher_ids) != 1 or expected_teacher not in teacher_ids:
        raise ValueError("calibration teacher identity must be homogeneous and match dataset metadata")
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration:
        by_root[str(row["instance_id"])].append(row)
    root_rows = []
    for root, rows in sorted(by_root.items()):
        correct, comparable = _root_agreement(rows, margin)
        if comparable:
            root_rows.append({"root": root, "correct": correct, "comparable": comparable, "agreement": correct / comparable})
    if len(root_rows) < minimum_roots:
        status = "rejected_insufficient_calibration_roots"
        lower = None
    else:
        agreement = np.asarray([row["agreement"] for row in root_rows], dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        sample_indices = rng.integers(0, len(agreement), size=(int(draws), len(agreement)))
        lower = float(np.quantile(agreement[sample_indices].mean(axis=1), alpha))
        status = "passed_calibration_alignment" if lower > 0.5 else "rejected_calibration_alignment"
    total_correct = int(sum(row["correct"] for row in root_rows))
    total_comparable = int(sum(row["comparable"] for row in root_rows))
    return {
        "schema": SCHEMA,
        "status": status,
        "passed": status == "passed_calibration_alignment",
        "claim_boundary": "Calibration-only teacher execution-alignment gate; it neither reads outcome roots nor establishes held-out planning benefit.",
        "candidate_dataset_sha256": _sha256(dataset_path),
        "teacher_model_id": str(expected_teacher),
        "calibration_only": True,
        "outcome_rows_read": False,
        "protocol": {
            "pair_margin": float(margin), "chance_agreement": 0.5,
            "one_sided_root_bootstrap_alpha": float(alpha), "bootstrap_draws": int(draws),
            "bootstrap_seed": int(seed), "minimum_comparable_root_count": int(minimum_roots),
        },
        "calibration": {
            "root_count": len(by_root), "comparable_root_count": len(root_rows),
            "comparable_pair_count": total_comparable, "correct_pair_count": total_correct,
            "pair_weighted_agreement": None if not total_comparable else float(total_correct / total_comparable),
            "root_mean_agreement": None if not root_rows else float(np.mean([row["agreement"] for row in root_rows])),
            "root_bootstrap_lower_bound": lower,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=202610381)
    parser.add_argument("--minimum-roots", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite teacher alignment audit: {args.output}")
    result = audit(args.dataset, margin=args.margin, alpha=args.alpha, draws=args.draws, seed=args.seed, minimum_roots=args.minimum_roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
