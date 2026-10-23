"""Audit why the sealed V22 action-supported C3 panel released no roots.

This is a post-outcome diagnostic only.  It validates that V22 truly reused
V20's declared candidate-action support, retained root-level isolation, and
sealed its gate before opening outcome labels.  It then contrasts the
calibrated simultaneous error envelope with model winner margins and true
within-bank oracle gaps.  It never changes a gate, model, or outcome label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V22 = ROOT / "results" / "pointmaze_c3_action_supported_v22_20261023_cuda"
DEFAULT_V20_CONTEXT = ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_20261023_cuda" / "temporal_ood_v6_context.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("audit values must be non-empty finite numbers")
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def audit(*, v22_dir: Path = DEFAULT_V22, v20_context: Path = DEFAULT_V20_CONTEXT) -> dict[str, Any]:
    context = _read(v22_dir / "temporal_ood_v6_context.json")
    report = _read(v22_dir / "v21_c3_report.json")
    gate = _read(v22_dir / "v21_frozen_c3_gate.json")
    labels_payload = _read(v22_dir / "v21_outcome_labels.json")
    prior = _read(v20_context)
    records, prior_records = context.get("records"), prior.get("records")
    labels = labels_payload.get("labels")
    if not isinstance(records, list) or not isinstance(prior_records, list) or not isinstance(labels, dict):
        raise ValueError("V22/V20 contexts or V22 outcome labels are malformed")

    v22_hashes = {str(row["candidate_action_hash"]) for row in records}
    v20_hashes = {str(row["candidate_action_hash"]) for row in prior_records}
    v22_roots = {int(row["start_episode_id"]) for row in records}
    v20_roots = {int(row["start_episode_id"]) for row in prior_records}
    splits: dict[str, set[int]] = defaultdict(set)
    for row in records:
        splits[str(row["split"])].add(int(row["start_episode_id"]))
    if set(splits) != {"train", "calibration", "outcome"}:
        raise ValueError("V22 context lacks a complete train/calibration/outcome split")
    if any(splits[left] & splits[right] for left in splits for right in splits if left != right):
        raise ValueError("V22 root splits overlap")

    outcome_rows = [row for row in records if str(row["split"]) == "outcome"]
    expected_label_keys = {f"{row['instance_id']}/{int(row['candidate_id'])}" for row in outcome_rows}
    if set(labels) != expected_label_keys:
        raise ValueError("V22 outcome labels do not exactly bind the outcome candidate rows")
    gate_path = v22_dir / "v21_frozen_c3_gate.json"
    if str(labels_payload.get("gate_sha256", "")) != _sha256(gate_path):
        raise ValueError("V22 outcome labels are not hash-bound to the sealed gate")
    if gate.get("test_outcomes_opened") is not False:
        raise ValueError("V22 gate was not sealed before outcome labels")

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in outcome_rows:
        key = f"{row['instance_id']}/{int(row['candidate_id'])}"
        value = labels[key]
        if not isinstance(value, dict) or not isinstance(value.get("real_task_cost"), (float, int)):
            raise ValueError(f"invalid V22 outcome cost for {key}")
        grouped[str(row["instance_id"])].append(float(value["real_task_cost"]))
    if len(grouped) != 256 or any(len(costs) != 16 for costs in grouped.values()):
        raise ValueError("V22 must contain 256 complete sixteen-candidate outcome roots")
    oracle_gaps, oracle_spreads = [], []
    for costs in grouped.values():
        ordered = np.sort(np.asarray(costs, dtype=np.float64))
        oracle_gaps.append(float(ordered[1] - ordered[0]))
        oracle_spreads.append(float(ordered[-1] - ordered[0]))

    test = report.get("test")
    calibration = report.get("calibration")
    if not isinstance(test, dict) or not isinstance(calibration, dict) or not isinstance(calibration.get("contrast"), dict):
        raise ValueError("V22 C3 report lacks calibration/test records")
    root_rows = test.get("roots")
    if not isinstance(root_rows, list) or len(root_rows) != len(grouped):
        raise ValueError("V22 C3 report lacks one record per outcome root")
    bound = float(calibration["contrast"]["bound"])
    margins = [float(row["minimum_predicted_margin"]) for row in root_rows]
    release_count = int(test["released_root_count"])
    all_margin_fail = int(test["gate_failure_counts"]["certificate_margin_failure"]) == len(root_rows)
    max_oracle_gap = float(max(oracle_gaps))
    if release_count != 0 or not all_margin_fail:
        diagnosis = "not_the_v22_all_margin_failure_pattern"
    elif max_oracle_gap < bound:
        # Even an otherwise correct ranking cannot receive this *unique
        # winner* certificate until its simultaneous contrast error is below
        # the tightest physical decision resolution in the declared bank.
        # This is distinct from mere score rescaling: the envelope itself,
        # and/or the choice of unique-winner certificate, must change.
        diagnosis = "action_supported_but_unique_winner_certificate_resolution_limited"
    elif max(margins) < bound:
        diagnosis = "action_supported_but_predicted_contrast_resolution_limited"
    else:
        diagnosis = "action_supported_with_mixed_certificate_failures"

    return {
        "schema": "pointmaze_c3_action_supported_v22_failure_audit.v1",
        "claim_boundary": (
            "Post-outcome V22 failure diagnosis only. It cannot alter the sealed V22 model, gate, "
            "or release policy, and it is not a conditional-safety or online-planning result."
        ),
        "integrity": {
            "v20_v22_action_hash_sets_equal": v20_hashes == v22_hashes,
            "v20_action_hash_count": len(v20_hashes),
            "v22_action_hash_count": len(v22_hashes),
            "v20_v22_root_overlap_count": len(v20_roots & v22_roots),
            "v22_split_root_counts": {name: len(values) for name, values in sorted(splits.items())},
            "v22_gate_sha256_matches_outcome_binding": True,
            "v22_gate_precedes_outcome_by_contract": True,
        },
        "certificate": {
            "contrast_bound": bound,
            "released_root_count": release_count,
            "outcome_root_count": len(root_rows),
            "all_roots_failed_margin_certificate": all_margin_fail,
            "predicted_winner_margin": _quantiles(margins),
            "predicted_winner_margin_above_bound_count": int(sum(value > bound for value in margins)),
        },
        "outcome_only_oracle_geometry": {
            "best_to_second_cost_gap": _quantiles(oracle_gaps),
            "full_bank_cost_spread": _quantiles(oracle_spreads),
            "best_to_second_gap_ge_contrast_bound_fraction": float(np.mean(np.asarray(oracle_gaps) >= bound)),
        },
        "diagnosis": diagnosis,
        "next_experiment_requirement": (
            "Do not relax the V22 calibration threshold. A subsequent repair must improve rootwise action-cost "
            "contrast resolution on new train roots and use new independent calibration/outcome roots. Because "
            "the V22 contrast envelope exceeds every outcome-only unique-winner gap, that protocol must also "
            "predeclare a regret-budget or set-valued certificate rather than relabeling an empty unique-winner "
            "gate as a control success."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22-dir", type=Path, default=DEFAULT_V22)
    parser.add_argument("--v20-context", type=Path, default=DEFAULT_V20_CONTEXT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(v22_dir=args.v22_dir, v20_context=args.v20_context)
    output = args.output or (args.v22_dir / "v22_action_supported_failure_audit.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "diagnosis": report["diagnosis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
