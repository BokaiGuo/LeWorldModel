"""Post-outcome integrity and limitation audit for V23's epsilon-regret rule.

The audit never changes V23's frozen model, calibration gate, or release
decisions.  It validates the root/action isolation and gate-to-outcome binding,
then separates a positive matched-fallback effect from empirical coverage on
the one independent outcome panel.
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
DEFAULT_DIR = ROOT / "results" / "pointmaze_c3_regret_budget_v23_20261023_cuda"
DEFAULT_PREFLIGHT = ROOT / "results" / "pointmaze_c3_regret_budget_v23_static_preflight_20261023.json"


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
    return {name: float(value) for name, value in {
        "min": np.min(array), "p10": np.quantile(array, .10), "p50": np.median(array),
        "p90": np.quantile(array, .90), "max": np.max(array),
    }.items()}


def _root_number(root: str) -> int:
    prefix = "pointmaze_v6_episode_"
    if not str(root).startswith(prefix):
        raise ValueError(f"unexpected PointMaze root id: {root!r}")
    return int(str(root)[len(prefix):])


def audit(*, v23_dir: Path = DEFAULT_DIR, preflight_path: Path = DEFAULT_PREFLIGHT) -> dict[str, Any]:
    context = _read(v23_dir / "temporal_ood_v6_context.json")
    source = _read(v23_dir / "temporal_ood_v6_source_dataset.json")
    report = _read(v23_dir / "v23_c3_report.json")
    gate_path = v23_dir / "v23_frozen_c3_gate.json"
    gate = _read(gate_path)
    labels_payload = _read(v23_dir / "v23_outcome_labels.json")
    preflight = _read(preflight_path)
    records, source_rows, labels = context.get("records"), source.get("records"), labels_payload.get("labels")
    if not isinstance(records, list) or not isinstance(source_rows, list) or not isinstance(labels, dict):
        raise ValueError("V23 context, source, or outcome labels are malformed")
    splits: dict[str, set[int]] = defaultdict(set)
    for row in records:
        splits[str(row["split"])].add(int(row["start_episode_id"]))
    if set(splits) != {"train", "calibration", "outcome"} or any(
        splits[left] & splits[right] for left in splits for right in splits if left != right
    ):
        raise ValueError("V23 does not have disjoint train/calibration/outcome roots")
    outcome_rows = [row for row in records if str(row["split"]) == "outcome"]
    expected_label_keys = {f"{row['instance_id']}/{int(row['candidate_id'])}" for row in outcome_rows}
    if set(labels) != expected_label_keys:
        raise ValueError("V23 outcome labels do not exactly bind outcome candidate rows")
    if labels_payload.get("gate_sha256") != _sha256(gate_path) or gate.get("test_outcomes_opened") is not False:
        raise ValueError("V23 outcome labels are not bound to a gate sealed before opening outcomes")
    if gate.get("certificate_kind") != "regret_budget" or float(gate.get("regret_budget", -1.0)) <= 0.0:
        raise ValueError("V23 gate lacks its fixed epsilon-regret declaration")
    candidate = preflight.get("candidate_bank")
    prior_contexts = preflight.get("prior_context_sha256")
    if not isinstance(candidate, dict) or candidate.get("c3_certificate") != "fixed_bank_epsilon_regret_v1":
        raise ValueError("V23 preflight lacks its declared certificate protocol")
    if not isinstance(prior_contexts, dict) or not prior_contexts:
        raise ValueError("V23 preflight lacks bound predecessor contexts")
    prior_roots, prior_hashes = set(), set()
    for raw_path, expected_hash in prior_contexts.items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"V23 prior context changed or disappeared: {path}")
        prior = _read(path)
        prior_records = prior.get("records")
        if not isinstance(prior_records, list):
            raise ValueError(f"V23 prior context is malformed: {path}")
        prior_roots.update(int(row["start_episode_id"]) for row in prior_records)
        prior_hashes.update(str(row["candidate_action_hash"]) for row in prior_records)
    held_hashes = {str(row["candidate_action_hash"]) for row in records}
    train_hashes = {str(row["candidate_action_hash"]) for row in source_rows}
    v23_roots = set().union(*splits.values())
    if not held_hashes <= train_hashes:
        raise ValueError("V23 held actions are not represented in its train-source bank")
    if v23_roots & prior_roots or held_hashes & prior_hashes:
        raise ValueError("V23 violates frozen predecessor root/action isolation")

    test = report.get("test")
    calibration = report.get("calibration")
    if not isinstance(test, dict) or not isinstance(calibration, dict) or not isinstance(calibration.get("contrast"), dict):
        raise ValueError("V23 report lacks calibrated outcome evaluation")
    root_rows = test.get("roots")
    if not isinstance(root_rows, list) or len(root_rows) != len(splits["outcome"]):
        raise ValueError("V23 report lacks one decision record per outcome root")
    if calibration["contrast"].get("bound") != gate["contrast"].get("bound"):
        raise ValueError("V23 report contrast envelope differs from the sealed gate")
    if calibration.get("risk_threshold") != gate["risk"].get("bound"):
        raise ValueError("V23 report risk threshold differs from the sealed gate")
    if report.get("protocol", {}).get("test_outcome_used_for_selection") is not False:
        raise ValueError("V23 protocol reports outcome-dependent action selection")

    actual_release = [row for row in root_rows if row["status"] == "permit"]
    unique_releases = [row for row in root_rows if row["margin_certificate_status"] == "permit"]
    gains = [float(row["paired_regret_gain"]) for row in root_rows]
    realized = [float(row["htsc_regret"]) for row in root_rows]
    coverage = test.get("conformal_outcome_coverage", {})
    empirical_coverage = float(coverage["empirical_root_coverage"])
    target_coverage = float(calibration["contrast"]["target_marginal_coverage"])
    pool_path = Path(str(preflight["dataset"])) / "mixed_velocity_repairability_manifest.json"
    pool = _read(pool_path)
    high_range, direct_range = pool.get("high_root_range"), pool.get("direct_root_range")
    if not (
        isinstance(high_range, list) and isinstance(direct_range, list)
        and len(high_range) == len(direct_range) == 2
    ):
        raise ValueError("V23 pool lacks high/direct root ranges")

    def regime_for(root: str) -> str:
        number = _root_number(root)
        if int(high_range[0]) <= number <= int(high_range[1]):
            return "high_motion"
        if int(direct_range[0]) <= number <= int(direct_range[1]):
            return "direct_zero_motion"
        raise ValueError("V23 root lies outside its mixed-pool ranges")

    calibration_groups = calibration["contrast"].get("calibration_groups_used")
    calibration_errors = calibration["contrast"].get("root_max_contrast_errors")
    if not (
        isinstance(calibration_groups, list) and isinstance(calibration_errors, list)
        and len(calibration_groups) == len(calibration_errors) == 256
    ):
        raise ValueError("V23 calibration rootwise contrast table is malformed")
    calibration_by_regime: dict[str, list[float]] = {"high_motion": [], "direct_zero_motion": []}
    for root, error in zip(calibration_groups, calibration_errors):
        calibration_by_regime[regime_for(str(root))].append(float(error))
    calibration_regime_summary = {
        regime: {
            "root_count": len(errors), "max_contrast_error": _quantiles(errors),
            "global_bound_exceedance_rate": float(np.mean(np.asarray(errors) > float(gate["contrast"]["bound"]))),
        }
        for regime, errors in calibration_by_regime.items()
    }
    coverage_by_root = {
        str(row["group_id"]): row for row in coverage.get("roots", [])
    }
    by_regime: dict[str, list[dict[str, Any]]] = {"high_motion": [], "direct_zero_motion": []}
    for row in root_rows:
        root = str(row["root_id"])
        if root not in coverage_by_root:
            raise ValueError("V23 coverage table does not align with decisions")
        regime = regime_for(root)
        by_regime[regime].append({**row, "coverage": coverage_by_root[root]})
    regime_summary: dict[str, dict[str, Any]] = {}
    for regime, regime_rows in by_regime.items():
        if not regime_rows:
            raise ValueError(f"V23 has no outcome roots in {regime}")
        regime_summary[regime] = {
            "root_count": len(regime_rows),
            "empirical_coverage": float(np.mean([bool(row["coverage"]["covered"]) for row in regime_rows])),
            "epsilon_release_rate": float(np.mean([row["status"] == "permit" for row in regime_rows])),
            "unique_winner_release_rate": float(np.mean([row["margin_certificate_status"] == "permit" for row in regime_rows])),
            "mean_paired_regret_gain": float(np.mean([float(row["paired_regret_gain"]) for row in regime_rows])),
            "mean_selected_regret": float(np.mean([float(row["htsc_regret"]) for row in regime_rows])),
            "posthoc_ranking_failure_count": int(sum(bool(row["posthoc_utility_ranking_failure"]) for row in regime_rows)),
        }
    if empirical_coverage < target_coverage:
        diagnosis = "positive_epsilon_regret_transfer_with_empirical_coverage_shortfall"
    elif float(test["paired_regret_gain_bootstrap"]["lower"]) > 0.0 and actual_release:
        diagnosis = "positive_epsilon_regret_transfer_with_coverage_consistent_outcome"
    else:
        diagnosis = "epsilon_regret_protocol_completed_without_positive_matched_transfer"
    return {
        "schema": "pointmaze_c3_regret_budget_v23_audit.v1",
        "claim_boundary": (
            "Post-outcome V23 integrity and descriptive audit only. The finite-sample certificate remains "
            "marginal over fixed-bank exchangeable roots; this audit does not create permit-conditional safety "
            "or an online-planning claim."
        ),
        "integrity": {
            "split_root_counts": {name: len(values) for name, values in sorted(splits.items())},
            "split_root_disjoint": True,
            "predecessor_context_count": len(prior_contexts),
            "v23_predecessor_root_overlap_count": len(v23_roots & prior_roots),
            "v23_predecessor_action_hash_overlap_count": len(held_hashes & prior_hashes),
            "held_action_hash_count": len(held_hashes), "train_action_hash_count": len(train_hashes),
            "held_actions_supported_by_train": True,
            "gate_sha256_matches_outcome_binding": True,
            "gate_precedes_outcome_by_contract": True,
        },
        "certificate": {
            "kind": "epsilon_regret", "regret_budget": float(gate["regret_budget"]),
            "contrast_bound": float(gate["contrast"]["bound"]),
            "risk_threshold": float(gate["risk"]["bound"]),
            "outcome_root_count": len(root_rows), "epsilon_regret_released_root_count": len(actual_release),
            "unique_winner_released_root_count": len(unique_releases),
            "minimum_predicted_margin": _quantiles([float(row["minimum_predicted_margin"]) for row in root_rows]),
            "joint_bound_holds_on_covered_roots": coverage.get("covered_root_regret_bound_all_hold"),
        },
        "outcome": {
            "epsilon_regret_release_rate": float(len(actual_release) / len(root_rows)),
            "paired_regret_gain": _quantiles(gains),
            "paired_regret_gain_bootstrap": test["paired_regret_gain_bootstrap"],
            "selected_action_regret": _quantiles(realized),
            "empirical_simultaneous_coverage": empirical_coverage,
            "target_marginal_coverage": target_coverage,
            "empirical_coverage_wilson_95_interval": coverage.get("empirical_root_coverage_wilson_95_interval"),
            "posthoc_utility_ranking_failure_count": int(test["posthoc_utility_ranking_failure_count"]),
            "calibration_by_regime": calibration_regime_summary,
            "by_regime": regime_summary,
        },
        "diagnosis": diagnosis,
        "next_experiment_requirement": (
            "Treat the paired regret gain as an independent positive direction only after reporting the observed "
            "coverage shortfall. Do not lower alpha or alter epsilon on this opened panel. A replication must use "
            "fresh roots and predeclare whether it targets improved contrast coverage, a different action bank, or "
            "a set-valued probe policy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v23-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(v23_dir=args.v23_dir, preflight_path=args.preflight)
    output = args.output or (args.v23_dir / "v23_epsilon_regret_audit.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "diagnosis": report["diagnosis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
