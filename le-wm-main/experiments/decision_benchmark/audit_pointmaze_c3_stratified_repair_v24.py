"""Audit V24's frozen stratified C3 outcome without retuning it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "results" / "pointmaze_c3_stratified_repair_v24_20261023_cuda"
DEFAULT_PREFLIGHT = ROOT / "results" / "pointmaze_c3_stratified_repair_v24_static_preflight_20261023.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _root_number(root: str) -> int:
    prefix = "pointmaze_v6_episode_"
    if not str(root).startswith(prefix): raise ValueError(f"unexpected root: {root!r}")
    return int(str(root)[len(prefix):])


def audit(*, v24_dir: Path = DEFAULT_DIR, preflight_path: Path = DEFAULT_PREFLIGHT) -> dict[str, Any]:
    context, source = _read(v24_dir / "temporal_ood_v6_context.json"), _read(v24_dir / "temporal_ood_v6_source_dataset.json")
    report, gate = _read(v24_dir / "v24_c3_report.json"), _read(v24_dir / "v24_frozen_stratified_gate.json")
    labels_payload, preflight = _read(v24_dir / "v24_outcome_labels.json"), _read(preflight_path)
    records, source_rows, labels = context.get("records"), source.get("records"), labels_payload.get("labels")
    if not isinstance(records, list) or not isinstance(source_rows, list) or not isinstance(labels, dict):
        raise ValueError("V24 context/source/outcome labels malformed")
    splits: dict[str, set[int]] = defaultdict(set)
    for row in records: splits[str(row["split"])].add(int(row["start_episode_id"]))
    if set(splits) != {"train", "calibration", "outcome"} or any(splits[a] & splits[b] for a in splits for b in splits if a != b):
        raise ValueError("V24 root splits overlap or are incomplete")
    outcome_rows = [row for row in records if str(row["split"]) == "outcome"]
    expected_labels = {f"{row['instance_id']}/{int(row['candidate_id'])}" for row in outcome_rows}
    gate_path = v24_dir / "v24_frozen_stratified_gate.json"
    if set(labels) != expected_labels or labels_payload.get("gate_sha256") != _sha256(gate_path) or gate.get("test_outcomes_opened") is not False:
        raise ValueError("V24 outcome labels are not bound to a sealed gate")
    candidate, prior_contexts = preflight.get("candidate_bank"), preflight.get("prior_context_sha256")
    if not isinstance(candidate, dict) or candidate.get("c3_certificate") != "stratified_fixed_bank_epsilon_regret_v1":
        raise ValueError("V24 lacks a stratified certificate declaration")
    if not isinstance(prior_contexts, dict) or not prior_contexts: raise ValueError("V24 prior contexts missing")
    prior_roots, prior_hashes = set(), set()
    for raw, expected in prior_contexts.items():
        path = Path(raw)
        if not path.is_file() or _sha256(path) != expected: raise ValueError(f"bound context changed: {path}")
        prior_records = _read(path).get("records")
        if not isinstance(prior_records, list): raise ValueError(f"malformed prior context: {path}")
        prior_roots.update(int(row["start_episode_id"]) for row in prior_records)
        prior_hashes.update(str(row["candidate_action_hash"]) for row in prior_records)
    held_hashes = {str(row["candidate_action_hash"]) for row in records}
    train_hashes = {str(row["candidate_action_hash"]) for row in source_rows}
    v24_roots = set().union(*splits.values())
    if not held_hashes <= train_hashes or v24_roots & prior_roots or held_hashes & prior_hashes:
        raise ValueError("V24 violates shared action support or predecessor isolation")
    pool = _read(Path(str(preflight["dataset"])) / "mixed_velocity_repairability_manifest.json")
    high, direct = pool.get("high_root_range"), pool.get("direct_root_range")
    if not (isinstance(high, list) and isinstance(direct, list) and len(high) == len(direct) == 2): raise ValueError("V24 pool ranges missing")

    def stratum(root: str) -> str:
        number = _root_number(root)
        if int(high[0]) <= number <= int(high[1]): return "high_motion"
        if int(direct[0]) <= number <= int(direct[1]): return "direct_zero_motion"
        raise ValueError("V24 root outside pool stratum")

    top_test, reports = report.get("test"), report.get("strata")
    if not isinstance(top_test, dict) or not isinstance(reports, dict) or set(reports) != {"high_motion", "direct_zero_motion"}:
        raise ValueError("V24 report lacks two stratum reports")
    summaries: dict[str, Any] = {}
    for name, arm in reports.items():
        calibration, test = arm.get("calibration"), arm.get("test")
        if not isinstance(calibration, dict) or not isinstance(test, dict): raise ValueError("V24 stratum report malformed")
        sealed = gate.get("strata", {}).get(name)
        if not isinstance(sealed, dict) or calibration["contrast"]["bound"] != sealed["contrast"]["bound"] or calibration["risk_threshold"] != sealed["risk"]["bound"]:
            raise ValueError("V24 stratum report differs from sealed calibration")
        rows = test.get("roots")
        if not isinstance(rows, list) or not rows or any(stratum(str(row["root_id"])) != name for row in rows):
            raise ValueError("V24 decision rows have incorrect stratum assignment")
        coverage = test["conformal_outcome_coverage"]
        summaries[name] = {"root_count": len(rows), "release_count": test["released_root_count"], "release_rate": test["release_rate"],
                           "coverage": coverage["empirical_root_coverage"], "coverage_wilson_95": coverage["empirical_root_coverage_wilson_95_interval"],
                           "covered_regret_bound_all_hold": coverage["covered_root_regret_bound_all_hold"],
                           "paired_regret_gain": test["paired_regret_gain_mean"], "paired_regret_gain_bootstrap": test["paired_regret_gain_bootstrap"],
                           "paired_success_gain": test["paired_success_gain_mean"], "ranking_failure_count": test["posthoc_utility_ranking_failure_count"],
                           "contrast_bound": sealed["contrast"]["bound"], "risk_threshold": sealed["risk"]["bound"]}
    high_gain = float(summaries["high_motion"]["paired_regret_gain_bootstrap"]["lower"])
    high_coverage = float(summaries["high_motion"]["coverage"])
    direct_gain = float(summaries["direct_zero_motion"]["paired_regret_gain"])
    direct_coverage = float(summaries["direct_zero_motion"]["coverage"])
    if high_gain > 0.0 and high_coverage >= .9 and direct_gain == 0.0 and direct_coverage < .9:
        diagnosis = "high_motion_repair_transfers_with_direct_stratum_residual_coverage_shortfall"
    elif high_gain > 0.0 and high_coverage >= .9:
        diagnosis = "stratified_repair_transfers_with_coverage_consistent_high_motion_stratum"
    else:
        diagnosis = "stratified_repair_completed_without_confirmed_high_motion_transfer"
    return {"schema": "pointmaze_c3_stratified_repair_v24_audit.v1",
            "claim_boundary": "Post-outcome audit only; marginal fixed-bank stratum coverage, not permit-conditional safety or online MPC.",
            "integrity": {"split_root_counts": {name: len(values) for name, values in sorted(splits.items())}, "split_root_disjoint": True,
                          "predecessor_context_count": len(prior_contexts), "predecessor_root_overlap_count": len(v24_roots & prior_roots),
                          "predecessor_action_hash_overlap_count": len(held_hashes & prior_hashes), "held_action_hash_count": len(held_hashes),
                          "held_actions_supported_by_train": True, "gate_sha256_matches_outcome_binding": True, "gate_precedes_outcome_by_contract": True},
            "certificate": {"kind": "stratified_epsilon_regret", "regret_budget": gate["regret_budget"], "route": gate["policy"], "strata": summaries},
            "combined_outcome": {"root_count": top_test["root_count"], "release_rate": top_test["release_rate"], "paired_regret_gain": top_test["paired_regret_gain_mean"],
                                 "paired_regret_gain_bootstrap": top_test["paired_regret_gain_bootstrap"], "paired_success_gain": top_test["paired_success_gain_mean"],
                                 "coverage": top_test["conformal_outcome_coverage"]["empirical_root_coverage"], "covered_regret_bound_all_hold": top_test["conformal_outcome_coverage"]["covered_root_regret_bound_all_hold"]},
            "diagnosis": diagnosis,
            "next_experiment_requirement": "Do not tune V24's strata, thresholds, or epsilon on this opened panel. Fresh data must test whether direct-zero-motion needs a wider calibration, stricter abstention, or a different observable support gate."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(v24_dir=args.v24_dir, preflight_path=args.preflight)
    output = args.output or (args.v24_dir / "v24_stratified_repair_audit.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "diagnosis": result["diagnosis"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
