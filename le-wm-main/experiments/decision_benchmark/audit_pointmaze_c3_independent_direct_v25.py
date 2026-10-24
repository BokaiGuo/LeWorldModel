"""Audit V25 without retrospectively changing its frozen outcome report.

V25 fixed V24's duplicate zero-motion roots.  Its high-motion arm still
builds each observable context as an opposed-velocity pair and deliberately
keeps a pair in one split.  Those two roots are therefore a source-level
cluster, not two independent calibration draws.  This audit makes that
cluster contract explicit and fail-closes the high-motion *root-level*
conformal claim made by the legacy V24 evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.root_observation_identity import assert_unique_root_observations


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "results" / "pointmaze_c3_independent_direct_v25_20261023_cuda"
DEFAULT_PREFLIGHT = ROOT / "results" / "pointmaze_c3_independent_direct_v25_static_preflight_20261023.json"
DEFAULT_HIGH_MANIFEST = ROOT / "data" / "point_maze_v77_c3_independent_direct_high_20261023" / "velocity_matched_manifest.json"
_PREFIX = "pointmaze_v6_episode_"


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


def _root_number(root: str) -> int:
    if not str(root).startswith(_PREFIX):
        raise ValueError(f"unexpected root ID: {root!r}")
    return int(str(root)[len(_PREFIX) :])


def _root_id(number: int) -> str:
    return f"{_PREFIX}{int(number)}"


def _pairs(manifest: dict[str, Any]) -> dict[str, str]:
    bounds, pairs = manifest.get("new_root_id_range"), manifest.get("pairs")
    if not (isinstance(bounds, list) and len(bounds) == 2 and isinstance(pairs, list)):
        raise ValueError("high-motion manifest lacks paired-root ledger")
    start, stop_inclusive = (int(value) for value in bounds)
    if stop_inclusive - start + 1 != 2 * len(pairs):
        raise ValueError("high-motion root range does not match its pair ledger")
    result: dict[str, str] = {}
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or int(pair.get("pair_index", -1)) != index:
            raise ValueError("high-motion pair ledger is not ordered")
        left, right = _root_id(start + 2 * index), _root_id(start + 2 * index + 1)
        result[left], result[right] = _root_id(start + 2 * index), _root_id(start + 2 * index)
    return result


def _cluster_values(values: dict[str, float], root_to_cluster: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for root, value in values.items():
        if root not in root_to_cluster:
            raise ValueError(f"high-motion report contains a root outside the pair ledger: {root}")
        grouped[root_to_cluster[root]].append(float(value))
    incomplete = {cluster: len(entries) for cluster, entries in grouped.items() if len(entries) != 2}
    if incomplete:
        raise ValueError(f"paired high-motion calibration/outcome is incomplete: {incomplete}")
    return {cluster: float(max(entries)) for cluster, entries in grouped.items()}


def _conformal_upper_bound(values: dict[str, float], *, alpha: float) -> dict[str, Any]:
    if not values or not 0.0 < float(alpha) < 1.0:
        raise ValueError("cluster calibration requires non-empty values and alpha in (0, 1)")
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(alpha))))
    if rank > len(values):
        return {"bound": None, "finite_sample_rank": rank, "finite_sample_lower_coverage": None}
    ordered = np.sort(np.asarray(list(values.values()), dtype=np.float64))
    return {
        "bound": float(ordered[rank - 1]),
        "finite_sample_rank": rank,
        "finite_sample_lower_coverage": float(rank / (len(values) + 1)),
    }


def _cluster_bootstrap(values: dict[str, float], *, seed: int, draws: int = 10_000) -> dict[str, float | int]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    if array.size < 2:
        raise ValueError("cluster bootstrap requires at least two independent pair clusters")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(int(draws), len(array)))].mean(axis=1)
    return {
        "estimate": float(array.mean()), "lower": float(np.quantile(means, .025)),
        "upper": float(np.quantile(means, .975)), "clusters": int(len(array)),
    }


def audit(
    *,
    v25_dir: Path = DEFAULT_DIR,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    high_manifest_path: Path = DEFAULT_HIGH_MANIFEST,
) -> dict[str, Any]:
    context, report = _read(v25_dir / "temporal_ood_v6_context.json"), _read(v25_dir / "v24_c3_report.json")
    gate, labels = _read(v25_dir / "v24_frozen_stratified_gate.json"), _read(v25_dir / "v24_outcome_labels.json")
    frozen, preflight, high_manifest = (
        _read(v25_dir / "temporal_ood_v6_frozen_selection.json"), _read(preflight_path), _read(high_manifest_path)
    )
    rows = context.get("records")
    if not isinstance(rows, list):
        raise ValueError("V25 context records are malformed")
    if report.get("schema") != "pointmaze_c3_independent_direct_v25.v1":
        raise ValueError("audit requires the V25 successor report schema")
    if report.get("successor_protocol") != "fresh_roots_after_v24_duplicate_zero_motion_root_invalidation":
        raise ValueError("V25 successor provenance is missing")
    if report.get("recovered_after_post_label_report_assembly_bug") is not True:
        raise ValueError("this audit is only for the sealed post-label V25 recovery")
    assert_unique_root_observations(rows)

    gate_path = v25_dir / "v24_frozen_stratified_gate.json"
    outcome_rows = [row for row in rows if str(row["split"]) == "outcome"]
    expected_labels = {f"{row['instance_id']}/{int(row['candidate_id'])}" for row in outcome_rows}
    if not isinstance(labels.get("labels"), dict) or set(labels["labels"]) != expected_labels:
        raise ValueError("V25 outcome label panel is incomplete")
    if labels.get("gate_sha256") != _sha256(gate_path) or gate.get("test_outcomes_opened") is not False:
        raise ValueError("V25 labels are not bound to the sealed pre-outcome gate")
    if frozen.get("context_sha256") != _sha256(v25_dir / "temporal_ood_v6_context.json"):
        raise ValueError("V25 frozen predictor is not bound to its scored context")
    checkpoint = Path(str(frozen.get("model_checkpoint_path", "")))
    if not checkpoint.is_file() or frozen.get("model_checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("V25 frozen predictor checkpoint binding is invalid")

    split_by_root: dict[str, str] = {}
    for row in rows:
        root, split = str(row["instance_id"]), str(row["split"])
        if split_by_root.setdefault(root, split) != split:
            raise ValueError("V25 root crosses splits")
    preflight_splits = preflight.get("split_roots")
    if not isinstance(preflight_splits, dict):
        raise ValueError("V25 preflight split ledger is missing")
    for split in ("train", "calibration", "outcome"):
        expected = {_root_id(root) for root in preflight_splits.get(split, [])}
        actual = {root for root, value in split_by_root.items() if value == split}
        if actual != expected:
            raise ValueError(f"V25 {split} context differs from its frozen preflight ledger")
    checks = preflight.get("checks")
    if not isinstance(checks, dict) or not all(bool(value) for value in checks.values()):
        raise ValueError("V25 static preflight did not pass all checks")

    root_to_cluster = _pairs(high_manifest)
    high_roots = set(root_to_cluster)
    panel_high = high_roots & set(split_by_root)
    design_high = high_roots - panel_high
    if len(panel_high) != 384 or len(design_high) != 64:
        raise ValueError("V25 high-motion panel/design partition is malformed")
    split_cluster_counts: dict[str, int] = {}
    for split in ("train", "calibration", "outcome"):
        roots = {root for root, value in split_by_root.items() if value == split} & high_roots
        clusters = {root_to_cluster[root] for root in roots}
        if len(roots) != 2 * len(clusters) or any(
            {root for root, cluster in root_to_cluster.items() if cluster == pair} - roots
            for pair in clusters
        ):
            raise ValueError(f"V25 high-motion paired roots are split across {split}")
        split_cluster_counts[split] = len(clusters)
    if split_cluster_counts != {"train": 64, "calibration": 64, "outcome": 64}:
        raise ValueError("V25 high-motion cluster counts are not 64/64/64")

    high = report.get("strata", {}).get("high_motion")
    if not isinstance(high, dict):
        raise ValueError("V25 high-motion report is missing")
    contrast = high.get("calibration", {}).get("contrast")
    coverage = high.get("test", {}).get("conformal_outcome_coverage")
    decision_rows = high.get("test", {}).get("roots")
    if not isinstance(contrast, dict) or not isinstance(coverage, dict) or not isinstance(decision_rows, list):
        raise ValueError("V25 high-motion contrast report is malformed")
    calibration_errors = dict(zip(contrast.get("calibration_groups_used", []), contrast.get("root_max_contrast_errors", [])))
    outcome_errors = {str(row["group_id"]): float(row["max_contrast_error"]) for row in coverage.get("roots", [])}
    if set(calibration_errors) != {root for root, split in split_by_root.items() if split == "calibration" and root in high_roots}:
        raise ValueError("V25 high-motion calibration errors do not cover its declared roots")
    if set(outcome_errors) != {root for root, split in split_by_root.items() if split == "outcome" and root in high_roots}:
        raise ValueError("V25 high-motion outcome errors do not cover its declared roots")
    calibration_clusters, outcome_clusters = _cluster_values(calibration_errors, root_to_cluster), _cluster_values(outcome_errors, root_to_cluster)
    pair_calibration = _conformal_upper_bound(calibration_clusters, alpha=float(contrast["alpha"]))
    sealed_bound = float(contrast["bound"])
    cluster_coverage_under_sealed = float(np.mean(np.asarray(list(outcome_clusters.values())) <= sealed_bound))
    counterfactual_bound = pair_calibration["bound"]
    counterfactual_coverage = None if counterfactual_bound is None else float(
        np.mean(np.asarray(list(outcome_clusters.values())) <= float(counterfactual_bound))
    )
    gains = {str(row["root_id"]): float(row["paired_regret_gain"]) for row in decision_rows}
    if set(gains) != set(outcome_errors):
        raise ValueError("V25 high-motion decision rows do not align with contrast outcomes")
    paired_gains = _cluster_values(gains, root_to_cluster)

    return {
        "schema": "pointmaze_c3_independent_direct_v25_audit.v1",
        "claim_boundary": "Post-outcome integrity audit only. It neither refits V25 nor retunes its sealed gate and does not retrofit a conformal certificate.",
        "integrity": {
            "fresh_direct_observations_unique": True,
            "root_observations_unique_within_split": True,
            "preflight_checks_all_passed": True,
            "frozen_checkpoint_bound": True,
            "sealed_gate_bound_to_outcome_labels": True,
            "split_roots_match_preflight": True,
        },
        "high_motion_dependency": {
            "construction": "opposed_velocity source pairs share a target source position and future candidate controls",
            "pair_preserved_within_split": True,
            "panel_root_count_by_split": {split: count * 2 for split, count in split_cluster_counts.items()},
            "independent_source_pair_count_by_split": split_cluster_counts,
            "rootwise_finite_sample_claim_valid": False,
            "reason": "the sealed evaluator calibrated 128 roots although its declared split preserves 64 coupled source pairs",
        },
        "high_motion_cluster_diagnostic": {
            "sealed_rootwise_contrast_bound": sealed_bound,
            "sealed_rootwise_rank": contrast.get("finite_sample_rank"),
            "sealed_rootwise_outcome_coverage": coverage.get("empirical_root_coverage"),
            "cluster_outcome_coverage_under_sealed_rootwise_bound": cluster_coverage_under_sealed,
            "cluster_calibration_count": len(calibration_clusters),
            "cluster_conformal_rank": pair_calibration["finite_sample_rank"],
            "cluster_conformal_lower_coverage": pair_calibration["finite_sample_lower_coverage"],
            "counterfactual_cluster_bound_for_successor_design_only": counterfactual_bound,
            "counterfactual_cluster_outcome_coverage_for_diagnosis_only": counterfactual_coverage,
            "cluster_bootstrap_paired_regret_gain": _cluster_bootstrap(paired_gains, seed=202611811),
        },
        "diagnosis": "high_motion_calibration_shortfall_is_protocol_contaminated_by_paired_cluster_pseudoreplication",
        "next_experiment_requirement": "Do not alter V25's gate or reopen its labels. A fresh successor must either generate independent high-motion roots or calibrate and bootstrap one maximum-error/mean-utility statistic per opposed-velocity source pair.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v25-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--high-manifest", type=Path, default=DEFAULT_HIGH_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(v25_dir=args.v25_dir, preflight_path=args.preflight, high_manifest_path=args.high_manifest)
    output = args.output or (args.v25_dir / "v25_independent_direct_audit.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "diagnosis": result["diagnosis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
