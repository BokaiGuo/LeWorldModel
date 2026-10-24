"""Integrity audit for the fresh, cluster-calibrated V26 successor."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.root_observation_identity import assert_unique_root_observations


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "results" / "pointmaze_c3_clustered_high_v26_20261023_cuda"
DEFAULT_PREFLIGHT = ROOT / "results" / "pointmaze_c3_clustered_high_v26_static_preflight_20261023.json"
DEFAULT_HIGH_MANIFEST = ROOT / "data" / "point_maze_v79_c3_clustered_high_20261023" / "velocity_matched_manifest.json"
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


def _root_id(number: int) -> str:
    return f"{_PREFIX}{int(number)}"


def _paired_clusters(manifest: dict[str, Any]) -> dict[str, str]:
    bounds, pairs = manifest.get("new_root_id_range"), manifest.get("pairs")
    if not (isinstance(bounds, list) and len(bounds) == 2 and isinstance(pairs, list)):
        raise ValueError("V26 high manifest lacks its pair ledger")
    first, last = (int(value) for value in bounds)
    if last - first + 1 != 2 * len(pairs):
        raise ValueError("V26 high pair ledger has inconsistent range")
    result: dict[str, str] = {}
    for index, row in enumerate(pairs):
        if not isinstance(row, dict) or int(row.get("pair_index", -1)) != index:
            raise ValueError("V26 high pair ledger order is invalid")
        cluster, other = _root_id(first + 2 * index), _root_id(first + 2 * index + 1)
        result[cluster], result[other] = cluster, cluster
    return result


def audit(*, v26_dir: Path = DEFAULT_DIR, preflight_path: Path = DEFAULT_PREFLIGHT,
          high_manifest_path: Path = DEFAULT_HIGH_MANIFEST) -> dict[str, Any]:
    context, report = _read(v26_dir / "temporal_ood_v6_context.json"), _read(v26_dir / "v24_c3_report.json")
    gate, labels, frozen, preflight, manifest = (
        _read(v26_dir / "v24_frozen_stratified_gate.json"), _read(v26_dir / "v24_outcome_labels.json"),
        _read(v26_dir / "temporal_ood_v6_frozen_selection.json"), _read(preflight_path), _read(high_manifest_path),
    )
    rows = context.get("records")
    if not isinstance(rows, list) or report.get("schema") != "pointmaze_c3_clustered_high_v26.v1":
        raise ValueError("V26 context or successor report schema is invalid")
    assert_unique_root_observations(rows)
    if not all(bool(value) for value in preflight.get("checks", {}).values()):
        raise ValueError("V26 static preflight did not pass")
    if frozen.get("context_sha256") != _sha256(v26_dir / "temporal_ood_v6_context.json"):
        raise ValueError("V26 frozen model is not bound to the scored context")
    checkpoint = Path(str(frozen.get("model_checkpoint_path", "")))
    if not checkpoint.is_file() or frozen.get("model_checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("V26 frozen model checkpoint binding is invalid")
    gate_path = v26_dir / "v24_frozen_stratified_gate.json"
    outcome = [row for row in rows if str(row["split"]) == "outcome"]
    expected_labels = {f"{row['instance_id']}/{int(row['candidate_id'])}" for row in outcome}
    if set(labels.get("labels", {})) != expected_labels or labels.get("gate_sha256") != _sha256(gate_path):
        raise ValueError("V26 outcome labels are not bound to the sealed gate")
    if gate.get("test_outcomes_opened") is not False:
        raise ValueError("V26 gate does not predate opened outcomes")

    paired = _paired_clusters(manifest)
    split_by_root = {str(row["instance_id"]): str(row["split"]) for row in rows}
    panel_pairs: dict[str, set[str]] = defaultdict(set)
    for root, cluster in paired.items():
        if root in split_by_root:
            panel_pairs[cluster].add(split_by_root[root])
    if len(panel_pairs) != 192 or any(len(splits) != 1 for splits in panel_pairs.values()):
        raise ValueError("V26 high-motion source pairs cross a split or are incomplete")
    paired_counts = {split: sum(splits == {split} for splits in panel_pairs.values()) for split in ("train", "calibration", "outcome")}
    if paired_counts != {"train": 64, "calibration": 64, "outcome": 64}:
        raise ValueError("V26 high-motion cluster split counts are invalid")

    high = report["strata"]["high_motion"]
    contrast = high["calibration"]["contrast"]
    coverage = high["test"]["cluster_conformal_outcome_coverage"]
    if contrast.get("calibration_cluster_count") != 64 or contrast.get("calibration_root_count") != 128 or contrast.get("finite_sample_rank") != 59:
        raise ValueError("V26 high-motion calibration did not use paired clusters")
    if coverage.get("outcome_cluster_count") != 64:
        raise ValueError("V26 high-motion outcome did not report paired clusters")
    root_gain = {str(row["root_id"]): float(row["paired_regret_gain"]) for row in high["test"]["roots"]}
    clustered_gain: dict[str, list[float]] = defaultdict(list)
    for root, gain in root_gain.items(): clustered_gain[paired[root]].append(gain)
    if any(len(values) != 2 for values in clustered_gain.values()):
        raise ValueError("V26 high-motion utility rows do not retain complete source pairs")
    gain = float(np.mean([np.mean(values) for values in clustered_gain.values()]))
    if not np.isclose(gain, float(high["test"]["paired_regret_gain_mean"])):
        raise ValueError("V26 high-motion cluster utility does not reproduce the report")

    return {
        "schema": "pointmaze_c3_clustered_high_v26_audit.v1",
        "claim_boundary": "Post-outcome integrity audit only; no V26 threshold, model, candidate bank, or label was changed.",
        "integrity": {"fresh_roots_after_v25": min(int(root[len(_PREFIX):]) for root in split_by_root) > 35079,
                      "root_observations_unique_within_split": True, "preflight_checks_all_passed": True,
                      "frozen_checkpoint_bound": True, "sealed_gate_bound_to_outcome_labels": True},
        "high_motion_cluster_protocol": {"pair_preserved_within_split": True, "independent_source_pairs_by_split": paired_counts,
                                           "calibration_cluster_count": 64, "calibration_rank": 59,
                                           "finite_sample_marginal_coverage_lower_bound": contrast["finite_sample_marginal_coverage_lower_bound"],
                                           "outcome_cluster_count": coverage["outcome_cluster_count"],
                                           "outcome_cluster_coverage": coverage["empirical_cluster_coverage"]},
        "outcome": {"high_motion_release_rate": high["test"]["release_rate"], "high_motion_paired_regret_gain": gain,
                    "high_motion_cluster_bootstrap": high["test"]["paired_regret_gain_bootstrap"],
                    "direct_zero_motion_release_rate": report["strata"]["direct_zero_motion"]["test"]["release_rate"],
                    "direct_zero_motion_cluster_coverage": report["strata"]["direct_zero_motion"]["test"]["cluster_conformal_outcome_coverage"]["empirical_cluster_coverage"]},
        "diagnosis": "v25_high_motion_negative_coverage_was_a_paired_cluster_protocol_defect_not_supported_theory_evidence",
        "reporting_note": "The initial V26 core report retained a legacy rootwise wording; its numeric high-motion calibration and this audit are clusterwise. The successor source now writes the precise wording for any recovery assembly.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v26-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--high-manifest", type=Path, default=DEFAULT_HIGH_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(v26_dir=args.v26_dir, preflight_path=args.preflight, high_manifest_path=args.high_manifest)
    output = args.output or args.v26_dir / "v26_clustered_high_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "diagnosis": result["diagnosis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
