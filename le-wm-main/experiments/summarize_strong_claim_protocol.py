"""Fail-closed aggregation for the six strong world-model evidence gates.

The script never pools missing rows, substitutes a checkpoint, or promotes a
proxy to an end-to-end result.  It emits both a machine-readable audit and a
reader-facing status report; raw per-episode/per-action artifacts remain the
authoritative evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.real_checkpoint_selection_protocol import evaluate_records


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95": None}
    if array.size == 1:
        return {"n": 1, "mean": float(array[0]), "ci95": None}
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, array.size, size=(4000, array.size))].mean(axis=1)
    return {"n": int(array.size), "mean": float(array.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def expected_paths(manifest: dict[str, Any], protocol: Path, stage: str) -> list[Path]:
    commands = manifest["commands"][stage]
    paths: list[Path] = []
    for command in commands:
        tokens = command.split()
        output = Path(tokens[tokens.index("--output") + 1])
        if stage == "training":
            paths.append(output)
        elif stage == "selection":
            paths.append(output / "checkpoint_record.json")
        else:
            paths.append(output / "causal_intervention_audit.json")
    return paths


def summarize_selection(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    invalid: list[str] = []
    for path in paths:
        payload = load_json(path)
        if payload is None:
            invalid.append(str(path))
            continue
        if payload.get("record_schema") != "real_checkpoint_selection.v1":
            invalid.append(str(path))
            continue
        protocol = payload.get("protocol", {})
        interface = protocol.get("interface", {})
        quality = protocol.get("prediction_quality", {})
        if interface.get("direct_utility") is None or quality.get("metric") != "planner_matched_open_loop_rollout_mse":
            invalid.append(str(path))
            continue
        records.append(payload["record"])
    report: dict[str, Any] = {
        "expected": len(paths), "valid": len(records), "invalid_or_missing": invalid,
        "status": "complete" if len(records) == len(paths) else "partial_or_invalid",
    }
    if records:
        report["benchmark"] = evaluate_records(records, mse_relative_tolerance=0.05)
    return report


def summarize_causal(paths: list[Path]) -> dict[str, Any]:
    rows: dict[str, list[float]] = defaultdict(list)
    coverage: dict[str, list[float]] = defaultdict(list)
    rank: dict[str, list[float]] = defaultdict(list)
    valid = 0
    invalid: list[str] = []
    for path in paths:
        payload = load_json(path)
        if payload is None or not finite(payload) or "summary" not in payload:
            invalid.append(str(path)); continue
        valid += 1
        for key, item in payload["summary"].items():
            if not isinstance(item, dict):
                continue
            if item.get("heldout_effect_rmse") is not None:
                rows[key].append(float(item["heldout_effect_rmse"]))
            if item.get("counterfactual_goal_rank_top1") is not None:
                rank[key].append(float(item["counterfactual_goal_rank_top1"]))
            if item.get("conformal_effect_coverage_90") is not None:
                coverage[key].append(float(item["conformal_effect_coverage_90"]))
    return {
        "expected": len(paths), "valid": valid, "invalid_or_missing": invalid,
        "status": "complete" if valid == len(paths) else "partial_or_invalid",
        "effect_rmse": {key: bootstrap(values, 61000 + index) for index, (key, values) in enumerate(sorted(rows.items()))},
        "rank_top1": {key: bootstrap(values, 62000 + index) for index, (key, values) in enumerate(sorted(rank.items()))},
        "coverage_90": {key: bootstrap(values, 63000 + index) for index, (key, values) in enumerate(sorted(coverage.items()))},
    }


def assess_alignment(selection: dict[str, Any]) -> tuple[bool, str]:
    """Assess evidence, rather than merely checking record completeness."""
    if selection.get("status") != "complete":
        return False, "selection_records_incomplete_or_invalid"
    summary = selection.get("benchmark", {}).get("summary", {})
    regret = summary.get("selection_regret", {})
    alignment = regret.get("alignment_only", {}).get("mean")
    spectrum = regret.get("spectrum_only", {}).get("mean")
    if alignment is None or spectrum is None:
        return False, "selection_regret_missing"
    if not float(alignment) < float(spectrum):
        return False, "standalone_alignment_does_not_beat_spectrum_only"
    # The current benchmark reports per-group bootstrap intervals, but not a
    # paired selector-difference interval.  Do not promote a mean comparison
    # to a significance claim until that paired estimand is added.
    return False, "paired_selector_difference_CI_not_registered"


def assess_causal(causal: dict[str, Any]) -> tuple[bool, str]:
    if causal.get("status") != "complete":
        return False, "causal_artifacts_incomplete_or_invalid"
    report = causal.get("rank_top1", {})
    expected_modes = {"contact", "sliding", "separation", "no_contact"}
    observed_modes = {key.split("/", 1)[1] for key in report if "/" in key}
    if not expected_modes.issubset(observed_modes):
        return False, "heldout_contact_mode_missing"
    h8 = [item.get("mean") for key, item in report.items() if key.startswith("h8/")]
    if not h8 or any(value is None or float(value) <= 0.5 for value in h8):
        return False, "long_horizon_action_ranking_not_stable"
    return True, "mode_stratified_long_horizon_ranking_passed"


def assess_external_prediction(payload: dict[str, Any] | None) -> tuple[bool, str]:
    if not payload or not payload.get("reports"):
        return False, "cross_environment_prediction_missing"
    for report in payload["reports"]:
        if report.get("status") != "completed":
            return False, f"environment_not_completed:{report.get('environment')}"
        for item in report.get("summary", {}).get("paired_action_effect", {}).values():
            interval = item.get("zero_minus_observed_mse", {}).get("ci95")
            if not interval or float(interval[0]) <= 0.0:
                return False, f"observed_action_not_better_than_zero:{report.get('environment')}"
    return True, "all_environments_show_action_sensitive_prediction"


def assess_visual_online_planning(payload: dict[str, Any] | None) -> tuple[bool, str]:
    if not payload:
        return False, "frozen_visual_online_planning_missing"
    aggregate = payload.get("aggregate", {})
    if int(aggregate.get("n_seed_clusters", 0)) < 2:
        return False, "frozen_visual_online_planning_needs_two_seed_clusters"
    success = aggregate.get("vanilla_success_total")
    zero_success = aggregate.get("zero_success_total")
    delta = aggregate.get("vanilla_minus_zero_block_xy_distance", {})
    if success is None or zero_success is None or not delta:
        return False, "frozen_visual_online_planning_summary_incomplete"
    if int(success) <= int(zero_success):
        return False, "vanilla_strict_success_not_above_zero"
    if float(delta.get("ci_high", 0.0)) >= 0.0:
        return False, "vanilla_block_xy_not_better_than_zero"
    return True, "frozen_visual_online_planning_passed"


def build(
    protocol: Path,
    external: Path | None,
    visual_summary: Path | None = None,
    cross_environment_prediction: Path | None = None,
    trajectory_audit: Path | None = None,
    local_operator: Path | None = None,
    frozen_visual_audit: Path | None = None,
) -> dict[str, Any]:
    manifest = load_json(protocol / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(protocol / "manifest.json")
    training_paths = expected_paths(manifest, protocol, "training")
    selection_paths = expected_paths(manifest, protocol, "selection")
    causal_paths = expected_paths(manifest, protocol, "causal_ood")
    training_complete = sum((path / "causal_visual_world_model.pt").is_file() and (path / "split_manifest.json").is_file() for path in training_paths)
    selection = summarize_selection(selection_paths)
    causal = summarize_causal(causal_paths)
    external_report = None
    if cross_environment_prediction is not None:
        external_report = load_json(cross_environment_prediction)
    elif external is not None:
        external_report = load_json(external / "frozen_pusht_checkpoint_prediction" / "cross_environment_prediction.json")
    visual_report = load_json(visual_summary) if visual_summary is not None else None
    trajectory_report = load_json(trajectory_audit) if trajectory_audit is not None else None
    local_operator_report = load_json(local_operator) if local_operator is not None else None
    frozen_visual_report = load_json(frozen_visual_audit) if frozen_visual_audit is not None else None
    planning_readiness = load_json((external or protocol) / "frozen_planning_readiness" / "cross_environment_readiness.json")
    completion = {
        "training": training_complete == len(training_paths),
        "alignment_selection_records": selection["status"] == "complete",
        "causal_ood_artifacts": causal["status"] == "complete",
        "cross_environment_prediction_artifacts": bool(
            external_report and all(row.get("status") == "completed" for row in external_report.get("reports", []))
        ),
        "cross_environment_planning_protocol": bool(
            planning_readiness and planning_readiness.get("status") == "ready"
        ),
        "visual_multiseed_planning_artifact": bool(
            visual_report and visual_report.get("protocol", {}).get("n_seed_clusters", 0) >= 5
        ),
        "trajectory_local_operator_artifact": bool(
            trajectory_report and trajectory_report.get("artifact") == "external_frozen_lewm_trajectory_operator_audit"
        ),
        "action_conditioned_local_operator_artifact": bool(
            local_operator_report and local_operator_report.get("protocol", {}).get("source_fields") == ["latents", "predicted_dynamics", "actions"]
        ),
        "frozen_visual_online_planning_artifact": bool(
            frozen_visual_report and frozen_visual_report.get("aggregate", {}).get("n_seed_clusters", 0) >= 2
        ),
    }
    alignment_pass, alignment_reason = assess_alignment(selection)
    causal_pass, causal_reason = assess_causal(causal)
    external_pass, external_reason = assess_external_prediction(external_report)
    visual_online_pass, visual_online_reason = assess_visual_online_planning(frozen_visual_report)
    planning_pass = bool(planning_readiness and planning_readiness.get("status") == "ready")
    gates = {
        "alignment_real_checkpoint_selection": alignment_pass,
        "causal_ood_intervention": causal_pass,
        "cross_environment_prediction": external_pass,
        "cross_environment_planning": planning_pass,
        "frozen_visual_online_planning": visual_online_pass,
    }
    return {
        "schema": "strong_world_model_evidence_bundle.v1",
        "protocol": str(protocol),
        "manifest_sha256": manifest.get("sha256_before_execution"),
        "training": {"expected": len(training_paths), "complete": int(training_complete)},
        "completion": completion,
        "selection": selection,
        "causal_ood": causal,
        "external_prediction": external_report,
        "visual_multiseed_planning": visual_report,
        "trajectory_operator_audit": trajectory_report,
        "action_conditioned_local_operator": local_operator_report,
        "frozen_visual_online_planning": frozen_visual_report,
        "cross_environment_planning_readiness": planning_readiness,
        "gates": gates,
        "gate_reasons": {
            "alignment_real_checkpoint_selection": alignment_reason,
            "causal_ood_intervention": causal_reason,
            "cross_environment_prediction": external_reason,
            "cross_environment_planning": "planning_readiness_not_ready" if not planning_pass else "ready",
            "frozen_visual_online_planning": visual_online_reason,
        },
        "claim_status": {
            "alignment": "validated_protocol_only" if not gates["alignment_real_checkpoint_selection"] else "empirical_selection_benchmark_complete_requires_interpretation",
            "physical_causal_dynamics": "not_established_until_causal_gate_and_mode_stability_pass",
            "frozen_visual_planning": "not_established_by_prediction_audit",
            "universal_world_model": "not_established" if not all(gates.values()) else "scoped_external_validity_only",
            "commutator_certificate": "not_universal; composite_diagnostic_only",
        },
        "claim_boundary": "This bundle is an evidence gate. Missing, invalid, or negative results remain visible and cannot be promoted by pooled averages.",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Strong world-model evidence bundle", "", "## Evidence gate status", "", "| Gate | Status | Reason |", "| --- | --- | --- |"]
    for key, value in report["gates"].items():
        lines.append(f"| {key} | **{'PASS' if value else 'NOT PASS'}** | {report['gate_reasons'][key]} |")
    lines += ["", "## Artifact completion", "", f"- Training: {report['training']['complete']}/{report['training']['expected']}.", f"- Selection: {report['selection']['valid']}/{report['selection']['expected']} valid.", f"- Causal OOD: {report['causal_ood']['valid']}/{report['causal_ood']['expected']} valid.", f"- Multi-seed visual planning artifact: `{report['completion']['visual_multiseed_planning_artifact']}`.", f"- Frozen visual online-planning artifact: `{report['completion']['frozen_visual_online_planning_artifact']}`.", f"- Trajectory local-operator artifact: `{report['completion']['trajectory_local_operator_artifact']}`.", f"- Action-conditioned local-operator artifact: `{report['completion']['action_conditioned_local_operator_artifact']}`.", "- External prediction: completed artifacts are checked separately from evidence quality.", "", "## Claim boundary", "", report["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--visual-summary", type=Path)
    parser.add_argument("--cross-environment-prediction", type=Path)
    parser.add_argument("--trajectory-audit", type=Path)
    parser.add_argument("--local-operator", type=Path)
    parser.add_argument("--frozen-visual-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(
        args.protocol,
        args.external,
        visual_summary=args.visual_summary,
        cross_environment_prediction=args.cross_environment_prediction,
        trajectory_audit=args.trajectory_audit,
        local_operator=args.local_operator,
        frozen_visual_audit=args.frozen_visual_audit,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "strong_world_model_evidence_bundle.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "strong_world_model_evidence_bundle.md").write_text(markdown(report))
    print(json.dumps({"gates": report["gates"], "training": report["training"], "selection": {k: report["selection"][k] for k in ("expected", "valid", "status")}, "causal_ood": {k: report["causal_ood"][k] for k in ("expected", "valid", "status")}}, indent=2))


if __name__ == "__main__":
    main()
