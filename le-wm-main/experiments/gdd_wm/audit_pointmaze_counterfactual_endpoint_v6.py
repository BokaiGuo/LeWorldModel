"""Fail-closed contract audit for a completed PointMaze endpoint V6 report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "pointmaze_counterfactual_endpoint_v6_contract_audit.v1"
REPORT_SCHEMA = "pointmaze_counterfactual_endpoint_control.v1"
REQUIRED_STRESS_SPLITS = ("train", "calibration", "outcome")
REQUIRED_ALIAS_SPLITS = ("calibration", "outcome")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_preflight_binding_is_intact(report: dict[str, Any], split: Any) -> bool:
    """Follow the report -> panel-context -> preflight-ledger hash chain.

    A serialized SHA string is only an assertion.  This independent audit
    reopens the emitted panel context and its referenced static-preflight file,
    so a later reader can distinguish a bound V6 panel from a report that only
    claims to have used one.
    """

    if not isinstance(split, dict):
        return False
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        return False
    context_name, context_hash = provenance.get("context"), provenance.get("context_sha256")
    if not isinstance(context_name, str) or not isinstance(context_hash, str):
        return False
    context_path = Path(context_name)
    if not context_path.is_file() or _sha256(context_path) != context_hash:
        return False
    try:
        context_payload = _load(context_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    metadata = context_payload.get("metadata")
    ledger = metadata.get("frozen_preflight_ledger") if isinstance(metadata, dict) else None
    if not isinstance(ledger, dict):
        return False
    ledger_name, ledger_hash = ledger.get("path"), ledger.get("sha256")
    if not isinstance(ledger_name, str) or not isinstance(ledger_hash, str) or len(ledger_hash) != 64:
        return False
    ledger_path = Path(ledger_name)
    if not ledger_path.is_file() or _sha256(ledger_path) != ledger_hash:
        return False
    try:
        ledger_payload = _load(ledger_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    checks = ledger_payload.get("checks")
    return bool(
        ledger_payload.get("schema") == "pointmaze_counterfactual_endpoint_v6_static_preflight.v1"
        and ledger_payload.get("status") == "passed_static_preflight"
        and isinstance(checks, dict) and checks and all(value is True for value in checks.values())
        and split.get("frozen_preflight_ledger_sha256") == ledger_hash
        and split.get("frozen_preflight_ledger_context_count") == ledger.get("context_count")
    )


def _calibration_role_binding_is_intact(report: dict[str, Any], split: Any) -> bool:
    """Verify that the frozen role partition uses the runner's root identity."""

    if not isinstance(split, dict):
        return False
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        return False
    context_name, context_hash = provenance.get("context"), provenance.get("context_sha256")
    if not isinstance(context_name, str) or not isinstance(context_hash, str):
        return False
    context_path = Path(context_name)
    if not context_path.is_file() or _sha256(context_path) != context_hash:
        return False
    try:
        context_payload = _load(context_path)
        metadata = context_payload.get("metadata")
        ledger = metadata.get("frozen_preflight_ledger") if isinstance(metadata, dict) else None
        ledger_path = Path(str(ledger.get("path", ""))) if isinstance(ledger, dict) else None
        ledger_payload = _load(ledger_path) if ledger_path is not None and ledger_path.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    names = {"selector", "arm_certificate", "risk_calibration"}
    expected = ledger_payload.get("calibration_role_counts") if isinstance(ledger_payload, dict) else None
    actual = split.get("calibration_roles")
    return (
        isinstance(expected, dict)
        and isinstance(actual, dict)
        and set(expected) == names
        and set(actual) == names
        and all(isinstance(expected[name], int) and isinstance(actual[name], int) for name in names)
        and expected == actual
    )


def _stress_covers_all_splits(per_root: Any, *, minimum_pairs: int) -> bool:
    """Require collision evidence for every split used by the V6 protocol."""

    if not isinstance(per_root, dict) or not per_root:
        return False
    observed = {str(key).split("/", 1)[0] for key in per_root}
    return (
        set(REQUIRED_STRESS_SPLITS) <= observed
        and all(
            isinstance(value, dict) and int(value.get("summary_collision_pair_count", 0)) >= minimum_pairs
            for value in per_root.values()
        )
    )


def _alias_split_is_exact(aliases: Any, split: str) -> bool:
    """Validate one pre-outcome summary-control alias report.

    The runner checks both calibration and sealed outcome candidates before
    materializing outcome terminals.  The independent auditor must preserve
    that same scope; auditing outcome alone leaves calibration-arm selection
    vulnerable to a hidden candidate-identity side channel.
    """

    if not isinstance(aliases, dict):
        return False
    record = aliases.get(split)
    if not isinstance(record, dict) or record.get("checked_before_outcome_execution") is not True:
        return False
    tolerance = record.get("score_tolerance")
    per_root = record.get("per_root")
    if not isinstance(tolerance, (float, int)) or float(tolerance) < 0.0 or not isinstance(per_root, dict) or not per_root:
        return False
    return all(
        isinstance(value, dict)
        and float(value.get("maximum_summary_cost_score_delta", float("inf"))) <= float(tolerance)
        and float(value.get("maximum_summary_endpoint_score_delta", float("inf"))) <= float(tolerance)
        for value in per_root.values()
    )


def audit(report_path: Path) -> dict[str, Any]:
    """Verify evidence eligibility without estimating any new outcome effect."""

    report = _load(report_path)
    split = report.get("split_isolation", {})
    stress = report.get("action_representation_stress", {})
    aliases = report.get("summary_prediction_aliasing_pre_outcome", {})
    ensemble_diagnostics = report.get("endpoint_ensemble_diagnostics_pre_outcome", {})
    control = report.get("control_contract", {})
    arms = report.get("arms", {})
    policy = report.get("calibration", {}).get("policy", {})
    outcome = report.get("outcome", {})
    expected_ordered_fields = ["source_history_model", "goal_raw", "action_sequence"]
    expected_summary_fields = ["source_history_model", "goal_raw", "action_sequence_summary"]
    ordered_names = ("cost_only", "planning_shaped_auxiliary", "counterfactual_endpoint")
    summary_names = ("summary_cost_only", "summary_counterfactual_endpoint")
    per_root_stress = stress.get("per_root", {}) if isinstance(stress, dict) else {}
    minimum_collision_pairs = int(stress.get("minimum_summary_collision_pairs_per_root", 1)) if isinstance(stress, dict) else 1
    checks = {
        "report_schema": report.get("schema") == REPORT_SCHEMA,
        "status_completed": report.get("status") == "completed_counterfactual_endpoint_control",
        "root_disjoint": split.get("root_disjoint") is True,
        "shared_candidate_support": split.get("candidate_generalization_regime") == "shared_candidate_in_support",
        "goal_balanced_candidate_design": split.get("candidate_design") == "goal_balanced_interaction_pairs",
        "frozen_preflight_ledger_bound": isinstance(split.get("frozen_preflight_ledger_sha256"), str)
        and len(split["frozen_preflight_ledger_sha256"]) == 64
        and isinstance(split.get("frozen_preflight_ledger_context_count"), int)
        and split["frozen_preflight_ledger_context_count"] > 0,
        "frozen_preflight_ledger_hash_chain": _frozen_preflight_binding_is_intact(report, split),
        "calibration_role_partition_bound": _calibration_role_binding_is_intact(report, split),
        "goal_balanced_entropy": float(split.get("goal_balanced_design_winner_entropy_normalized", -1.0)) >= float(
            split.get("minimum_goal_balanced_design_winner_entropy_normalized", 0.5)
        ),
        "strict_collision_stress": _stress_covers_all_splits(per_root_stress, minimum_pairs=minimum_collision_pairs),
        "summary_alias_checked_pre_outcome": all(
            isinstance(aliases, dict) and isinstance(aliases.get(split), dict)
            and aliases[split].get("checked_before_outcome_execution") is True
            for split in REQUIRED_ALIAS_SPLITS
        ),
        "summary_alias_preserved": all(_alias_split_is_exact(aliases, split) for split in REQUIRED_ALIAS_SPLITS),
        "endpoint_ensemble_diagnostics_pre_outcome": all(
            isinstance(ensemble_diagnostics, dict)
            and isinstance(ensemble_diagnostics.get(split), dict)
            and ensemble_diagnostics[split].get("available") is True
            and isinstance(ensemble_diagnostics[split].get("root_count"), int)
            and ensemble_diagnostics[split]["root_count"] > 0
            and isinstance(ensemble_diagnostics[split].get("per_root"), dict)
            and ensemble_diagnostics[split]["per_root"]
            for split in REQUIRED_ALIAS_SPLITS
        ),
        "primary_policy_only_ordered_endpoint_vs_cost": control.get("primary_authorization_holds_ordered_action_representation_fixed") is True
        and control.get("factorial_diagnostic", {}).get("selection_uses_ordered_endpoint_vs_ordered_cost_only") is True,
        "endpoint_ensemble_diagnostics_not_policy_input": control.get("endpoint_ensemble_disagreement_diagnostic_only") is True,
        "ordered_arm_contracts": all(
            arms.get(name, {}).get("action_representation") == "full_ordered_20x2"
            and arms.get(name, {}).get("input_fields") == expected_ordered_fields
            for name in ordered_names
        ),
        "summary_arm_contracts": all(
            arms.get(name, {}).get("action_representation") == "canonical_summary_mean_std_first_last"
            and arms.get(name, {}).get("input_fields") == expected_summary_fields
            for name in summary_names
        ),
        "policy_authorized_or_fallback": policy.get("deployment_policy") in {"cost", "endpoint"}
        and outcome.get("authorized_deployment_policy") == policy.get("deployment_policy"),
        "outcomes_deferred": split.get("outcome_labels_sealed_before_execution") is True
        and control.get("outcome_execution_deferred_until_policy_and_contrast_bound_frozen") is True,
    }
    return {
        "schema": SCHEMA,
        "status": "passed_contract_audit" if all(checks.values()) else "rejected_contract_audit",
        "checks": checks,
        "claim_boundary": (
            "Protocol eligibility audit only. A pass does not establish endpoint efficacy, "
            "calibration validity beyond its declared protocol, or online planning benefit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite contract audit: {args.output}")
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_contract_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
