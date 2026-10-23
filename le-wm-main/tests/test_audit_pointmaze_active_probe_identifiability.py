from __future__ import annotations

import numpy as np

from experiments.audit_pointmaze_active_probe_identifiability import audit
from experiments.run_live_pointmaze_planning import VerifiedLocalActionEffectModel


def _row(
    *,
    actions: list[list[float]],
    trusted: bool,
    override: bool,
    method: str = "effect_adapter_rgb_online_verified_local",
) -> dict:
    return {
        "episode_id": 7,
        "method": method,
        "policy_contract": {
            "candidate_actions": [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [-0.2, 0.0]],
        },
        "active_probe_trace": [
            {
                "probe_index": index,
                "action": action,
                "residual_trusted": trusted if index == len(actions) - 1 else False,
                "verification_error": 0.1,
            }
            for index, action in enumerate(actions)
        ],
        "trace": [{"action_transfer_gate": {
            "effect_policy_released": False,
            "local_action_support_passed": True,
            "fallback_reason": (
                "global_teacher_bound_failed_local_verified_override" if override else "local_gate_failed"
            ),
        }}],
    }


def test_audit_identifies_legacy_residual_underdetermination_despite_affine_rank() -> None:
    result = audit({}, [_row(
        actions=[[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]],
        trusted=True,
        override=False,
    )])
    root = result["per_root"][0]
    assert root["affine_design_rank"] == 3
    assert root["legacy_residual_design_rank"] < root["legacy_residual_feature_count"]
    assert root["failure_mode"] == "legacy_residual_basis_underdetermined"


def test_audit_prioritizes_no_probe_over_model_diagnosis() -> None:
    row = _row(actions=[], trusted=False, override=False)
    result = audit({}, [row])
    assert result["per_root"][0]["failure_mode"] == "probe_not_triggered_or_early_rejected"


def test_audit_records_an_exercised_override_without_calling_it_task_success() -> None:
    result = audit({}, [_row(
        actions=[[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]],
        trusted=True,
        override=True,
    )])
    assert result["per_root"][0]["local_override_count"] == 1
    assert result["summary"]["local_override_exercised_roots"] == 1
    assert "task benefit" in result["claim_boundary"]


def test_affine_local_chart_support_retains_only_actions_inside_probe_hull() -> None:
    chart = VerifiedLocalActionEffectModel()
    for action in ([0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]):
        chart.update(np.asarray(action), np.asarray(action))
    supported = chart.action_support_mask(np.asarray([[0.0, 0.0], [0.7, 0.0]]))
    assert chart.trusted is True
    assert supported.tolist() == [True, False]


def test_affine_local_chart_diagnostic_is_available_before_planning() -> None:
    chart = VerifiedLocalActionEffectModel()
    for action in ([0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]):
        chart.update(np.asarray(action), np.asarray(action))
    diagnostic = chart.design_diagnostics(np.asarray([[0.0, 0.0], [0.7, 0.0]]))
    assert diagnostic["design_rank"] == diagnostic["feature_count"] == 3
    assert diagnostic["candidate_hull_coverage_count"] == 1
    assert np.isfinite(float(diagnostic["leave_one_out_error"]))


def test_affine_audit_reports_crossfit_rejection_before_any_planning_outcome() -> None:
    row = _row(
        actions=[[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]],
        trusted=True,
        override=False,
        method="effect_adapter_rgb_online_affine_verified_local",
    )
    row["active_probe_trace"].append({
        "preplanning_affine_repairability": True,
        "local_chart_loo_reliable": False,
    })
    result = audit({}, [row])
    assert result["per_root"][0]["failure_mode"] == "affine_chart_crossfit_unreliable"


def test_affine_audit_records_candidate_anchored_bank_membership() -> None:
    actions = [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [-0.2, 0.0]]
    row = _row(
        actions=actions,
        trusted=True,
        override=False,
        method="effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
    )
    for index, event in enumerate(row["active_probe_trace"]):
        event.update({
            "probe_design": "candidate_anchored_task_safe_affine_d_optimal",
            "probe_candidate_index": index,
        })
    result = audit({}, [row])
    root = result["per_root"][0]
    assert root["candidate_anchored_probe_event_count"] == 4
    assert root["candidate_anchored_probe_membership_passed"] is True
    assert result["summary"]["candidate_anchored_probe_membership_failed_roots"] == 0
