from __future__ import annotations

import pytest

from experiments.run_pointmaze_htsc_active_probe_strict import (
    ACTIVE_METHOD,
    DEPLOYABLE_METHODS,
    expected_probe_design_for_method,
    TOTAL_BLOCK_BUDGET,
    active_probe_mechanism_diagnostics,
    active_planning_budget,
    candidate_anchored_repairable_candidates,
    choose_calibration_only_probe_config,
    _run_live,
    validate_active_probe_report,
)


def test_active_probe_budget_charges_each_observed_probe_to_shared_cap() -> None:
    # One RGB-history warm-up pulse is charged to every arm before planning.
    assert active_planning_budget(TOTAL_BLOCK_BUDGET, 4) == TOTAL_BLOCK_BUDGET - 1 - 4
    assert active_planning_budget(TOTAL_BLOCK_BUDGET, 6) == TOTAL_BLOCK_BUDGET - 1 - 6
    with pytest.raises(ValueError, match="exceeds"):
        active_planning_budget(TOTAL_BLOCK_BUDGET, TOTAL_BLOCK_BUDGET - 1)


def test_candidate_anchored_arm_has_a_distinct_frozen_probe_geometry() -> None:
    assert expected_probe_design_for_method(
        "effect_adapter_rgb_online_affine_candidate_anchored_verified_local"
    ) == "candidate_anchored_task_safe_affine_d_optimal"
    assert expected_probe_design_for_method(
        "effect_adapter_rgb_online_affine_verified_local"
    ) == "task_aligned_positive_half_plane_with_predicted_progress_gate"


def test_candidate_anchored_selection_seals_heldout_without_repairable_calibration() -> None:
    candidates = [
        {
            "candidate_anchored_probe_root_count": 1,
            "candidate_anchored_probe_coverage": 1.0,
            "preplanning_repairable_root_count": 0,
            "candidate_anchored_repairable_root_count": 0,
        },
        {
            "candidate_anchored_probe_root_count": 0,
            "candidate_anchored_probe_coverage": 0.0,
            "preplanning_repairable_root_count": 3,
            "candidate_anchored_repairable_root_count": 0,
        },
    ]
    with pytest.raises(RuntimeError, match="held-out roots remain sealed"):
        candidate_anchored_repairable_candidates(candidates)
    assert candidate_anchored_repairable_candidates([
        *candidates,
        {
            "candidate_anchored_probe_root_count": 2,
            "candidate_anchored_probe_coverage": 1.0,
            "preplanning_repairable_root_count": 1,
            "candidate_anchored_repairable_root_count": 1,
        },
    ])[-1]["candidate_anchored_repairable_root_count"] == 1


def test_candidate_anchored_selection_rejects_a_lucky_low_coverage_chart() -> None:
    low_coverage = {
        "candidate_anchored_probe_root_count": 1,
        "candidate_anchored_probe_coverage": 1.0 / 16.0,
        "preplanning_repairable_root_count": 1,
        "candidate_anchored_repairable_root_count": 1,
    }
    with pytest.raises(RuntimeError, match="sufficiently covered"):
        candidate_anchored_repairable_candidates([low_coverage])
    accepted = candidate_anchored_repairable_candidates([{
        **low_coverage,
        "candidate_anchored_probe_root_count": 12,
        "candidate_anchored_probe_coverage": 0.75,
    }])
    assert accepted[0]["candidate_anchored_probe_coverage"] == 0.75


def test_v5_calibration_counts_only_exact_candidate_anchored_repairability(monkeypatch) -> None:
    import experiments.run_pointmaze_htsc_active_probe_strict as module

    method = "effect_adapter_rgb_online_affine_candidate_anchored_verified_local"
    monkeypatch.setattr(module, "ACTIVE_METHOD", method)
    actions = [[0.2, 0.0], [0.0, 0.2], [0.2, 0.1]]
    active_row = {
        "episode_id": 7,
        "method": method,
        "final_distance": 1.0,
        "success": False,
        "policy_contract": {"candidate_actions": actions},
        "active_probe_trace": [
            {
                "probe_index": index,
                "probe_design": "candidate_anchored_task_safe_affine_d_optimal",
                "probe_candidate_index": index,
                "action": action,
            }
            for index, action in enumerate(actions)
        ] + [{
            "preplanning_affine_repairability": True,
            "repairable_before_planning": True,
        }],
    }
    baseline = {"rows": [{
        "episode_id": 7,
        "method": module.HTSC_METHOD,
        "final_distance": 1.0,
        "success": False,
    }]}
    candidate = module._calibration_candidate(
        baseline=baseline,
        active={"rows": [active_row]},
        probe_directions=4,
        probe_magnitude=0.5,
        active_probe_trigger="conditional_state_support_failure",
        active_probe_ambiguity_margin=0.0,
        local_chart_loo_tolerance=0.1,
        artifact_name="fixture",
    )
    assert candidate["candidate_anchored_probe_root_count"] == 1
    assert candidate["candidate_anchored_probe_coverage"] == 1.0
    assert candidate["candidate_anchored_repairable_root_count"] == 1

    active_row["active_probe_trace"][0]["action"] = [0.19, 0.0]
    mismatch = module._calibration_candidate(
        baseline=baseline,
        active={"rows": [active_row]},
        probe_directions=4,
        probe_magnitude=0.5,
        active_probe_trigger="conditional_state_support_failure",
        active_probe_ambiguity_margin=0.0,
        local_chart_loo_tolerance=0.1,
        artifact_name="fixture_mismatch",
    )
    assert mismatch["candidate_anchored_probe_root_count"] == 0
    assert mismatch["candidate_anchored_repairable_root_count"] == 0


def test_calibration_selector_never_reads_test_outcomes_and_requires_paired_gain() -> None:
    candidates = [
        {
            "probe_directions": 4,
            "calibration_only": True,
            "paired_success_delta_vs_htsc": 0.0,
            "paired_final_distance_delta_vs_htsc": -0.03,
        },
        {
            "probe_directions": 6,
            "calibration_only": True,
            "paired_success_delta_vs_htsc": 0.0,
            "paired_final_distance_delta_vs_htsc": 0.02,
        },
    ]
    selected = choose_calibration_only_probe_config(candidates)
    assert selected["probe_directions"] == 4
    assert selected["trigger"] == "conditional_state_support_failure_then_local_rerank"
    assert selected["outcome_test_used_for_selection"] is False

    candidates[0]["calibration_only"] = False
    with pytest.raises(ValueError, match="calibration-only"):
        choose_calibration_only_probe_config(candidates)


def test_calibration_selector_rejects_success_tie_with_distance_harm() -> None:
    selected = choose_calibration_only_probe_config([
        {
            "probe_directions": 4,
            "probe_magnitude": 0.25,
            "calibration_only": True,
            "paired_success_delta_vs_htsc": 0.0,
            "paired_final_distance_delta_vs_htsc": 0.01,
        },
    ])
    assert selected["selection_authorized"] is False


def test_calibration_selector_can_freeze_decision_ambiguity_trigger_without_outcome_access() -> None:
    selected = choose_calibration_only_probe_config([
        {
            "probe_directions": 4,
            "probe_magnitude": 0.5,
            "active_probe_trigger": "conditional_state_support_failure",
            "active_probe_ambiguity_margin": 0.0,
            "calibration_only": True,
            "paired_success_delta_vs_htsc": 0.0,
            "paired_final_distance_delta_vs_htsc": 0.02,
        },
        {
            "probe_directions": 4,
            "probe_magnitude": 0.5,
            "active_probe_trigger": "conditional_decision_ambiguity",
            "active_probe_ambiguity_margin": 0.03,
            "calibration_only": True,
            "paired_success_delta_vs_htsc": 0.1,
            "paired_final_distance_delta_vs_htsc": -0.04,
        },
    ])
    assert selected["active_probe_trigger"] == "conditional_decision_ambiguity"
    assert selected["active_probe_ambiguity_margin"] == 0.03
    assert selected["outcome_test_used_for_selection"] is False


def test_affine_calibration_freezes_a_finite_local_loo_threshold() -> None:
    selected = choose_calibration_only_probe_config([
        {
            "probe_directions": 4,
            "probe_magnitude": 0.5,
            "local_chart_loo_tolerance": 0.20,
            "calibration_only": True,
            "paired_success_delta_vs_htsc": 0.0,
            "paired_final_distance_delta_vs_htsc": 0.0,
        },
        {
            "probe_directions": 4,
            "probe_magnitude": 0.5,
            "local_chart_loo_tolerance": 0.05,
            "calibration_only": True,
            "paired_success_delta_vs_htsc": 0.0,
            "paired_final_distance_delta_vs_htsc": 0.0,
        },
    ])
    assert selected["local_chart_loo_tolerance"] == 0.05


def test_active_probe_report_rejects_unequal_total_budget_or_state_access() -> None:
    rows = []
    for method in DEPLOYABLE_METHODS:
        active = method == ACTIVE_METHOD
        rows.append({
            "episode_id": 99,
            "method": method,
            "blocks_executed": 20 if active else 24,
            "warmup_blocks": 1,
            "active_probe_blocks": 4 if active else 0,
            "probe_triggered": bool(active),
            "phase_reprobe_blocks": 0,
            "final_distance": 0.5,
            "success": False,
            "active_probe_trace": (
                [
                    {
                        "probe_index": index,
                        "probe_design": "task_aligned_positive_half_plane_with_predicted_progress_gate",
                        "action_goal_projection": 0.1,
                        "predicted_goal_progress": 0.05,
                        "observed_goal_progress": 0.02,
                    }
                    for index in range(4)
                ] if active else []
            ),
            "policy_contract": {
                "simulator_state_in_policy_input": False,
                "strict_shared_root_candidate_bank": True,
                "candidate_bank_digest": "shared",
                "candidate_actions": [[0.0, 0.0]] * 32,
            },
        })
    report = {"rows": rows}
    validated = validate_active_probe_report(
        report,
        expected_roots=[99],
        total_block_budget=25,
        selected_probe_directions=4,
    )
    assert validated[99][ACTIVE_METHOD]["charged_blocks"] == 25

    active_row = next(row for row in report["rows"] if row["method"] == ACTIVE_METHOD)
    for event in active_row["active_probe_trace"]:
        event.update({
            "probe_design": "candidate_anchored_task_safe_affine_d_optimal",
            "probe_candidate_index": 0,
            "action": [0.0, 0.0],
        })
    with pytest.raises(ValueError, match="method-specific frozen probe geometry"):
        validate_active_probe_report(
            report,
            expected_roots=[99],
            total_block_budget=25,
            selected_probe_directions=4,
        )
    for event in active_row["active_probe_trace"]:
        event["probe_design"] = "task_aligned_positive_half_plane_with_predicted_progress_gate"
        event.pop("probe_candidate_index", None)
        event.pop("action", None)

    report["rows"][0]["policy_contract"]["simulator_state_in_policy_input"] = True
    with pytest.raises(ValueError, match="simulator state"):
        validate_active_probe_report(
            report,
            expected_roots=[99],
            total_block_budget=25,
            selected_probe_directions=4,
        )

    active_row["active_probe_trace"][0]["predicted_goal_progress"] = -0.01
    report["rows"][0]["policy_contract"]["simulator_state_in_policy_input"] = False
    with pytest.raises(ValueError, match="pre-action safety gate"):
        validate_active_probe_report(
            report,
            expected_roots=[99],
            total_block_budget=25,
            selected_probe_directions=4,
        )


def test_decision_ambiguity_validator_accepts_a_safe_skip_after_the_trigger() -> None:
    """A safe skip supplements, rather than duplicates, the margin decision."""
    import experiments.run_pointmaze_htsc_active_probe_strict as module

    method = "effect_adapter_rgb_online_verified_local"
    methods = (
        "visual_dynamics",
        "effect_adapter_rgb",
        "effect_adapter_rgb_htsc",
        method,
        "effect_adapter_rgb_htsc_consistency",
    )
    original_active, original_methods = module.ACTIVE_METHOD, module.DEPLOYABLE_METHODS
    module.ACTIVE_METHOD, module.DEPLOYABLE_METHODS = method, methods
    try:
        rows = []
        for candidate_method in methods:
            active = candidate_method == method
            rows.append({
                "episode_id": 99,
                "method": candidate_method,
                "blocks_executed": 24,
                "warmup_blocks": 1,
                "active_probe_blocks": 0,
                "probe_triggered": False,
                "phase_reprobe_blocks": 0,
                "final_distance": 0.5,
                "success": False,
                "active_probe_trace": [
                    {
                        "probe_decision": True,
                        "probe_skipped": False,
                        "reason": "episode_start_decision_ambiguity",
                        "dry_run_decision_margin": 0.02,
                        "dry_run_decision_ambiguity_margin": 0.03,
                    },
                    {
                        "probe_decision": True,
                        "probe_skipped": True,
                        "reason": "no_task_safe_probe_under_calibration_dynamics",
                    },
                ] if active else [],
                "policy_contract": {
                    "simulator_state_in_policy_input": False,
                    "strict_shared_root_candidate_bank": True,
                    "candidate_bank_digest": "shared",
                    "candidate_actions": [[0.0, 0.0]] * 32,
                    **({
                        "active_probe_trigger": "conditional_decision_ambiguity",
                        "active_probe_ambiguity_margin": 0.03,
                    } if active else {}),
                },
            })
        validated = module.validate_active_probe_report(
            {"rows": rows},
            expected_roots=[99],
            total_block_budget=25,
            selected_probe_directions=4,
            selected_probe_trigger="conditional_decision_ambiguity",
            selected_probe_ambiguity_margin=0.03,
        )
        assert validated[99][method]["charged_blocks"] == 25
    finally:
        module.ACTIVE_METHOD, module.DEPLOYABLE_METHODS = original_active, original_methods


def test_mechanism_diagnostic_distinguishes_a_probe_from_a_local_rerank_override() -> None:
    grouped = {
        7: {
            ACTIVE_METHOD: {
                "active_probe_blocks": 4,
                "probe_progress": -0.2,
                "trace": [
                    {"action_transfer_gate": {
                        "effect_policy_released": False,
                        "state_support_passed": False,
                        "action_support_passed": True,
                        "local_verified_release": False,
                    }},
                ],
            },
        },
    }
    diagnostic = active_probe_mechanism_diagnostics(grouped)
    assert diagnostic["active_probe_schedule_executed"] is True
    assert diagnostic["htsc_nonrelease_decisions"] == 1
    assert diagnostic["support_failure_nonrelease_decisions"] == 1
    assert diagnostic["repair_mechanism_exercised"] is False


def test_external_held_out_state_root_is_renderer_only_and_reaches_the_live_command(
    monkeypatch, tmp_path,
) -> None:
    import experiments.run_pointmaze_htsc_active_probe_strict as module

    output = tmp_path / "held_out.json"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        calls.append(command)
        output.write_text('{"rows": []}', encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = _run_live(
        dataset_root=tmp_path / "calibration_dataset",
        checkpoint=tmp_path / "checkpoint.pt",
        dinov2_source=tmp_path / "dinov2",
        effect_policy_checkpoint=tmp_path / "effect_policy.pt",
        output=output,
        visual_calibration_roots=[1800],
        effect_consistency_roots=None,
        test_roots=[9658],
        method="effect_adapter_rgb_online_verified_local",
        max_blocks=24,
        active_probe_directions=4,
        active_probe_magnitude=0.5,
        local_chart_loo_tolerance=0.1,
        test_state_root=tmp_path / "fresh_states",
        test_visual_source="dataset_matched_renderer",
    )
    assert result == {"rows": []}
    assert "--test-state-root" in calls[0]
    assert "--test-visual-source" in calls[0]
    assert calls[0][calls[0].index("--test-visual-source") + 1] == "dataset_matched_renderer"
    assert calls[0][calls[0].index("--local-chart-loo-tolerance") + 1] == "0.1"

    with pytest.raises(ValueError, match="external test-state root"):
        _run_live(
            dataset_root=tmp_path / "calibration_dataset",
            checkpoint=tmp_path / "checkpoint.pt",
            dinov2_source=tmp_path / "dinov2",
            effect_policy_checkpoint=tmp_path / "effect_policy.pt",
            output=output,
            visual_calibration_roots=[1800],
            effect_consistency_roots=None,
            test_roots=[9658],
            method="effect_adapter_rgb_online_verified_local",
            max_blocks=24,
            active_probe_directions=4,
            active_probe_magnitude=0.5,
            test_state_root=tmp_path / "fresh_states",
        )
