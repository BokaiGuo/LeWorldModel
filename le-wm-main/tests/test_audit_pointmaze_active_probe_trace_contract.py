from __future__ import annotations

from experiments.audit_pointmaze_active_probe_trace_contract import audit


def _row(method: str, *, probe_design: str = "task_aligned_positive_half_plane_with_predicted_progress_gate") -> dict:
    active = method == "effect_adapter_rgb_online_verified_local"
    return {
        "episode_id": 7,
        "method": method,
        "blocks_executed": 20 if active else 24,
        "warmup_blocks": 1,
        "active_probe_blocks": 4 if active else 0,
        "phase_reprobe_blocks": 0,
        "probe_triggered": active,
        "final_distance": 1.0,
        "success": False,
        "active_probe_trace": [
            {
                "probe_index": index,
                "probe_design": probe_design,
                "action_goal_projection": 0.1,
                "predicted_goal_progress": 0.1,
                "observed_goal_progress": 0.1,
            }
            for index in range(4)
        ] if active else [],
        "policy_contract": {
            "simulator_state_in_policy_input": False,
            "strict_shared_root_candidate_bank": True,
            "candidate_bank_digest": "shared",
            "candidate_actions": [[0.0, 0.0]],
        },
    }


def _report(*, include_budget: bool = True) -> dict:
    methods = (
        "visual_dynamics",
        "effect_adapter_rgb",
        "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_online_verified_local",
        "effect_adapter_rgb_htsc_consistency",
    )
    split_protocol = {"held_out_test_roots": [7]}
    if include_budget:
        split_protocol["total_physical_block_budget"] = 25
    return {
        "schema": "pointmaze.htsc_active_probe.strict_rgb_closed_loop.v1",
        "split_protocol": split_protocol,
        "frozen_probe": {"probe_directions": 4, "probe_magnitude": 0.5},
        "artifacts": {
            "held_out_arm_reports": {method: f"unused/{method}.json" for method in methods},
        },
    }


def test_trace_audit_rejects_a_raw_cardinal_probe_even_if_aggregate_labels_it_active(monkeypatch) -> None:
    methods = (
        "visual_dynamics", "effect_adapter_rgb", "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_online_verified_local", "effect_adapter_rgb_htsc_consistency",
    )
    payload = _report()
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    def raw_rows(_report: dict) -> list[dict]:
        rows = [_row(method, probe_design="cardinal_full_plane") for method in methods]
        for row in rows:
            for event in row["active_probe_trace"]:
                event.pop("predicted_goal_progress")
        return rows

    monkeypatch.setattr(module, "_raw_rows", raw_rows)
    result = audit(payload)
    assert result["status"] == "rejected_trace_contract"
    assert "declared task-safe probe geometry" in result["reason"]
    assert result["raw_trace_evidence"]["probe_design_counts"] == {"cardinal_full_plane": 4}


def test_trace_audit_reports_missing_budget_and_raw_geometry_together(monkeypatch) -> None:
    methods = (
        "visual_dynamics", "effect_adapter_rgb", "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_online_verified_local", "effect_adapter_rgb_htsc_consistency",
    )
    payload = _report(include_budget=False)
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    def raw_rows(_report: dict) -> list[dict]:
        rows = [_row(method, probe_design="cardinal_full_plane") for method in methods]
        for row in rows:
            for event in row["active_probe_trace"]:
                event.pop("predicted_goal_progress")
        return rows

    monkeypatch.setattr(module, "_raw_rows", raw_rows)
    result = audit(payload)
    assert result["status"] == "rejected_trace_contract"
    assert "shared physical block budget" in result["reason"]
    assert "declared task-safe probe geometry" in result["reason"]
    assert "lacks predicted_goal_progress" in result["reason"]


def test_trace_audit_accepts_budget_contract_as_the_authoritative_budget_location(monkeypatch) -> None:
    methods = (
        "visual_dynamics", "effect_adapter_rgb", "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_online_verified_local", "effect_adapter_rgb_htsc_consistency",
    )
    payload = _report(include_budget=False)
    payload["budget_contract"] = {"total_physical_block_budget": 25}
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    monkeypatch.setattr(module, "_raw_rows", lambda _report: [_row(method) for method in methods])
    result = audit(payload)
    assert result["status"] == "passed_trace_contract"
    assert result["expected_contract"]["shared_total_block_budget"] == 25


def test_trace_audit_accepts_a_fully_auditable_task_safe_probe(monkeypatch) -> None:
    methods = (
        "visual_dynamics", "effect_adapter_rgb", "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_online_verified_local", "effect_adapter_rgb_htsc_consistency",
    )
    payload = _report()
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    monkeypatch.setattr(module, "_raw_rows", lambda _report: [_row(method) for method in methods])
    result = audit(payload)
    assert result["status"] == "passed_trace_contract"
    assert result["active_root_count"] == 1


def test_trace_audit_binds_decision_ambiguity_execution_to_the_frozen_margin(monkeypatch) -> None:
    methods = (
        "visual_dynamics", "effect_adapter_rgb", "effect_adapter_rgb_htsc",
        "effect_adapter_rgb_online_verified_local", "effect_adapter_rgb_htsc_consistency",
    )
    payload = _report()
    payload["frozen_probe"].update({
        "active_probe_trigger": "conditional_decision_ambiguity",
        "active_probe_ambiguity_margin": 0.03,
    })
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    rows = [_row(method) for method in methods]
    active = next(row for row in rows if row["method"] == "effect_adapter_rgb_online_verified_local")
    active["policy_contract"].update({
        "active_probe_trigger": "conditional_decision_ambiguity",
        "active_probe_ambiguity_margin": 0.03,
    })
    active["active_probe_trace"].insert(0, {
        "probe_decision": True,
        "dry_run_decision_margin": 0.02,
        "dry_run_decision_ambiguity_margin": 0.03,
    })
    monkeypatch.setattr(module, "_raw_rows", lambda _report: rows)
    assert audit(payload)["status"] == "passed_trace_contract"

    active["active_probe_trace"][0]["dry_run_decision_margin"] = 0.04
    rejected = audit(payload)
    assert rejected["status"] == "rejected_trace_contract"
    assert "probed outside its frozen margin" in rejected["reason"]


def test_affine_trace_contract_binds_finite_frozen_loo_threshold(monkeypatch) -> None:
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    monkeypatch.setattr(module, "ACTIVE_METHOD", "effect_adapter_rgb_online_affine_verified_local")
    row = _row("effect_adapter_rgb_online_affine_verified_local")
    row["policy_contract"]["local_chart_loo_tolerance"] = 0.1
    row["trace"] = [{"action_transfer_gate": {
        "local_affine_recovery_considered": True,
        "local_affine_chart_loo_tolerance": 0.1,
    }}]
    assert module._affine_loo_contract_violations(
        [row], {"local_chart_loo_tolerance": 0.1},
    ) == []
    row["trace"][0]["action_transfer_gate"]["local_affine_chart_loo_tolerance"] = 0.2
    assert module._affine_loo_contract_violations(
        [row], {"local_chart_loo_tolerance": 0.1},
    ) == ["affine local release did not use its frozen LOO threshold"]


def test_trace_audit_reports_candidate_anchored_geometry_for_v5(monkeypatch) -> None:
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    monkeypatch.setattr(
        module,
        "ACTIVE_METHOD",
        "effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
    )
    assert module._expected_probe_design() == "candidate_anchored_task_safe_affine_d_optimal"
    violations = module._raw_trace_violations({
        "raw_probe_event_count": 1,
        "probe_design_counts": {
            "task_aligned_positive_half_plane_with_predicted_progress_gate": 1,
        },
        "missing_predicted_goal_progress_count": 0,
        "nonpositive_predicted_goal_progress_count": 0,
        "nonpositive_action_goal_projection_count": 0,
    })
    assert "method-specific frozen probe geometry" in "; ".join(violations)


def test_trace_audit_records_only_genuine_runtime_safety_rejections() -> None:
    import experiments.audit_pointmaze_active_probe_trace_contract as module

    safe_stop = {
        "raw_probe_event_count": 0,
        "probe_design_counts": {},
        "missing_predicted_goal_progress_count": 0,
        "nonpositive_predicted_goal_progress_count": 0,
        "nonpositive_action_goal_projection_count": 0,
        "runtime_safety_rejection_count": 1,
        "invalid_runtime_safety_rejection_count": 0,
    }
    assert module._raw_trace_violations(safe_stop) == []
    unsafe_label = {**safe_stop, "invalid_runtime_safety_rejection_count": 1}
    assert module._raw_trace_violations(unsafe_label) == [
        "runtime task-safety rejection did not record an unsafe pulse"
    ]
