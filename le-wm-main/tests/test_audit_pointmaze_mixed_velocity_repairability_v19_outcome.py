from __future__ import annotations

import inspect

from experiments.gdd_wm import audit_pointmaze_mixed_velocity_repairability_v19_outcome as audit


def test_v19_audit_is_cuda_only_and_binds_the_fixed_visual_motion_policy() -> None:
    assert audit.POLICY["motion_source"] == "source_history_visual_state"
    assert audit.POLICY["threshold"] == 0.02
    source = inspect.getsource(audit.audit)
    assert "no CPU fallback" in source
    assert "generated_after_frozen_selection_sha256" in source
    assert "selected_policy" in source


def test_v19_audit_reports_the_aggregate_and_both_predeclared_route_strata() -> None:
    source = inspect.getsource(audit.audit)
    assert "history_routed" in inspect.getsource(audit._strata)
    assert "action_fallback" in inspect.getsource(audit._strata)
    assert "routed_over_action_only" in source
    assert "routed_over_recurrent" in source
    assert "repair_gain_heterogeneity" in source
    assert "recurrent_dominates_route" in source
    assert "strata" in source
    assert "_paired_root_bootstrap" in inspect.getsource(audit._gain)


def test_v19_audit_exposes_mechanism_heterogeneity_and_never_overwrites_an_audit() -> None:
    source = inspect.getsource(audit._gain_difference)
    assert "history-routed roots than action-fallback roots" in source
    main_source = inspect.getsource(audit.main)
    assert "--output" in main_source
    assert "refusing to overwrite immutable V19 outcome audit" in main_source


def test_v19_audit_supports_a_predeclared_equal_budget_control() -> None:
    source = inspect.getsource(audit.audit)
    assert "comparison_policy" in source
    assert "equal-budget control does not use the same recurrent fraction" in source
    assert "route_beats_equal_budget_control" in source


def test_v19_audit_does_not_mislabel_arbitrary_policy_comparisons_as_recurrent() -> None:
    source = inspect.getsource(audit._gain)
    assert "first-listed policy" in source
