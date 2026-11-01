"""Pure CPU carry and controller-contract fixtures."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_fixed_h3_40_20261101_codex as controller


def test_real_boundary_lineage_and_zero_hold_extension():
    old = np.arange(30, dtype=np.float32).reshape(3, 5, 2) / 30
    plan = controller.carry.BoundPlan.seal("zero", 5, old)
    extended, linked = controller.extended_carry(plan, episode="zero", boundary=10,
                                                issued=old[0], pixel_sha="a"*64)
    assert extended.shape == (4, 5, 2)
    np.testing.assert_array_equal(extended[:3], old)
    assert not extended[3].any()
    assert linked["boundary"] == 10 and linked["previous_plan_sha256"] == plan.plan_sha256
    assert not np.shares_memory(old, extended)
    with pytest.raises(ValueError):
        controller.extended_carry(plan, episode="zero", boundary=15, issued=old[0], pixel_sha="a"*64)
    with pytest.raises(ValueError):
        controller.extended_carry(plan, episode="zero", boundary=10,
                                  issued=np.zeros((5, 2), np.float32), pixel_sha="a"*64)


def test_each_actual_boundary_keeps_logical_h3_and_correct_predecessor():
    previous = controller.carry.BoundPlan.seal("native", 5, np.zeros((3, 5, 2), np.float32))
    for boundary in controller.BOUNDARIES:
        extended, linked = controller.extended_carry(previous, episode="native", boundary=boundary,
                                                     issued=previous.plan[0], pixel_sha="b"*64)
        assert linked["boundary"] == boundary and extended.shape == (4, 5, 2)
        previous = controller.carry.BoundPlan.seal("native", boundary, np.zeros((3, 5, 2), np.float32))
    assert controller.LOGICAL_BOUNDARY == 5
    cost = controller.h4.h4.node_cost("structured", 5)
    assert (cost["predict_calls"], cost["backward_calls"], cost["replay_calls"]) == (390, 240, 81)
    assert len(controller.BOUNDARIES) * len(controller.fb.PREFIXES) == 36


def test_task_binds_returned_history_and_zero_extension():
    runner = SimpleNamespace(rgb=SimpleNamespace(moving=SimpleNamespace(
        normalized=lambda task, raw: np.asarray(raw).reshape(2, 10).tolist())),
        p3=SimpleNamespace(core=SimpleNamespace(validate_task=lambda task: None)))
    base = dict(root_context={}, binding=dict(task_id="old"))
    raw = np.zeros((10, 2), np.float32)
    previous = np.zeros((4, 5, 2), np.float32)
    latent = np.ones((3, 192), np.float32)
    t10 = controller.task_for(base, latent, raw, "c"*64, previous, 10, runner)
    t15 = controller.task_for(base, latent, raw, "c"*64, previous, 15, runner)
    assert t10["binding"]["root_sha256"] != t15["binding"]["root_sha256"]
    assert base["root_context"] == {}
    previous[-1, 0, 0] = .5
    with pytest.raises(ValueError, match="hold-extended"):
        controller.task_for(base, latent, raw, "c"*64, previous, 10, runner)


def test_observed_early_success_and_censor_are_complete_not_full_horizon():
    rows = [dict(prefix=name, status="completed40", error=None) for name in controller.fb.PREFIXES]
    rows[0].update(status="native_stopped", stop_reason=dict(native_success=True, step=22))
    rows[1].update(status="canvas_censored", stop_reason=dict(outside=True, step=17))
    result = controller.study_summary(rows, None)
    assert result["status"] == "complete" and result["all_full_horizon"] is False
    assert result["branch_status_counts"]["native_stopped"] == 1
    assert result["branch_status_counts"]["canvas_censored"] == 1
    rows[2].update(status="failed", error=dict(message="invalid"))
    result = controller.study_summary(rows, None)
    assert result["status"] == "partial" and result["branch_status_counts"]["failed"] == 1
    assert controller.study_summary(rows[:5], None)["status"] == "partial"
