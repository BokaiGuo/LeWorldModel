"""Non-neural, non-physical tests for verified 40-to-50 continuation."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_fixed_h3_50_20261101_codex as ctrl


def test_carry_uses_real_40_45_lineage_and_appends_only_zero():
    original = np.arange(30, dtype=np.float32).reshape(3, 5, 2) / 30
    previous = ctrl.carry.BoundPlan.seal("native", 35, original)
    for boundary in ctrl.BOUNDARIES:
        extended, linked = ctrl.extended_carry(previous, episode="native", boundary=boundary,
                                                issued=previous.plan[0], pixel_sha="a" * 64)
        np.testing.assert_array_equal(extended[:3], previous.plan)
        assert extended.shape == (4, 5, 2) and not extended[-1].any()
        assert linked["boundary"] == boundary
        with pytest.raises(ValueError):
            ctrl.extended_carry(previous, episode="native", boundary=boundary,
                                issued=previous.plan[0] + .1, pixel_sha="a" * 64)
        previous = ctrl.carry.BoundPlan.seal("native", boundary, original)


def test_task_actual_history_and_no_fake_tail():
    runner = SimpleNamespace(rgb=SimpleNamespace(moving=SimpleNamespace(
        normalized=lambda task, raw: np.asarray(raw).reshape(2, 10).tolist())),
        p3=SimpleNamespace(core=SimpleNamespace(validate_task=lambda task: None)))
    base = dict(root_context={}, binding=dict(task_id="old"))
    raw = np.arange(20, dtype=np.float32).reshape(10, 2) / 20
    extended = np.zeros((4, 5, 2), np.float32)
    extended[0] = raw[-5:]
    latents = np.ones((3, 192), np.float32)
    task = ctrl.make_task(base, latents, raw, "a" * 64, extended, 40, runner)
    assert task["root_context"]["action_history"] == raw.reshape(2, 10).tolist()
    assert task["root_context"]["latent_history"] == latents.tolist()
    with pytest.raises(ValueError, match="actual history"):
        ctrl.make_task(base, latents, np.zeros((10, 2), np.float32), "a"*64, extended, 40, runner)
    extended[-1, 0, 0] = .1
    with pytest.raises(ValueError, match="hold-extended"):
        ctrl.make_task(base, latents, raw, "a"*64, extended, 45, runner)


def test_censored_denominator_and_early_stop():
    rows = [dict(prefix=name, status="completed50", error=None) for name in ctrl.fb.PREFIXES]
    rows[2]["status"] = "inherited_censored"
    result = ctrl.study_summary(rows, None)
    assert result["status"] == "complete" and result["all_survivors_full_horizon"]
    rows[1]["status"] = "native_stopped"
    result = ctrl.study_summary(rows, None)
    assert result["status"] == "complete" and not result["all_survivors_full_horizon"]
    rows[0].update(status="failed", error={"message": "replay mismatch"})
    assert ctrl.study_summary(rows, None)["status"] == "partial"


def test_ledger_charges_are_separate_and_encode_is_not_inferred_from_node_count():
    node = dict.fromkeys(ctrl.spec.AXES, 0)
    node.update(predict_calls=390, backward_calls=240)
    encoding = dict.fromkeys(ctrl.spec.AXES, 0)
    encoding.update(encode_calls=1, encode_samples=1)
    rows = [dict(node_ledgers=[dict(account=dict(charged_upper_bounds=node))],
                 encodings=[dict(account=dict(charged_upper_bounds=encoding)),
                            dict(account=dict(charged_upper_bounds=encoding))])]
    measured = ctrl.measured_charges(rows)
    assert measured["planner"]["predict_calls"] == 390
    assert measured["planner"]["encode_calls"] == 0
    assert measured["rgb_encoding"]["encode_calls"] == 2
    assert measured["rgb_encoding"]["predict_calls"] == 0


def test_measured_ledger_requires_exact_charges(monkeypatch):
    cost = ctrl.h4.h4.node_cost("structured", 5)
    receipt = dict(journal="synthetic", sha256="d"*64)
    class Driver:
        def finish(self):
            return receipt
    actual = dict(status="closed_unverified", charged_upper_bounds=cost.copy())
    monkeypatch.setattr(ctrl.fb.own.consumer, "consume", lambda *args: actual)
    assert ctrl.fb.account(Driver(), {"fixture": True}, "fixture_node", cost) == (receipt, actual)
    actual["charged_upper_bounds"]["predict_calls"] -= 1
    with pytest.raises(ValueError, match="measured durable charges"):
        ctrl.fb.account(Driver(), {"fixture": True}, "fixture_node", cost)
