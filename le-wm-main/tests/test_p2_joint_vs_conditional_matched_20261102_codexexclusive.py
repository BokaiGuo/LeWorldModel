import numpy as np
import pytest
import json
import torch

from experiments.spectral_reachability import p2_joint_vs_conditional_matched_20261102_codexexclusive as m
from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as ledger


def test_initializers_are_both_used_by_real_optimizer_core():
    bank = [np.full((10, 2), 0.1, dtype=np.float32).tobytes()]
    seen = []
    def init(arm, action, index):
        assert index == 0
        return action + np.float32(0.1 if arm == m.ARMS[0] else -0.1)
    def evaluate(arm, action, gradient):
        seen.append((arm, action.copy(), gradient))
        return float(np.square(action).sum()), (np.zeros_like(action) if gradient else None)
    result = m.optimize_schedule(bank, init, evaluate, lambda event: None)
    assert len(result) == 1 and len(seen) == 18
    assert np.allclose(seen[0][1], .2) and np.allclose(seen[9][1], 0.)
    assert result[0]["arms"][0]["attempted_score_calls"] == 9


def test_invalid_bank_or_initializers_fail_before_evaluation():
    seen = []
    bank = [np.zeros((10, 2), np.float32).tobytes()]
    def bad(arm, action, index):
        return np.full((10, 2), np.nan, np.float32)
    with pytest.raises(ValueError):
        m.optimize_schedule(bank, bad, lambda *a: seen.append(a), lambda e: None)
    assert not seen


def test_meter_counts_nested_hooks_and_rejects_extra_before_dispatch(tmp_path):
    journal = ledger.Ledger(tmp_path / "events")
    meter = m.MatchedMeter(journal)
    called = []
    try:
        with meter.ledger.phase("composition_inherited"):
            for _ in range(m.CAPS["composition_inherited"]["init_transition"]):
                meter.ledger.call("init_transition", lambda: meter.call("predict", lambda: 1))
            with pytest.raises(ValueError, match="cap"):
                meter.ledger.call("init_transition", lambda: called.append(1))
        assert not called and meter.failed
        with pytest.raises(ValueError, match="incomplete"):
            meter.finish()
    finally:
        journal.close()
    assert not ledger.verify_ledger(tmp_path / "events")["outstanding"]


def test_refuses_cpu_neural_entry():
    with pytest.raises(ValueError, match="CUDA"):
        m.run_native(None, None, None, "cpu")
    with pytest.raises(ValueError, match="CUDA"):
        m.initialize_native(None, None, None, None, "cpu", m.ARMS[0], 0)


def test_failed_operation_latches_and_prohibits_later_work(tmp_path):
    journal = ledger.Ledger(tmp_path / "failure")
    meter = m.MatchedMeter(journal)
    try:
        with meter.phase("sra_initialization"):
            with pytest.raises(RuntimeError, match="broken"):
                meter.ledger.call("init_jacobian", lambda: (_ for _ in ()).throw(RuntimeError("broken")))
            with pytest.raises(ValueError, match="failed"):
                meter.ledger.call("init_transition", lambda: pytest.fail("dispatch after failure"))
    finally:
        journal.close()
    records = ledger.verify_ledger(tmp_path / "failure")
    assert records["counts"]["sra_initialization", "init_jacobian"] == {
        "attempt": 1, "returned": 0, "failed": 1}


def test_actual_initializer_paths_use_jacobian_control_updates_without_neural_model(tmp_path):
    context = {"latent_history": np.zeros((3, 192)).tolist(),
               "action_history": np.zeros((2, 10)).tolist()}
    fixed = {"context_json": json.dumps(context), "goal_json": json.dumps([1.] * 192)}
    original = np.zeros((10, 2), np.float32)
    def analytic_transition(state, history, action):
        signal = action.reshape(10).sum()
        successor = state[:, -1] + signal.expand(1, 192)
        return torch.cat([state[:, 1:], successor[:, None]], dim=1), history
    journal = ledger.Ledger(tmp_path / "analytic")
    meter = m.MatchedMeter(journal)
    try:
        comp = m._initialize_with_transition(fixed, original, analytic_transition, meter, "cpu", m.ARMS[0], 0)
        joint = m._initialize_with_transition(fixed, original, analytic_transition, meter, "cpu", m.ARMS[1], 0)
    finally:
        journal.close()
    assert np.array_equal(comp[:5], original[:5])
    assert np.any(comp[5:] != original[5:]) and np.any(joint[:5] != original[:5])
    for phase in m.INIT_CATEGORY.values():
        assert meter.attempted[phase]["init_transition"] == 2
        assert meter.attempted[phase]["init_jacobian"] == meter.attempted[phase]["init_bounded_solve"] == 1
    records = [json.loads(line)["record"] for line in (tmp_path / "analytic").read_text().splitlines()]
    dims = [r["payload"]["jacobian_shape"] for r in records if r["kind"] == "init_returned"]
    assert dims == [[192, 10], [192, 20]]
    assert not ledger.verify_ledger(tmp_path / "analytic")["outstanding"]


def test_actual_native_objective_api_routes_operation_calls_with_analytic_transition(tmp_path):
    fixed = m.native.verified_input()
    journal = ledger.Ledger(tmp_path / "objective")
    meter = m.MatchedMeter(journal)
    def analytic_transition(state, history, action):
        token = action.reshape(1, 10)
        projected = torch.cat((token, token.new_zeros((1, 182))), dim=1)
        return torch.cat((state[:, 1:], (state[:, -1] + projected)[:, None]), dim=1), history
    objective = m.native.make_objective(fixed, analytic_transition, meter, torch.device("cpu"))
    action = np.frombuffer(fixed["bank"][0], dtype="<f4").reshape(10, 2).copy()
    try:
        score, gradient = objective(m.ARMS[0], json.loads(fixed["context_json"]),
                                    json.loads(fixed["goal_json"]), action, True)
    finally:
        journal.close()
    assert np.isfinite(score) and gradient.shape == (10, 2)
    assert meter.attempted["common_composition"]["transition_packet"] == 2
    assert meter.attempted["common_composition"]["autograd_grad"] == 1
    assert not ledger.verify_ledger(tmp_path / "objective")["outstanding"]


def test_one_real_bank_candidate_end_to_end_with_only_analytic_transition(tmp_path):
    fixed = m.native.verified_input()
    journal = ledger.Ledger(tmp_path / "one_pair")
    meter = m.MatchedMeter(journal)
    def analytic_transition(state, history, action):
        token = action.reshape(1, 10)
        projected = torch.cat((token, token.new_zeros((1, 182))), dim=1)
        return torch.cat((state[:, 1:], (state[:, -1] + projected)[:, None]), dim=1), history
    objective = m.native.make_objective(fixed, analytic_transition, meter, torch.device("cpu"))
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    try:
        rows = m.optimize_schedule(
            fixed["bank"][:1],
            lambda arm, action, index: m._initialize_with_transition(
                fixed, action, analytic_transition, meter, "cpu", arm, index),
            lambda arm, action, grad: objective(arm, context, goal, action, grad),
            lambda event: meter.ledger.record("optimizer_" + event["kind"], event))
    finally:
        journal.close()
    assert len(rows) == 1 and len(rows[0]["arms"]) == 2
    for arm in m.ARMS:
        assert meter.attempted[m.INIT_CATEGORY[arm]]["init_transition"] == 2
        assert meter.attempted["common_" + arm]["transition_packet"] == 18
        assert meter.attempted["common_" + arm]["autograd_grad"] == 8
    assert not ledger.verify_ledger(tmp_path / "one_pair")["outstanding"]
