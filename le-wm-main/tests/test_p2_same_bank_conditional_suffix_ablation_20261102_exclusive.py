import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as ledger_core
from experiments.spectral_reachability import p2_same_bank_conditional_suffix_ablation_20261102_exclusive as bridge


def transition(state, history, packet):
    # A differentiable fixture in which the suffix's response depends on the
    # prefix-predicted context, so a detached or wrong mid-context is observable.
    control = packet.reshape(10)
    addition = torch.cat((control * (1 + 0.1 * state[0, -1, :10]),
                          torch.zeros(182, dtype=state.dtype, device=state.device)))
    successor = state[:, -1] + addition[None]
    token = control.reshape(1, 1, 10)
    return torch.cat((state[:, 1:], successor[:, None]), 1), torch.cat((history[:, 1:], token), 1)


def journal(path):
    return ledger_core.Ledger(path)


def test_real_bank_two_real_different_paths_cpu_fixture(tmp_path):
    fixed = bridge.native.verified_input()
    events = journal(tmp_path / "events.jsonl")
    try:
        result = bridge.run_bridge(fixed, transition, events, torch.device("cpu"))
    finally:
        events.close()
    assert len(result["rows"]) == 16
    assert any(row["composition_initial_sha256"] != row["spectral_only_initial_sha256"]
               for row in result["rows"])
    changed = sum(row["composition_initial_sha256"] != row["spectral_only_initial_sha256"]
                  for row in result["rows"])
    print(f"CPU fixture: changed initial actions {changed}/16")
    for row, source in zip(result["rows"], fixed["bank"], strict=True):
        compiled = bytes.fromhex(row["composition_initial_hex"])
        assert compiled[:40] == source[:40]
        assert row["spectral_only_initial_sha256"] == bridge._digest(source)
    for arm in bridge.native.optimizer.ARMS:
        assert result["common_callback_attempts"][arm] == {"score": 144, "gradient": 128}
        assert result["common_instrumented_attempts"][arm]["transition_packet"] == 288
    audit = ledger_core.verify_ledger(tmp_path / "events.jsonl")
    assert not audit["outstanding"]
    counts = audit["counts"]
    assert counts["composition_inherited", "conditional_prefix_transition"]["returned"] == 16
    assert counts["composition_inherited", "conditional_suffix_transition"]["returned"] == 32
    assert counts["composition_inherited", "conditional_jacobian_192_output"]["returned"] == 16
    assert counts["composition_inherited", "conditional_bounded_ridge_svd_solve"]["returned"] == 16
    records = [json.loads(line)["record"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    attempts = [r["payload"] for r in records if r["kind"] == "optimizer_attempt" and
                r["payload"]["iteration"] == 0]
    assert len(attempts) == 32
    for row in result["rows"]:
        a = [x for x in attempts if x["candidate_index"] == row["candidate_index"]]
        assert [x["action_sha256"] for x in a] == [row["composition_initial_sha256"],
                                                     row["spectral_only_initial_sha256"]]
    assert not result["equal_total_compute"] and not result["P2_admission"]


def test_reject_changed_fixed_before_work(tmp_path):
    fixed = dict(bridge.native.verified_input())
    fixed["goal_json"] = b"[]"
    events = journal(tmp_path / "events.jsonl")
    try:
        with pytest.raises(ValueError, match="bound bank"):
            bridge.run_bridge(fixed, transition, events, torch.device("cpu"))
    finally:
        events.close()


def test_failure_logged_and_never_optimized(tmp_path):
    fixed = bridge.native.verified_input()
    events = journal(tmp_path / "events.jsonl")
    def broken(*args):
        raise RuntimeError("fixture failure")
    try:
        with pytest.raises(RuntimeError, match="fixture failure"):
            bridge.run_bridge(fixed, broken, events, torch.device("cpu"))
    finally:
        events.close()
    audit = ledger_core.verify_ledger(tmp_path / "events.jsonl")
    assert audit["counts"]["composition_inherited", "conditional_prefix_transition"]["failed"] == 1
    records = (tmp_path / "events.jsonl").read_text()
    assert "optimizer_attempt" not in records


def test_jacobian_and_context_contract(tmp_path):
    fixed = bridge.native.verified_input()
    raw = np.frombuffer(fixed["bank"][0], dtype="<f4").reshape(10, 2).copy()
    events = journal(tmp_path / "events.jsonl")
    try:
        output = bridge.conditional_suffix(fixed, raw, transition, events, torch.device("cpu"), 0)
    finally:
        events.close()
    assert output.dtype == np.float32 and output.shape == (10, 2)
    assert np.array_equal(output[:5], raw[:5])
    assert np.max(np.abs(output)) <= 1
    assert np.linalg.norm(output[5:] - raw[5:]) <= bridge.DELTA_CAP + 1e-6


def test_prefix_changes_conditional_suffix(tmp_path):
    fixed = bridge.native.verified_input()
    raw = np.frombuffer(fixed["bank"][0], dtype="<f4").reshape(10, 2).copy()
    altered = raw.copy()
    altered[:5] = np.clip(altered[:5] + np.float32(0.25), -1, 1)
    events = journal(tmp_path / "events.jsonl")
    try:
        a = bridge.conditional_suffix(fixed, raw, transition, events, torch.device("cpu"), 0)
        b = bridge.conditional_suffix(fixed, altered, transition, events, torch.device("cpu"), 1)
    finally:
        events.close()
    assert not np.array_equal(a[5:], b[5:])


def test_common_gradient_matches_finite_difference(tmp_path):
    fixed = bridge.native.verified_input()
    events = journal(tmp_path / "events.jsonl")
    meter = bridge.native.CappedMeter(events)
    objective = bridge.native.make_objective(fixed, transition, meter, torch.device("cpu"))
    raw = np.frombuffer(fixed["bank"][0], dtype="<f4").reshape(10, 2).copy()
    raw[0, 0] = 0
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    arm = bridge.native.optimizer.ARMS[0]
    try:
        _, grad = objective(arm, context, goal, raw, True)
        xplus, xminus = raw.copy(), raw.copy()
        xplus[0, 0], xminus[0, 0] = 0.005, -0.005
        plus, _ = objective(arm, context, goal, xplus, False)
        minus, _ = objective(arm, context, goal, xminus, False)
    finally:
        events.close()
    assert np.isclose((plus - minus) / 0.01, grad[0, 0], rtol=0.02, atol=2e-5)
