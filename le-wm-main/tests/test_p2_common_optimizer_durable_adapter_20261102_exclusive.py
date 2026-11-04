import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p2_common_optimizer_durable_adapter_20261102_exclusive as adapter
from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as core
from experiments.spectral_reachability import p2_sra_common_optimizer_bridge_20261102_exclusive as optimizer


def initial():
    return {arm: np.full((10, 2), 0.25, dtype=np.float32) for arm in optimizer.ARMS}


def callback(arm, action, gradient):
    return float(np.sum(action * action)), 2 * action if gradient else None


def factory(ledger):
    return callback


def test_complete_same_callbacks_and_durable_events(tmp_path):
    out = tmp_path / "fixture"
    report = adapter.execute_fixture(initial(), factory, out)
    assert report["status"] == "fixture_completed"
    assert report["callback_counts"]["composition"] == report["callback_counts"]["no_composition_sra"]
    assert report["callback_counts"]["composition"] == {
        "attempted_scores": 9, "attempted_gradients": 8, "returned_scores": 9, "returned_gradients": 8}
    assert not report["production_execution_authorized"] and not report["equal_total_compute"]
    assert core.verify_ledger(out / "journal.jsonl")["tail_sha256"] == report["ledger_tail_sha256"]
    events = [json.loads(line)["record"] for line in (out / "journal.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "optimizer_attempt" for e in events) == 18
    assert sum(e["kind"] == "optimizer_returned" for e in events) == 18
    with pytest.raises(FileExistsError):
        adapter.execute_fixture(initial(), factory, out)


def test_failure_retains_first_attempt_no_second_arm(tmp_path):
    def failing(arm, action, gradient):
        raise RuntimeError("fixture failed")
    out = tmp_path / "failed"
    report = adapter.execute_fixture(initial(), lambda ledger: failing, out)
    assert report["status"] == "failed"
    assert report["callback_counts"]["composition"]["attempted_scores"] == 1
    assert report["callback_counts"]["composition"]["returned_scores"] == 0
    assert report["callback_counts"]["no_composition_sra"]["attempted_scores"] == 0
    events = [json.loads(line)["record"]["kind"] for line in (out / "journal.jsonl").read_text().splitlines()]
    assert events[-2:] == ["optimizer_attempt", "optimizer_failed"]
    core.verify_ledger(out / "journal.jsonl")


def test_invalid_return_not_counted_as_return(tmp_path):
    report = adapter.execute_fixture(initial(), lambda ledger: lambda arm, action, gradient: (float("nan"), None),
                                     tmp_path / "invalid")
    assert report["status"] == "failed"
    assert report["callback_counts"]["composition"]["returned_scores"] == 0


def test_bad_input_does_not_create_output(tmp_path):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        adapter.execute_fixture({}, factory, out)
    assert not out.exists()


def test_real_objective_fixture_shares_same_durable_ledger(tmp_path):
    def transition(state, history, packet):
        next_state = state[:, -1] + packet.sum().expand(192) / 100
        return torch.cat((state[:, 1:], next_state[:, None]), dim=1), history

    def make(ledger):
        return core.objective(transition,
                              {"latent_history": np.zeros((3, 192)),
                               "action_history": np.zeros((2, 10))},
                              np.ones(192), ledger, "cpu")

    out = tmp_path / "objective_fixture"
    report = adapter.execute_fixture(initial(), make, out)
    assert report["status"] == "fixture_completed"
    counts = core.verify_ledger(out / "journal.jsonl")["counts"]
    for arm in optimizer.ARMS:
        assert counts[("common_" + arm, "transition_packet")]["returned"] == 18
        assert counts[("common_" + arm, "autograd_grad")]["returned"] == 8
