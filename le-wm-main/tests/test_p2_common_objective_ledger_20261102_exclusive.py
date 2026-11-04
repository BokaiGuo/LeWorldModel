import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as core
from experiments.spectral_reachability import p2_sra_common_optimizer_bridge_20261102_exclusive as bridge


def fixture_transition(state, history, packet):
    # Analytic fixture only: no neural model, no physical claim.
    delta = packet.sum().expand(192) / 100
    successor = state[:, -1] + delta
    return torch.cat((state[:, 1:], successor[:, None]), dim=1), history


def test_same_objective_gradient_and_ledger(tmp_path):
    ledger = core.Ledger(tmp_path / "journal.jsonl")
    context = {"latent_history": np.zeros((3, 192)), "action_history": np.zeros((2, 10))}
    evaluate = core.objective(fixture_transition, context, np.ones(192), ledger, "cpu")
    initial = {arm: np.full((10, 2), 0.25, dtype=np.float32) for arm in bridge.ARMS}
    a = evaluate(bridge.ARMS[0], initial[bridge.ARMS[0]], True)
    b = evaluate(bridge.ARMS[1], initial[bridge.ARMS[1]], True)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_allclose(a[1], -0.019, rtol=1e-5)
    result = bridge.optimize_pair(initial, evaluate, lambda e: ledger.record("optimizer_" + e["kind"], e))
    ledger.close()
    audit = core.verify_ledger(tmp_path / "journal.jsonl")
    assert not audit["outstanding"]
    for arm in bridge.ARMS:
        category = "common_" + arm
        assert audit["counts"][(category, "transition_packet")]["returned"] == 20
        assert audit["counts"][(category, "autograd_grad")]["returned"] == 9
    assert result["rows"][0]["model_score"] == result["rows"][1]["model_score"]
    assert not result["equal_total_compute"] and not result["P2_admission"]


def test_failed_attempt_persists_and_tamper_fails(tmp_path):
    path = tmp_path / "journal.jsonl"
    ledger = core.Ledger(path)
    with ledger.phase("sra_initialization"), pytest.raises(ZeroDivisionError):
        ledger.call("jacobian", lambda: 1 / 0)
    ledger.close()
    summary = core.verify_ledger(path)
    assert summary["counts"][("sra_initialization", "jacobian")] == {
        "attempt": 1, "returned": 0, "failed": 1}
    lines = path.read_text().splitlines()
    event = json.loads(lines[1])
    event["record"]["payload"]["axis"] = "changed"
    lines[1] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="chain"):
        core.verify_ledger(path)


def test_exclusive_and_no_uncategorized_operations(tmp_path):
    path = tmp_path / "journal.jsonl"
    ledger = core.Ledger(path)
    with pytest.raises(ValueError, match="category"):
        ledger.call("predict", lambda: None)
    with pytest.raises(FileExistsError):
        core.Ledger(path)
    ledger.close()


def test_real_b0_source_authenticated_without_model():
    data = core.verified_b0()
    assert data["composition_raw"].shape == (10, 2)
    assert data["verification"]["independent_confirmation"] is False
    assert data["verification"]["P2_admission"] is False
