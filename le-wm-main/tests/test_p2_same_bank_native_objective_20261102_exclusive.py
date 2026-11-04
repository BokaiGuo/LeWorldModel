import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as ledger
from experiments.spectral_reachability import p2_same_bank_native_objective_20261102_exclusive as bridge


def transition(state, history, packet):
    assert packet.shape == (5, 2)
    successor = state[:, -1] + packet.sum().expand(192) / 100
    return torch.cat((state[:, 1:], successor[:, None]), dim=1), history


def setup(tmp_path):
    fixed = bridge.verified_input()
    journal = ledger.Ledger(tmp_path / "journal.jsonl")
    meter = bridge.CappedMeter(journal)
    evaluate = bridge.make_objective(fixed, transition, meter, torch.device("cpu"))
    return fixed, journal, meter, evaluate


def test_same_goal_gradient_full_schedule(tmp_path):
    fixed, journal, meter, evaluate = setup(tmp_path)
    events = []
    try:
        result = bridge.run_schedule(fixed, evaluate, events.append)
    finally:
        journal.close()
    assert len(result["rows"]) == 16
    for arm in bridge.optimizer.ARMS:
        assert evaluate.attempted[arm] == {"score": 144, "gradient": 128}
        assert meter.counts[arm]["transition_packet"] == 288
        assert meter.counts[arm]["autograd_grad"] == 128
        assert meter.counts[arm]["predict"] == 0  # fixture is not a native model
    assert len(events) == 576
    for row in result["rows"]:
        assert row["arms"][0]["raw_float32_hex"] == row["arms"][1]["raw_float32_hex"]
    assert not ledger.verify_ledger(tmp_path / "journal.jsonl")["outstanding"]


def test_context_action_and_gradient_guards(tmp_path):
    fixed, journal, meter, evaluate = setup(tmp_path)
    arm = bridge.optimizer.ARMS[0]
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    raw = np.frombuffer(fixed["bank"][0], dtype="<f4").reshape(10, 2).copy()
    try:
        with pytest.raises(ValueError, match="context or goal"):
            evaluate(arm, context, [0] * 192, raw, True)
        with pytest.raises(ValueError, match="float32"):
            evaluate(arm, context, goal, raw.astype(np.float64), True)
        score, grad = evaluate(arm, context, goal, raw, True)
        assert np.isfinite(score) and grad.shape == (10, 2) and grad.dtype == np.float32
        assert np.isfinite(grad).all()
    finally:
        journal.close()


def test_budget_cap_before_callback(tmp_path):
    fixed, journal, meter, evaluate = setup(tmp_path)
    arm = bridge.optimizer.ARMS[0]
    evaluate.attempted[arm]["score"] = bridge.CALLBACK_CAPS["score"]
    try:
        with pytest.raises(ValueError, match="cap"):
            evaluate(arm, json.loads(fixed["context_json"]), json.loads(fixed["goal_json"]),
                     np.zeros((10, 2), np.float32), False)
        assert meter.counts[arm]["transition_packet"] == 0
    finally:
        journal.close()


def test_normalizer_binding():
    metadata = {"mean": [0.1, 0.2], "std": [0.3, 0.4]}
    bridge.check_normalizer(metadata, [0.1, 0.2], [0.3, 0.4])
    with pytest.raises(ValueError, match="normalizer"):
        bridge.check_normalizer(metadata, [0.1, 0.3], [0.3, 0.4])
    with pytest.raises(ValueError, match="normalizer"):
        bridge.check_normalizer({"mean": [0, 0], "std": [0, 1]}, [0, 0], [0, 1])
