import numpy as np
import pytest

from experiments.spectral_reachability import p2_sra_common_optimizer_bridge_20261102_exclusive as bridge


def inputs():
    return {arm: np.full((10, 2), value, dtype=np.float32)
            for arm, value in zip(bridge.ARMS, (0.6, -0.6))}


def test_fixed_common_optimizer_real_callback_contract():
    events = []
    def evaluate(arm, x, need_grad):
        return float(np.square(x).mean()), (2 * x / x.size if need_grad else None)
    result = bridge.optimize_pair(inputs(), evaluate, events.append)
    assert len(events) == 36
    assert [x["arm"] for x in result["rows"]] == list(bridge.ARMS)
    assert all(x["returned_score_calls"] == 9 and x["returned_gradients"] == 8
               and x["winner_iteration"] == 8 for x in result["rows"])
    assert not result["equal_total_compute"] and not result["P2_admission"]


def test_failure_is_charged_and_second_arm_not_run():
    events = []
    def evaluate(arm, x, need_grad):
        if sum(e["kind"] == "attempt" for e in events) == 3:
            raise RuntimeError("test failure")
        return float(np.square(x).mean()), np.zeros_like(x)
    with pytest.raises(RuntimeError):
        bridge.optimize_pair(inputs(), evaluate, events.append)
    assert [e["kind"] for e in events] == ["attempt", "returned"] * 2 + ["attempt", "failed"]
    assert all(e["arm"] == bridge.ARMS[0] for e in events)


def test_invalid_gradient_or_unbounded_initial_fails():
    values = inputs()
    values[bridge.ARMS[0]][0, 0] = 2
    with pytest.raises(ValueError, match="bounded"):
        bridge.optimize_pair(values, lambda *_: None, lambda _: None)
    with pytest.raises(ValueError, match="gradient"):
        bridge.optimize_pair(inputs(), lambda *args: (1.0, np.zeros((10, 2))), lambda _: None)


def test_sra_adapter_requires_strict_bank_and_actual_full_action(monkeypatch):
    bank = {"commands": [[[0.0, 0.0]] * 5, [[0.5, -0.5]] * 5]}
    search = {"attempts": [{"candidate_id": "winner", "actions": bank["commands"][0] + bank["commands"][1]}]}
    monkeypatch.setattr(bridge.sra, "audit_candidates", lambda s, b: ([], "winner"))
    result = bridge.from_sra_search(search, bank)
    assert result["raw_actions"].shape == (10, 2)
    assert result["raw_actions"][5].tolist() == [0.5, -0.5]
    search["attempts"][0]["actions"][0][0] = 2.0
    with pytest.raises(ValueError, match="bounded"):
        bridge.from_sra_search(search, bank)


def test_real_capture_contract_remains_blocked():
    value = bridge.contract()
    assert value["sra"]["predict_and_action_encoder_calls"] == 88
    assert value["common_optimizer"]["steps"] == 8
    assert value["total_budget"]["status"] == "blocked"
    assert value["production_execution_authorized"] is False
    assert value["capture_source"]["fresh_root"] is False
