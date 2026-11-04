from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p2_same_bank_common_optimizer_20261102_exclusive as m


def test_real_fixed_bank_same_optimization_cpu_fixture():
    fixed = m.load_real()
    calls, events = [], []
    def objective(arm, context, goal, x, gradient):
        assert m.sha(m.canonical(context)) == fixed["context_sha256"]
        assert m.sha(m.canonical(goal)) == fixed["goal_sha256"]
        calls.append((arm, x.copy(), gradient))
        return float(np.sum(x * x)), 2 * x if gradient else None
    result = m.run_fixture(fixed, objective, events.append)
    assert len(result["rows"]) == 16 and len(calls) == 288
    assert all(result["counts"][arm] == {"scores": 144, "gradients": 128} for arm in m.ARMS)
    assert sum(e["kind"] == "attempt" for e in events) == 288
    for i in range(16):
        a, b = calls[i * 18], calls[i * 18 + 9]
        assert np.array_equal(a[1], b[1])
        assert a[1].astype("<f4").tobytes() == fixed["bank"][i]
        left, right = result["rows"][i]["arms"]
        assert left["raw_float32_hex"] == right["raw_float32_hex"]
    assert not result["production_execution_authorized"] and not result["equal_total_compute"]


def test_callback_failure_does_not_run_other_arm():
    calls, events = [], []
    def fail(arm, context, goal, action, gradient):
        calls.append(arm)
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        m.run_fixture(m.load_real(), fail, events.append)
    assert calls == [m.ARMS[0]]
    assert [e["kind"] for e in events] == ["attempt", "failed"]


def test_arm_fixture_can_differ_without_changing_input_bank():
    def route(arm, context, goal, action, gradient):
        shift = np.float32(0.1 if arm == m.ARMS[0] else -0.1)
        diff = action - shift
        return float(np.sum(diff * diff)), 2 * diff if gradient else None
    events = []
    result = m.run_fixture(m.load_real(), route, events.append)
    for i in range(16):
        attempts = [event for event in events if event["candidate_index"] == i
                    and event["kind"] == "attempt" and event["iteration"] == 0]
        assert len(attempts) == 2 and attempts[0]["action_sha256"] == attempts[1]["action_sha256"]
    assert result["rows"][0]["arms"][0]["raw_float32_hex"] != result["rows"][0]["arms"][1]["raw_float32_hex"]
    assert not result["composition_mechanism_authenticated"]


def test_pinned_source_and_context_reject_mutation():
    handoff, pair = m.HANDOFF.read_bytes(), m.PAIR.read_bytes()
    with pytest.raises(ValueError, match="SHA"):
        m.load_fixed(handoff + b" ", pair)
    with pytest.raises(ValueError, match="SHA"):
        m.load_fixed(handoff, pair + b" ")
    fixed = deepcopy(m.load_real())
    fixed["context_json"] += b" "
    with pytest.raises(ValueError, match="drift"):
        m.run_fixture(fixed, lambda *a: None, lambda e: None)
    fixed = deepcopy(m.load_real())
    fixed["bank"] = (b"\0" * 80,) + fixed["bank"][1:]
    with pytest.raises(ValueError, match="drift"):
        m.run_fixture(fixed, lambda *a: None, lambda e: None)
