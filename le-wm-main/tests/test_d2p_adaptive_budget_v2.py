"""Metadata-only contract tests: not neural inference or physical efficacy."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability.d2p_adaptive_budget_v2 import (
    Binding, FrozenSchedule, FrozenSignals, PredictionBudget, PredictionBudgetExceeded,
)


def inputs(batch, device="cuda"):
    return SimpleNamespace(shape=(batch, 3, 2), device=SimpleNamespace(type=device, index=0))


class Model:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def predict(self, embedding, action):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model failure")
        return embedding


def guard(model, limit=12):
    return PredictionBudget(model, binding=Binding("protocol", "task", "root", "model"),
                            method="adaptive", sample_limit=limit)


def phase(budget):
    return budget.phase("planning", method="adaptive", horizon_tokens=2)


def test_expansion_and_each_token_count_actual_batch_not_base_candidates():
    schedule = FrozenSchedule("seal", (1, 2), 2, 3, .5, .5)
    decision = schedule.choose(FrozenSignals("seal", .5, .5))
    assert decision == {"horizon_tokens": 2, "expanded_candidates": 6,
                        "single_pass_predict_samples": 12}
    model = Model()
    with guard(model) as budget:
        with phase(budget):
            for _ in range(2):
                model.predict(inputs(6), inputs(6))
            with pytest.raises(PredictionBudgetExceeded):
                model.predict(inputs(1), inputs(1))
    receipt = budget.receipt()
    assert model.calls == 2
    assert receipt["attempted_predict_samples"] == 12
    assert receipt["blocked_calls"] == 1
    assert not receipt["p3_admitted"] and not receipt["compute_advantage"]


def test_failures_retries_and_new_phase_do_not_refund():
    model = Model()
    with guard(model, 5) as budget:
        model.fail = True
        with phase(budget):
            with pytest.raises(RuntimeError, match="model failure"):
                model.predict(inputs(3), inputs(3))
        model.fail = False
        with budget.phase("evaluation", method="adaptive", horizon_tokens=1):
            model.predict(inputs(2), inputs(2))
            with pytest.raises(PredictionBudgetExceeded):
                model.predict(inputs(1), inputs(1))
    assert budget.receipt()["attempted_predict_samples"] == 5
    assert budget.records[0]["failed_predict_calls"] == 1
    assert not budget.records[0]["completed"]
    assert sum(r["predict_samples"] for r in budget.records) == 5


def test_unscoped_invalid_and_oversized_calls_never_delegate():
    model = Model()
    with guard(model, 2) as budget:
        with pytest.raises(RuntimeError):
            model.predict(inputs(1), inputs(1))
        with phase(budget):
            with pytest.raises(ValueError):
                model.predict(inputs(1, "cpu"), inputs(1, "cpu"))
            with pytest.raises(PredictionBudgetExceeded):
                model.predict(inputs(3), inputs(3))
            assert budget.receipt()["attempted_predict_samples"] == 0
            model.predict(embedding=inputs(2), action=inputs(2))
    assert model.calls == 1


def test_restoration_detached_receipts_single_use_and_method_binding():
    model = Model()
    original = model.predict
    budget = guard(model)
    with pytest.raises(KeyError):
        with budget:
            with pytest.raises(ValueError):
                budget.phase("planning", method="other", horizon_tokens=1)
            with phase(budget):
                model.predict(inputs(1), inputs(1))
                raise KeyError("abort")
    assert model.predict == original and "predict" not in vars(model)
    receipt = budget.receipt()
    receipt["phases"][0]["predict_samples"] = 100
    assert budget.records[0]["predict_samples"] == 1
    with pytest.raises(RuntimeError):
        with budget:
            pytest.fail("single-use guard reused")


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_limits(value):
    with pytest.raises(ValueError):
        guard(Model(), value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1, True])
def test_invalid_signals(value):
    with pytest.raises(ValueError):
        FrozenSignals("seal", value, .2)


def test_frozen_deterministic_no_outcome_interface():
    schedule = FrozenSchedule("seal", (2, 4), 3, 2, .5, .3)
    signal = FrozenSignals("seal", .1, .1)
    assert schedule.choose(signal) == schedule.choose(signal)
    assert schedule.choose(signal)["expanded_candidates"] == 3
    with pytest.raises(ValueError):
        schedule.choose(FrozenSignals("other", .1, .1))
    with pytest.raises(TypeError):
        FrozenSignals("seal", .1, .1, success=True)
    with pytest.raises(AttributeError):
        signal.horizon_gain = .9
