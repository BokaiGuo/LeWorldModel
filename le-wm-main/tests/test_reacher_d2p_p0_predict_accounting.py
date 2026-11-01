"""Runner wiring with metadata fakes only, never CPU neural inference."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress import _instrumented_planner


class Model:
    def predict(self, emb, actions):
        return emb


def value(batch):
    return SimpleNamespace(shape=(batch, 3, 192), device=SimpleNamespace(type="cuda", index=0))


def test_runner_preserves_result_and_separates_calls_from_samples():
    model = Model()
    expected = object()
    def planner(model, *, future_horizon):
        for _ in range(future_horizon): model.predict(value(32), value(32))
        return expected, {"forward_rollout_evaluations": 1}
    result, receipt = _instrumented_planner("cem", planner, model, future_horizon=4)
    assert result is expected
    assert receipt["forward_rollout_evaluations"] == 1
    assert receipt["actual_planning_predict"]["predict_calls"] == 4
    assert receipt["actual_planning_predict"]["predict_samples"] == 128
    assert "predict" not in vars(model)


def test_runner_restores_model_on_planner_error():
    model = Model()
    def planner(model, *, future_horizon):
        model.predict(value(1), value(1))
        raise RuntimeError("fixture")
    with pytest.raises(RuntimeError, match="fixture"):
        _instrumented_planner("gn", planner, model, future_horizon=1)
    assert "predict" not in vars(model)
