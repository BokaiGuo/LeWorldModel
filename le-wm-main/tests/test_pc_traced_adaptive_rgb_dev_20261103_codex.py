import json

import numpy as np
import pytest

from experiments import pc_traced_adaptive_rgb_dev_20261103_codex as experiment
from experiments.pc_budget_allocation_20261103_codex import optimize
from experiments.pc_equal_resource_dev_20261103_codex import Meter


def predictor(meter):
    def predict(actions):
        def raw(batch):
            positions = batch.sum(axis=1)
            return positions, np.linalg.norm(positions - np.array([.2, -.3]), axis=1)
        return meter.predict(np.asarray(actions), raw)
    return predict


@pytest.mark.parametrize("horizon", (10, 15))
def test_persistent_suffix_matches_existing_optimizer(horizon):
    prefix = np.full((5, 2), .1)
    def evaluate(actions):
        positions = actions.sum(axis=1)
        return positions, np.linalg.norm(positions, axis=1)
    state = experiment.SuffixState(prefix, horizon, 17)
    for count in (12, 12, 5):
        state.step(count, evaluate)
    chain, score, _ = optimize(prefix, horizon, 29, np.random.default_rng(17),
                               lambda x: evaluate(x)[1])
    np.testing.assert_array_equal(state.best, chain)
    assert state.score == score


@pytest.mark.parametrize("arm", experiment.ARMS)
@pytest.mark.parametrize("step", (0, 1, 2))
def test_trace_covers_every_actual_prediction_and_budget(arm, step):
    meter = Meter(480)
    horizon = 15 - 5 * step
    carry = None if step == 0 else np.zeros((horizon, 2))
    result = experiment.decide(arm, 1, step, np.array([.2, -.3]), predictor(meter), meter, carry)
    assert sum(x["tokens"] for x in result["prediction_trace"]) == meter.token_samples <= 480
    assert len(result["selected"]["chain"]) == horizon
    if arm == "adaptive" and step < 2:
        assert [x["candidate"] for x in result["allocation"][:3]] == [0, 1, 2]
        assert all(x["samples"] == 12 for x in result["allocation"][:3])


def test_saved_physical_panel_and_prior_controls():
    physical = experiment.PREFIX.with_name(experiment.PREFIX.name + "_physical.json")
    if not physical.exists():
        pytest.skip("real RGB development panel not yet audited")
    data = json.loads(physical.read_text())
    assert len(data["rows"]) == 16
    assert data["backend"] == "2.1.2.14"
    previous = experiment.policy.PREFIX
    for root in range(4):
        for arm, prior in (("uniform", "svd"), ("greedy", "greedy"), ("cem", "cem")):
            current_file = experiment.PREFIX.with_name(experiment.PREFIX.name + f"_root{root}_{arm}_execution.json")
            previous_file = previous.with_name(previous.name + f"_root{root}_{prior}_execution.json")
            current = json.loads(current_file.read_text())["executed_actions"]
            old = json.loads(previous_file.read_text())["executed_actions"]
            np.testing.assert_array_equal(current, old)
