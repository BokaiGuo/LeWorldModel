"""CPU logic/algebra only. No neural model, checkpoint load, or GPU access."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027 as runner


@pytest.mark.parametrize("method", runner.METHODS)
def test_fixed_work_determinism_zero_and_best_ever(method):
    banks = []
    def cost(bank):
        banks.append(bank.copy())
        return np.sum(bank**2, axis=(1, 2))
    best, receipt = runner.optimize(method, cost, runner.SEEDS[0])
    np.testing.assert_array_equal(best, np.zeros((2, 2)))
    assert receipt["cost_evaluations"] == 257
    assert receipt["completed"] and not receipt["official_reproduction"]
    assert sum(len(b) for b in banks) == 257
    assert all(np.abs(b).max() <= 3 for b in banks)
    second = []
    runner.optimize(method, lambda b: second.append(b.copy()) or np.sum(b*b, axis=(1, 2)), runner.SEEDS[0])
    for a, b in zip(banks, second, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", runner.METHODS)
def test_failure_is_not_a_deployable_result(method):
    calls = 0
    def cost(bank):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected scorer failure")
        return np.ones(len(bank))
    with pytest.raises(runner.kernels.CostEvaluationError) as failure:
        runner.optimize(method, cost, runner.SEEDS[0])
    r = failure.value.receipt
    assert r["completed"] is False and r["cost_calls_failed"] == 1
    assert r["cost_evaluations"] == 1
    assert r["cost_evaluations_failed"] == (64 if method == "cem" else 256)


def test_between_two_recurrence_does_not_bootstrap_or_shift_again():
    # Arithmetic fake: observable histories rather than a CPU neural experiment.
    class Model:
        def __init__(self):
            self.actions, self.states = [], []
        def action_encoder(self, actions):
            self.actions.append(actions.copy())
            return actions
        def predict(self, emb, actions):
            self.states.append(emb.copy())
            return emb + actions
    m = Model()
    latent = np.asarray([[[1.], [2.], [3.]]])
    between = np.asarray([[[10.], [20.]]])
    tokens = np.asarray([[[30.], [40.]]])
    actual = runner.recurrence(m, latent, between, tokens, cat=np.concatenate)
    np.testing.assert_array_equal(m.actions[0], [[[10.], [20.], [30.]]])
    np.testing.assert_array_equal(m.actions[1], [[[20.], [30.], [40.]]])
    np.testing.assert_array_equal(m.states[1], [[[2.], [3.], [33.]]])
    np.testing.assert_array_equal(actual, [[73.]])
    assert len(m.states) == 2
    with pytest.raises(ValueError):
        runner.recurrence(m, latent, np.zeros((1, 3, 1)), tokens, cat=np.concatenate)


def test_saved_four_roots_goal_metric_and_context():
    path = Path(runner.DEFAULT_COMPILATION)
    if not path.exists():
        pytest.skip("local calibration artifact absent")
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    source = json.loads(path.read_text())
    assert len(source["rows"]) == 4
    for row in source["rows"]:
        c = row["compiled"]
        context = runner.root_context(c)
        assert np.shape(context["latent_history"]) == (3, 192)
        np.testing.assert_array_equal(context["action_history"], np.asarray(c["input_context"]["actions"])[0, 1:])
        for r in c["receipts"]:
            actual = metric_for(c)({"latent_history": [r["predicted_terminal_vector"]]})
            assert actual == pytest.approx(r["nonlinear_terminal_cost_diagnostic_only"], abs=1e-10)


@pytest.mark.parametrize("value", [np.zeros((3, 2)), np.full((2, 2), np.nan), np.full((2, 2), 3.01), np.zeros((2, 2), complex)])
def test_illegal_parameters_rejected(value):
    with pytest.raises(ValueError):
        runner.checked_params(value)


def test_cpu_neural_bridge_rejected_before_model_access():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="host CUDA only"):
        runner.CandidateCost(None, SimpleNamespace(type="cpu"), None, None, None, None)


def test_writer_never_overwrites(tmp_path):
    path = tmp_path/"receipt.json"
    runner.write(path, {"value": 1})
    with pytest.raises(FileExistsError):
        runner.write(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
