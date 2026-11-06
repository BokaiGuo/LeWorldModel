import numpy as np
import pytest

from experiments.spectral_reachability.pc_bounded_fd_suffix_optimizer import Config, optimize


def test_projected_fd_respects_budget_and_retains_best_probe():
    prefix = np.array([[.2, -.3]], np.float32)
    initial = np.array([[1., -1.], [0., 0.]], np.float32)
    calls = []

    def predict(actions):
        calls.append(actions.copy())
        return actions.sum(axis=1), np.square(actions[:, 1:] - .3).sum(axis=(1, 2))

    initial_score = float(np.square(initial - .3).sum())
    result = optimize(predict, prefix, initial, initial_score, Config(samples=25))
    assert result["samples"] == sum(len(x) for x in calls)
    assert result["samples"] <= 25
    assert result["samples"] + result["unused_samples"] == 25
    assert result["score"] < initial_score
    assert all(np.all(x[:, :1] == prefix) and np.max(np.abs(x)) <= 1 for x in calls)


def test_budget_too_small_is_noop():
    def forbidden(_):
        raise AssertionError("no evaluations fit")

    result = optimize(forbidden, np.zeros((0, 2)), np.zeros((3, 2)), 1., Config(samples=11))
    assert result["samples"] == 0
    assert result["unused_samples"] == 11
    assert result["score"] == 1.


def test_flat_score_stops_without_filling_budget():
    result = optimize(lambda x: (None, np.ones(len(x))), np.zeros((1, 2)), np.zeros((2, 2)),
                      1., Config(samples=100))
    assert result["samples"] == 8
    assert result["unused_samples"] == 92


@pytest.mark.parametrize("step", [0., -1., float("nan")])
def test_invalid_step_rejected(step):
    with pytest.raises(ValueError):
        optimize(None, np.zeros((1, 2)), np.zeros((1, 2)), 0., Config(samples=10, fd_step=step))
