import numpy as np
import pytest

from experiments.spectral_reachability.pc_suffix_population_optimizer import Config, optimize


def test_exact_budget_reproducibility_and_incumbent():
    prefix = np.array([[0.2, -0.3]], np.float32)
    suffix = np.zeros((3, 2), np.float32)
    calls = []

    def predict(actions):
        calls.append(actions.copy())
        return np.zeros((len(actions), 1)), np.square(actions[:, 1:] - 0.4).sum(axis=(1, 2))

    config = Config(samples=29, batch_size=12, elites=4)
    initial = float(np.square(suffix - 0.4).sum())
    first = optimize(predict, prefix, suffix, initial, config, seed=17)
    assert sum(len(batch) for batch in calls) == 29
    assert [batch["size"] for batch in first["generations"]] == [12, 12, 5]
    assert [batch["global_count"] for batch in first["generations"]] == [3, 3, 1]
    assert all(np.all(batch[:, 0] == prefix[0]) and np.max(np.abs(batch)) <= 1 for batch in calls)
    assert first["score"] <= initial
    second = optimize(predict, prefix, suffix, initial, config, seed=17)
    assert first == second


def test_zero_prefix_and_single_sample_remainder():
    def predict(actions):
        return actions.sum(axis=1), np.square(actions).sum(axis=(1, 2))

    out = optimize(predict, np.empty((0, 3)), np.full((2, 3), 0.2), .24,
                   Config(samples=13), seed=4)
    assert [x["size"] for x in out["generations"]] == [12, 1]
    assert out["generations"][-1]["global_count"] == 0


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan")])
def test_invalid_policy_rejected(fraction):
    with pytest.raises(ValueError):
        optimize(None, np.zeros((1, 2)), np.zeros((1, 2)), 0.,
                 Config(samples=12, global_fraction=fraction), seed=0)


def test_rejects_invalid_predictor_scores():
    with pytest.raises(ValueError, match="predictor"):
        optimize(lambda x: (None, np.full(len(x), np.nan)),
                 np.zeros((1, 2)), np.zeros((1, 2)), 0., Config(samples=12), seed=0)
