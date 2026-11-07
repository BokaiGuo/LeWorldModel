import numpy as np

from experiments import pc_exact_temporal_tail_dev_20261107_codex as trial


def test_temporal_offsets_preserve_start_prefix_bounds_and_accounting(monkeypatch):
    monkeypatch.setattr(trial.prior.exact, "endpoints",
                        lambda task, chains: chains.sum(axis=1))
    prefix = np.full((5, 2), .1)
    tail = np.linspace(-.3, .3, 20).reshape(10, 2)
    result = trial.optimize({"goal": [0, 0]}, prefix, tail, seed=4, samples=24)
    chains = np.asarray(result["actions"])
    assert chains.shape == (24, 15, 2)
    np.testing.assert_array_equal(chains[0], np.concatenate((prefix, tail)))
    np.testing.assert_allclose(chains[:, :5], np.broadcast_to(prefix, (24, 5, 2)))
    diff = chains[:, 5:] - tail
    unprojected = (np.abs(chains[:, 5::2]) < 1) & (np.abs(chains[:, 6::2]) < 1)
    np.testing.assert_allclose(diff[:, ::2][unprojected], diff[:, 1::2][unprojected], atol=1e-15)
    assert np.abs(chains).max() <= 1
    assert result["best"] == int(np.argmin(result["costs"]))
