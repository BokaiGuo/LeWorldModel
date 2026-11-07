import numpy as np
import pytest

from experiments.pc_rgb_conditional_temporal_cem_20261107_codex import (
    K, N, ROUNDS, choose, optimize, sample_tails,
)


def test_common_first_population_and_fixed_prefix():
    prefix = np.zeros((10, 2), dtype=np.float32)
    tail = np.zeros((3, 2), dtype=np.float32)

    def scorer(actions):
        values = np.zeros((N, 5))
        values[:, -1] = actions[:, -1, 0]
        values[:, 2] = -actions[:, -1, 0]
        return values

    first, fs, records = optimize(prefix, tail, scorer, 20, "endpoint")
    second, ss, _ = optimize(prefix, tail, scorer, 20, "temporal")
    np.testing.assert_array_equal(first[:N], second[:N])
    assert not np.array_equal(first[N:2*N], second[N:2*N])
    assert first.shape == (N * ROUNDS, 25, 2)
    assert len(records[0]["elite"]) == K
    assert np.all(first[:, :10] == prefix)
    assert np.all(np.abs(first) <= 1)
    assert fs.shape == ss.shape == (N * ROUNDS, 5)


def test_final_choice_keeps_incumbent():
    incumbents = np.zeros((2, 25, 2), dtype=np.float32)
    candidates = np.zeros((N * ROUNDS, 25, 2), dtype=np.float32)
    inc_cost = np.full((2, 5), 10.0)
    inc_cost[1, -1] = 1.0
    tail_cost = np.full((N * ROUNDS, 5), 3.0)
    _, index, _ = choose(incumbents, inc_cost, candidates, tail_cost)
    assert index == 1


def test_zero_variance_rejected():
    with pytest.raises(ValueError):
        sample_tails(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((N, 3, 2)))
