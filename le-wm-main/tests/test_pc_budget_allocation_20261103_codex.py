import numpy as np
import pytest

from experiments.pc_budget_allocation_20261103_codex import allocate_samples, optimize


def test_equal_allocation_consumes_complete_samples():
    allocation = allocate_samples(900, 18, 14, 3)
    assert sum(allocation) == 294
    assert max(allocation) - min(allocation) <= 1
    assert allocate_samples(1800, 318, 14, 3) == [36] * 4 + [35] * 10


@pytest.mark.parametrize("count", [1, 2, 3, 11, 12, 13, 25])
def test_partial_batches_reproducible_and_bounded(count):
    batches = []

    def predict(actions):
        batches.append(len(actions))
        return np.square(actions).sum(axis=(1, 2))

    first = optimize([[0.5, 0.0]] * 5, 15, count, np.random.default_rng(3), predict)
    second = optimize([[0.5, 0.0]] * 5, 15, count, np.random.default_rng(3),
                      lambda a: np.square(a).sum(axis=(1, 2)))
    assert sum(batches) == count
    assert all(n <= 12 for n in batches)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert np.max(np.abs(first[0])) <= 1
    np.testing.assert_array_equal(first[0][:5], [[0.5, 0.0]] * 5)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        allocate_samples(10, 11, 1, 3)
    with pytest.raises(ValueError):
        optimize([], 15, 0, np.random.default_rng(0), lambda x: np.zeros(len(x)))
    with pytest.raises(ValueError):
        optimize([], 15, 1, np.random.default_rng(0), lambda x: np.array([np.nan]))
