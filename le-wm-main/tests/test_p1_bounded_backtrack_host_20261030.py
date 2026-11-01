import numpy as np
import pytest

from experiments.spectral_reachability.p1_bounded_backtrack_host_20261030 import (
    STEPS, choose, trial_logits,
)
from experiments.spectral_reachability.p1_regularized_suffix_host_20261030 import proposal


def test_first_improvement_and_rejection():
    assert choose(1.0, [1.0, .9, .8, .7]) == 1
    assert choose(1.0, [1., 1.1, 2., 3.]) is None
    with pytest.raises(ValueError):
        choose(1., [1., 2., 3., float("nan")])


def test_unit_step_matches_existing_producer_and_rejects_invalid():
    x = np.arange(20, dtype=np.float64).reshape(2, 10) / 100
    grad = np.linspace(-.1, .1, 20).reshape(2, 10)
    raw = np.tanh(x.astype(np.float32))
    expected = np.stack([proposal(x[i], grad[i], raw[i]) for i in range(2)])
    assert np.array_equal(trial_logits(x, grad, raw, STEPS[0]), expected)
    with pytest.raises(ValueError):
        trial_logits(x, grad, raw, .3)
