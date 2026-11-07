import numpy as np
import pytest
import torch

from experiments.pc_action_effect_descriptor import ActionWitness, describe, quadratic_prediction


def witness(name, x, length=5):
    return ActionWitness(name, tuple((x, 0.) for _ in range(length)))


def test_same_name_not_same_identity():
    assert witness("same", .1).fingerprint != witness("same", -.1).fingerprint


def test_hold_relative_sign():
    result = describe(lambda a: torch.tensor((a[0, 0] - .2)**2),
                      witness("hold", 0), [witness("positive", .1), witness("negative", -.1)])
    assert result["queries"] == 3
    np.testing.assert_allclose(result["effects"], [.03, -.05])


def test_duration_mismatch_rejected():
    with pytest.raises(ValueError):
        describe(lambda a: torch.tensor(0.), witness("hold", 0), [witness("short", .1, 4)])


def test_out_of_bounds_rejected():
    with pytest.raises(ValueError):
        witness("invalid", 1.1).array()


def test_quadratic_exact_on_known_function():
    x, eps = np.array([-.1, .03]), .003
    f = lambda v: 2 + 3*v + 4*v*v
    np.testing.assert_allclose(quadratic_prediction(f(0), f(-eps), f(eps), eps, x), f(x))
