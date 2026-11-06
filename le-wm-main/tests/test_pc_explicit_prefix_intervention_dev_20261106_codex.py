import numpy as np
import pytest

from experiments.pc_explicit_prefix_intervention_dev_20261106_codex import intervention_commands


def test_all_interventions_preserve_common_tail_and_bounds():
    plan = np.full((25, 2), 0.25, dtype=np.float32)
    arms = intervention_commands(plan, 7)
    for arm in arms.values():
        assert np.array_equal(arm[10:], plan[10:])
        assert np.abs(arm).max() <= 1
    assert np.all(arms["zero_prefix"][:10] == 0)
    assert np.array_equal(arms["random_prefix"], intervention_commands(plan, 7)["random_prefix"])
    assert not np.array_equal(arms["random_prefix"], intervention_commands(plan, 8)["random_prefix"])


def test_reject_out_of_bounds_and_wrong_shape():
    with pytest.raises(ValueError):
        intervention_commands(np.full((25, 2), 1.1), 0)
    with pytest.raises(ValueError):
        intervention_commands(np.zeros((24, 2)), 0)
