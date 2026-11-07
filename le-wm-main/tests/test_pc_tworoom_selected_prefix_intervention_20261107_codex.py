import numpy as np
import pytest

from experiments.pc_tworoom_selected_prefix_intervention_20261107_codex import interventions


def test_common_tails_and_controls():
    prefix = np.arange(20, dtype=np.float32).reshape(10, 2) / 20
    tails = np.full((4, 15, 2), 0.25, np.float32)
    result = interventions(prefix, tails)
    for v in result.values():
        np.testing.assert_array_equal(v[:, 10:], tails)
    np.testing.assert_array_equal(result["selected"][0, :10], prefix)
    np.testing.assert_array_equal(result["reversed"][0, :10], prefix[::-1])
    assert not result["zero"][:, :10].any()


def test_rejects_bad_or_nonfinite_actions():
    with pytest.raises(ValueError):
        interventions(np.zeros((9, 2)), np.zeros((4, 15, 2)))
    with pytest.raises(ValueError):
        interventions(np.full((10, 2), np.nan), np.zeros((4, 15, 2)))
