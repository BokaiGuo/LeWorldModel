import numpy as np
import pytest

from experiments.pc_rgb_bound_prefix_effect_20261107_codex import score_prefix_effects


def test_prefix_fixed_hold_relative_and_query_accounting():
    prefixes = np.stack([np.full((5, 2), .1), np.full((5, 2), -.1)])
    hold = np.zeros((5, 2))
    probes = np.stack([np.full((5, 2), .2), np.full((5, 2), -.3)])
    result = score_prefix_effects(lambda x: x.sum(axis=(1, 2)), prefixes, hold, probes, remaining_steps=10)
    np.testing.assert_allclose(result["effects"], [[-2, 3], [-2, 3]], atol=1e-6)
    np.testing.assert_allclose(result["chains"][:, 0, :5], prefixes)
    assert result["queries"] == 6
    assert result["model_packet_steps"] == 12


@pytest.mark.parametrize("hold,remaining", [(np.zeros((4, 2)), 9), (np.zeros((5, 2)), 15),
                                             (np.full((5, 2), 1.1), 10)])
def test_invalid_inputs_fail_before_query(hold, remaining):
    calls = []
    with pytest.raises(ValueError):
        score_prefix_effects(lambda x: calls.append(x), np.zeros((2, 5, 2)),
                             hold, np.zeros((3, *hold.shape)), remaining_steps=remaining)
    assert not calls


def test_invalid_scorer_output_rejected():
    with pytest.raises(ValueError):
        score_prefix_effects(lambda x: np.array([np.nan]), np.zeros((1, 5, 2)),
                             np.zeros((5, 2)), np.zeros((2, 5, 2)), remaining_steps=10)
