import numpy as np
import pytest

from experiments.pc_executable_effect_bank_20261107_codex import effect_bank


def test_effects_are_same_time_reference_minus_probe_and_accounted():
    prefix = np.array([[[0.1], [0.2]], [[0.3], [0.4]]])
    probes = np.array([[[0.2]], [[-0.5]]])
    reference = np.array([[0.0]])
    seen = []

    def score(chains):
        seen.append(chains.copy())
        return (chains ** 2).sum(axis=(1, 2))

    result = effect_bank(prefix, probes, reference, score)
    assert result["num_full_chain_scores"] == 6
    assert seen[0].shape == (6, 3, 1)
    np.testing.assert_allclose(result["effect"], [[-0.04, -0.25]] * 2)
    np.testing.assert_allclose(result["opportunity"],
                               result["reference_cost"] - result["effect"].max(axis=1))


def test_illegal_and_mismatched_actions_fail_closed():
    prefix = np.zeros((2, 2, 1))
    probes = np.zeros((2, 1, 1))
    score = lambda x: np.zeros(len(x))
    with pytest.raises(ValueError):
        effect_bank(prefix, probes, np.zeros((2, 1)), score)
    with pytest.raises(ValueError):
        effect_bank(prefix, probes + 2, np.zeros((1, 1)), score)
    with pytest.raises(ValueError):
        effect_bank(prefix, probes, np.zeros((1, 1)), lambda x: np.array([np.nan]))
