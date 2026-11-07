import numpy as np
import pytest

from experiments.pc_tworoom_actual_prefix_action_effect_20261107_codex import probes
from experiments.pc_tworoom_actual_prefix_action_effect_analysis_20261107_codex import agreement, ranking_flip


def test_probe_bank_contains_actual_suffix_only_when_available():
    chain = [[0.2, -0.4]] * 10
    result = probes(chain)
    assert result.shape == (6, 5, 2)
    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_allclose(result[-1], chain[5:10])
    assert probes(chain[:9]).shape == (5, 5, 2)


def test_probe_bank_rejects_illegal_actual_suffix():
    with pytest.raises(ValueError):
        probes([[0., 0.]] * 5 + [[2., 0.]] * 5)


def test_agreement_excludes_ties_and_ranking_flip_is_strict():
    assert agreement([1, -1, 0], [3, 2, 1], 1e-6, 1e-5) == (1, 2, 1)
    assert ranking_flip([0, 1], [1, 0])
    assert not ranking_flip([0, 0], [1, 0])
