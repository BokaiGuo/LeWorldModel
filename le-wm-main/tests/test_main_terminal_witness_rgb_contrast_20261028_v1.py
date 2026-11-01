import numpy as np
import pytest
from experiments.spectral_reachability import main_terminal_witness_rgb_contrast_20261028_v1 as m


@pytest.mark.parametrize('seed', range(8))
def test_signed_pair_identity(seed):
    rng = np.random.default_rng(seed)
    out = m.decompose(*rng.normal(size=(5, 192)))
    assert sum(out['signed_error_terms'].values()) == pytest.approx(out['gap'], abs=1e-12)


def test_common_offset_can_change_ranking_gap_without_response_error():
    z = np.zeros(192)
    out = m.decompose(z+2, z+3, z, z+1, z)
    assert out['gap'] == 4
    assert out['response_error_MSE'] == 0
    assert out['exact_action_independent_offset']


def test_same_action_control_is_exact_tie():
    rng = np.random.default_rng(3)
    p, a, g = rng.normal(size=(3, 192))
    out = m.decompose(p, p, a, a, g)
    assert out['predicted_delta'] == out['actual_delta'] == out['gap'] == 0


def test_ranking_reversal_retains_all_signed_terms():
    z = np.zeros(192)
    out = m.decompose(z+2, z+1, z+1, z+2, z)
    assert out['ranking_reversed'] and out['opposing_signed_terms']
    assert out['predicted_delta'] == -3 and out['actual_delta'] == 3


def test_nonfinite_rejected():
    z = np.zeros(192)
    with pytest.raises(ValueError):
        m.decompose(z, z, z, z, z+np.nan)
