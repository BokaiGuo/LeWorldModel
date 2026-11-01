import numpy as np
import pytest
from experiments.spectral_reachability import main_raw_hscale_20261028_v1 as m


def test_zero_not_counted_as_direction():
    x = m.vector_comparison([1., 0.], [0., 0.])
    assert x['cosine'] is None and x['small_exact_zero']
    assert m.scalar_relation(1., 0.) == 'small_zero'
    assert m.scalar_relation(0., 0.) == 'both_zero'


def test_vector_scale_and_opposite():
    x = m.vector_comparison([1., 0.], [-2., 0.])
    assert x['cosine'] == -1 and x['norm_ratio_small_over_large'] == 2
    assert x['relative_L2_to_large'] == 3
    assert m.scalar_relation(1., -2.) == 'opposite_sign'


def test_reject_nonfinite():
    with pytest.raises(ValueError):
        m.vector_comparison([1., 0.], [np.nan, 0.])


def test_saved_whole_panel_and_both_scales():
    result = m.build()
    assert len(result['rows']) == 40 and result['fixed_original_pairs'] == 80
    assert result['no_h_selected'] and not result['P1_admission']
    for row in result['rows']:
        assert [e['h'] for e in row['endpoints']] == [.01, .001]
    assert sum(result['summary']['actual_goal_signs'].values()) == 40
