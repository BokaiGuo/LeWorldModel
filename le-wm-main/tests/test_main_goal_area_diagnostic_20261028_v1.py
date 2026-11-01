from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_goal_area_diagnostic_20261028_v1 as m


def rows():
    return [dict(word=w,uniform_actual_Q=i,area_actual_Q=-i,area_predicted_Q=-i,coverage=i)
            for i,w in enumerate(m.patch.WORDS)]


def test_all_pairs_no_selection_and_coverage_orientation():
    pairs=m.paired(rows())
    assert len(pairs)==6
    assert all(p['uniform_to_coverage'] is False and p['area_to_coverage'] is True and
               p['area_prediction_to_actual'] is True for p in pairs)


@pytest.mark.parametrize('fault',['missing','reordered'])
def test_fixed_word_denominator(fault):
    value=deepcopy(rows())
    if fault=='missing':value.pop()
    else:value.reverse()
    with pytest.raises(ValueError):m.paired(value)


def test_tie_not_agreement():
    assert m.agreement(0.,-1.) is None
