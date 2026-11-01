import numpy as np
import pytest

from experiments.spectral_reachability import main_normalizer_contrast_20261028_v1 as m


def test_zero_and_opposite_not_confused():
    assert m.category(1, 0) == 'actual_tie'
    assert m.category(0, 0) == 'both_tie'
    assert m.category(0, -1) == 'predicted_tie'
    assert m.category(1, -1) == 'opposite'
    assert m.category(-1, -2) == 'same'


def test_fixed80_both_h_and_all_actual_ties():
    x = np.arange(164, dtype=float)
    result = m.pair_labels(x, np.zeros(164))
    assert result['counts'] == {'actual_tie': 80}
    assert [p['h'] for p in result['by_panel_h']] == [.01, .001]*4
    assert len(result['labels']) == 80


@pytest.mark.parametrize('bad', [np.zeros(163), np.full(164, np.nan)])
def test_missing_and_nonfinite_not_dropped(bad):
    with pytest.raises(ValueError):
        m.pair_labels(bad, np.zeros(164))


def test_completed_actual_contrast_preserves_denominator():
    result = m.build()
    assert result['public_all164_exact'] and not result['P1_admission']
    for arm in result['arms']:
        assert sum(arm['native_loss_vs_actual64']['counts'].values()) == 80
        assert sum(arm['independent_float64_vs_actual64']['counts'].values()) == 80
    assert sum(result['paired_transition'].values()) == 80
