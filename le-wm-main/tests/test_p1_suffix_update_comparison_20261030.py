from copy import deepcopy
import pytest
from experiments.spectral_reachability.p1_suffix_update_comparison_20261030 import compare


def panel():
    rows = []
    for arm in range(8):
        for i in range(20):
            rows.append(dict(slot=arm//2, arm=str(arm), start=i, iteration=0,
                stage_uphill=i == 1, best_iteration=0, best_regularized=float(i),
                uphill_and_better=i == 1, evaluations=[dict(raw=[[[0., 0.]]*5]*2)]))
    return dict(rows=rows)


def test_identity_and_ties():
    old = panel()
    new = deepcopy(old)
    new['rows'][1]['best_regularized'] = 0.
    result = compare(old, new)
    assert result['decisions'][0]['new_index'] == 0
    assert result['changed_candidate_actions'] == 0


def test_reject_misaligned_candidate():
    old = panel()
    new = deepcopy(old)
    new['rows'][1]['start'] = 7
    with pytest.raises(ValueError, match='identity'):
        compare(old, new)
