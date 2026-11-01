import pytest
from experiments.spectral_reachability.p1_long_suffix_selector_audit_20261029 import analyze


def test_fixed_denominator():
    with pytest.raises(ValueError, match='fixed panel'):
        analyze({'rows': []}, {'rows': []})


def test_stable_first_argmin_and_full_action_dedup():
    rows, arms = [], []
    for arm in range(8):
        candidates = []
        for i in range(20):
            rows.append(dict(slot=arm//2, arm=str(arm), start=i, iteration=0,
                best_regularized=0 if i < 2 else 1, stage_uphill=i == 1,
                uphill_and_better=i == 1, best_iteration=0,
                evaluations=[dict(raw=[[0., 0.]]*10)]))
            candidates.append(dict(prefix=[[0., 0.]]*5, reach_better=False, own_reach_value=2.))
        arms.append(dict(result=dict(candidates=candidates, baseline_reach_value=1.)))
    result = analyze(dict(rows=rows), dict(rows=arms))
    assert result['argmin_uphill'] == 0
    assert result['witnessed_occurrences'] == 8
    assert result['distinct_slot_raw15'] == 4
    assert result['witnessed_original_reach_better'] == 0
    assert all(x['candidate_index'] == 0 for x in result['winners'])
