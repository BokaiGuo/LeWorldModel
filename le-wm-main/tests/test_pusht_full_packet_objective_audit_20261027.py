import pytest
from experiments.spectral_reachability.pusht_full_packet_objective_audit_20261027 import pair_counts, audit
from tests.test_analyze_pusht_full_packet_20261027 import actual


def test_pair_directions_and_censoring():
    assert pair_counts([dict(x=0,y=2),dict(x=1,y=1),dict(x=2,y=None)], 'x','y') == dict(agree=0,reverse=1,tied=0,unavailable=2)


def test_exact_ties_not_reversals():
    assert pair_counts([dict(x=1,y=2),dict(x=1,y=1)], 'x','y')['tied'] == 1


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        pair_counts([dict(x=0,y=2),dict(x=float('nan'),y=1)], 'x','y')


def test_actual_saved_census():
    result = audit(*actual())
    assert len(result['groups']) == 8
    assert sum(len(g['candidates']) for g in result['groups']) == 80
    assert len(result['excluded_fixed_terminal_slots']) == 3
    assert result['original_physical_status'] == 'integrity_fail'
    assert result['original_physical_gates']['encodings_complete'] is False
    assert result['selector_changed'] is False and result['P1_admission'] is False
    for g in result['groups']:
        assert all(sum(c.values()) == 45 for c in g['comparisons'].values())
