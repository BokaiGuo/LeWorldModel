from copy import deepcopy

import pytest

from experiments.spectral_reachability import main_p1_action_response_analysis_20261028_v1 as m


@pytest.fixture(scope='module')
def actual_result(): return m.build()


def test_all_fixed_pairs_and_known_root_boundary(actual_result):
    assert actual_result['fixed_pairs'] == sum(len(a['pairs']) for a in actual_result['anchors']) == 80
    assert actual_result['known_roots'] == 2 and actual_result['fresh_roots'] == 0
    assert actual_result['P1_admission'] is False and actual_result['new_neural_calls'] == 0


def test_exact_no_float_roundoff_identity():
    x = m.vec([.1]*192); y = m.vec([.2]*192)
    assert m.inner(x, x)+m.inner(y, y)-2*m.inner(x, y) == m.inner(m.minus(x,y),m.minus(x,y))


def test_consumer_derived_margin_mutation_rejected():
    import json
    r = json.loads(m.REPORT.read_bytes()); s = json.loads(m.RECEIPT.read_bytes())
    strict = deepcopy(s['v2_adapted_consume']['analyses'][0])
    strict['effects']['pairs'][0]['response_remainder']['numerator'] = '0'
    with pytest.raises(ValueError): m.analyze_anchor(r['anchors'][0],r['rows'][:5],strict)
