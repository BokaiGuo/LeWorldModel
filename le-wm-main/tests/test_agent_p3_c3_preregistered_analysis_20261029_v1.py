from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import agent_p3_c3_preregistered_analysis_20261029_v1 as m


def fixture_receipt():
    arms = []
    for identity in m.original.roster():
        caps = m.original.caps(identity['K'])
        zero = dict.fromkeys(m.AXES, 0)
        resources = dict(caps=caps, attempted=zero.copy(), completed=zero.copy(), FLOPs=None,
            backward_engine_calls=None, cuda_ns=None, events_count=0, events_semantic_sha256='a'*64,
            wall_ns=0, inclusive_operation_wall_ns={})
        arms.append(dict(**identity, status='no_witness', plan_available=False, R_groups=1,
            eligible_counts=[0], resources=resources, AD_primal_predict_API=0,
            Jacobian_elements=0, native_calls=0, scalar_backward_calls=None, total_FLOPs=None))
    return dict(schema=m.HOST+'.consume', status='saved12_common_cap_engineering_verified',
        denominator=12, total_request_denominator=1728, request_slots_per_arm=144,
        production_ready=False, full_C3_demonstrated=False, physical_release=False,
        same_actual_resources=False, same_FLOPs=False, arms=arms,
        prepare_sha256='1'*64, manifest_sha256='2'*64, report_sha256='3'*64, completion_sha256='4'*64)


def test_complete_fixed_panel_and_unknowns():
    result = m.analyze_receipt(fixture_receipt(), 'a'*64)
    assert len(result['rows']) == 12 and result['request_identity_denominator'] == 1728
    assert all(len(r['resources']['caps']) == 27 for r in result['rows'])
    assert result['interpretation']['unknown'] == ['total_FLOPs','scalar_backward_calls','cuda_ns']
    assert [r['mode'] for r in result['rows'][:3]] == ['fixedH1','staticH3_progress','adaptive']


@pytest.mark.parametrize('mutation', ['drop', 'reorder', 'cap', 'axis', 'spend', 'plan', 'flops', 'schema', 'flag', 'eligible'])
def test_rejects_mutations(mutation):
    r = fixture_receipt(); a = r['arms'][0]
    if mutation == 'drop': r['arms'].pop()
    elif mutation == 'reorder': r['arms'][0],r['arms'][1] = r['arms'][1],r['arms'][0]
    elif mutation == 'cap': a['resources']['caps']['predict_API'] += 1
    elif mutation == 'axis': a['resources']['completed'].pop('predict_API')
    elif mutation == 'spend': a['resources']['completed']['predict_API'] = 1
    elif mutation == 'plan': a['plan_available'] = True
    elif mutation == 'flops': a['resources']['FLOPs'] = 0
    elif mutation == 'schema': r['schema'] += '.ram'
    elif mutation == 'flag': r['same_FLOPs'] = True
    elif mutation == 'eligible': a['eligible_counts'] = [9]
    with pytest.raises(ValueError): m.analyze_receipt(r, 'a'*64)


def test_receipt_pin_and_duplicate_keys(tmp_path):
    p = tmp_path/'receipt.json'; raw = json.dumps(fixture_receipt()).encode(); p.write_bytes(raw)
    assert m.read_pinned(p, m.sha(raw)) == fixture_receipt()
    assert len(m.build(p, m.sha(raw))['rows']) == 12
    with pytest.raises(ValueError, match='SHA256'): m.read_pinned(p, '0'*64)
    raw = b'{"x":1,"x":2}'; p.write_bytes(raw)
    with pytest.raises(ValueError, match='duplicate'): m.read_pinned(p, m.sha(raw))


def test_results_are_not_selected_or_aggregated():
    r = fixture_receipt(); r['arms'][3]['status'] = 'tree_complete_not_admitted'; r['arms'][3]['plan_available'] = True
    out = m.analyze_receipt(deepcopy(r), 'a'*64)
    assert [row['slot'] for row in out['rows']] == list(range(12))
    assert sum(row['plan_available'] for row in out['rows']) == 1
    assert not out['interpretation']['cross_K_aggregation']
