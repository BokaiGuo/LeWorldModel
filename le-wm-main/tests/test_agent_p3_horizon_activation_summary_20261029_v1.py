"""RAM-only explicit independent-summary ABI fixtures, not run evidence."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_horizon_activation_summary_20261029_v1 as m


def fixture():
    rows = []
    for root in (0, 1):
        for arm in m.ARMS:
            nodes = []
            for b in (5, 10, 15):
                h = 3 if arm.startswith('adaptive') else 1
                panels = []
                for ph in (1, 3):
                    available = b+5*ph <= 20
                    panels.append(dict(horizon=ph, selected_for_execution=ph == h,
                        causal_endpoint_pair=ph == h == 1, predicted_endpoint_step=b+5*ph,
                        actual_Q_at_predicted_endpoint=0.0 if available else None,
                        error_decomposition=dict(model_Q=0.0, real_Q=0.0, cross=0.0, quadratic=0.0) if available else None,
                        missing_actual_reason=None if available else 'beyond_executed20_lookahead',
                        first5_raw=[[0.0, 0.0] for _ in range(5)]))
                nodes.append(dict(boundary=b,status='sealed',executed_horizon=h,
                    decision=dict(horizon=h,program_type='progress',extension_type='progress',continue_expanding=True),panels=panels))
            steps = [dict(step=i,state=[0.0]*7) for i in range(21)]
            rows.append(dict(source_slot=root,arm=arm,episode_id=f's{root}_{arm}',control_returned=20,
                steps=steps,missing_steps=[],nodes=nodes))
    pairs = []
    for root in (0,1):
        for l,r in (('fixedH1_endpoints','adaptive_endpoints'),('fixedH1_midpoints','adaptive_midpoints'),
                    ('fixedH1_endpoints','fixedH1_midpoints'),('adaptive_endpoints','adaptive_midpoints')):
            pairs.append(dict(source_slot=root,left=f's{root}_{l}',right=f's{root}_{r}',
                steps=[dict(step=i,state_bits_equal=True) for i in range(21)]))
    return dict(schema=m.UPSTREAM+'.summary',status='known2_descriptive_NOT_admitted',rows=rows,pairs=pairs,
        fixed_denominator=8,node_denominator=24,panel_denominator=48,pair_denominator=8,known_roots=2,fresh_roots=0,
        P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False,
        bindings={'/fixture/strict.json':'a'*64},authority=dict(external_actual_strict_sha256='a'*64))


def refresh_pairs(s):
    lookup={r['episode_id']:r for r in s['rows']}
    for p in s['pairs']:
        for i,v in enumerate(p['steps']):
            a,b=lookup[p['left']]['steps'][i],lookup[p['right']]['steps'][i]
            v['state_bits_equal']=None if a is None or b is None else np.asarray(a['state'],np.float64).tobytes()==np.asarray(b['state'],np.float64).tobytes()


def stop(s, row_index, returned):
    r=s['rows'][row_index];r['control_returned']=returned
    r['missing_steps']=list(range(returned+1,21))
    for i in r['missing_steps']:r['steps'][i]=None
    for n in r['nodes']:
        if n['boundary']>returned:
            n.update(status='not_reached',executed_horizon=None,decision=None,panels=[])
        else:
            for p in n['panels']:
                if p['predicted_endpoint_step']>returned:
                    p.update(actual_Q_at_predicted_endpoint=None,error_decomposition=None,
                        missing_actual_reason='beyond_executed20_lookahead' if p['predicted_endpoint_step']>20 else 'not_observed_or_not_encoded')
    refresh_pairs(s)


def test_full24_and_all_pairs():
    o=m.summarize(fixture())
    assert o['node_counts']==dict(sealed=24,refused=0,not_reached=0,chosen_H1=12,chosen_H3=12,dispatched_H1=12,dispatched_H3=12)
    assert len(o['pairs'])==8 and all(len(p['steps'])==21 for p in o['pairs'])
    assert o['error_categories']==dict(same_action_H1_available=12,same_action_H1_missing=0,descriptive_H3_available=8,descriptive_H3_missing=16,unexecuted_or_incomplete_H1_available=12,unexecuted_or_incomplete_H1_missing=0,uncomputed_H1_missing=0,uncomputed_H3_missing=0)
    assert o['pairs'][0]['raw_counts']==dict(missing=5,same=15,different=0)


def test_refusal_overlapping_reasons_and_denominator():
    s=fixture();stop(s,2,5);n=s['rows'][2]['nodes'][0]
    n.update(status='refused',executed_horizon=None)
    n['decision'].update(program_type='reconfigure',extension_type='reconfigure',continue_expanding=False)
    for p in n['panels']:p.update(selected_for_execution=False,causal_endpoint_pair=False)
    o=m.summarize(s)
    assert o['node_denominator']==24 and o['node_counts']['refused']==1 and o['node_counts']['not_reached']==2
    assert o['refusal_reasons_nonexclusive']==dict(program_type_not_progress=1,extension_type_not_progress=1,continue_expanding_not_true=1)
    assert o['rows'][2]['nodes'][0]['chosen_horizon'] is None
    assert sum(o['error_categories'].values())==48
    assert o['error_categories']['uncomputed_H1_missing']==o['error_categories']['uncomputed_H3_missing']==2


def test_chosen_but_not_dispatched():
    s=fixture();stop(s,2,5)
    o=m.summarize(s)
    assert o['node_counts']['chosen_H3']==10 and o['node_counts']['dispatched_H3']==9
    assert o['rows'][2]['nodes'][0]['panels'][1]['category']=='descriptive_H3'


def test_all_unreached_full48_missing():
    s=fixture()
    for i in range(8):stop(s,i,0)
    o=m.summarize(s)
    assert o['node_counts']['not_reached']==24
    assert o['error_categories']['uncomputed_H1_missing']==24
    assert o['error_categories']['uncomputed_H3_missing']==24
    assert o['pairs'][0]['raw_counts']==dict(missing=20,same=0,different=0)
    assert o['pairs'][0]['state_counts']==dict(missing=20,same=1,different=0)


def test_partial_H1_not_same_action_complete():
    s=fixture();stop(s,0,7)
    o=m.summarize(s)
    assert o['rows'][0]['nodes'][0]['actual_selected_steps']==2
    assert o['rows'][0]['nodes'][0]['panels'][0]['category']=='unexecuted_or_incomplete_H1'
    assert o['pairs'][0]['steps'][8]['raw_bits_equal'] is None


def test_actual_divergence_and_signed_zero():
    s=fixture();s['rows'][2]['nodes'][0]['panels'][1]['first5_raw'][0][0]=-0.0
    s['rows'][2]['steps'][7]['state'][0]=-0.0
    refresh_pairs(s);o=m.summarize(s);p=o['pairs'][0]
    assert p['first_observed_raw_divergence']==6 and p['first_observed_state_divergence']==7
    assert p['steps'][5]['raw_missing_reason']=='prefix5_not_in_summary_ABI'


@pytest.mark.parametrize('change',[
    lambda s:s.update(schema='producer.report'),
    lambda s:s.update(node_denominator=23),
    lambda s:s['rows'].pop(),
    lambda s:s['pairs'].reverse(),
    lambda s:s['pairs'][0]['steps'][0].update(state_bits_equal=False),
    lambda s:s['rows'][0]['nodes'][0].update(executed_horizon=3),
    lambda s:s['rows'][0]['nodes'][0]['decision'].update(program_type='reconfigure'),
    lambda s:s['rows'][0]['nodes'][0]['panels'][0].update(selected_for_execution=False),
    lambda s:s['rows'][0]['nodes'][2]['panels'][1].update(actual_Q_at_predicted_endpoint=0.,error_decomposition={}),
    lambda s:s['rows'][0]['nodes'][0]['decision'].pop('extension_type'),
    lambda s:s['rows'][0]['steps'][2].update(state=[True]*7),
    lambda s:s['rows'][0]['nodes'][0]['panels'][0].update(first5_raw=[[.1,0.]]*5),
])
def test_fail_closed_mutations(change):
    s=fixture();change(s)
    with pytest.raises(ValueError):m.summarize(s)


def test_false_refusal_and_post_refusal_dispatch_rejected():
    for returned in (5,6):
        s=fixture();stop(s,2,returned);n=s['rows'][2]['nodes'][0]
        n.update(status='refused',executed_horizon=None)
        if returned==6:n['decision']['continue_expanding']=False
        for p in n['panels']:p['selected_for_execution']=False
        with pytest.raises(ValueError):m.summarize(s)


def test_real_file_cli_roundtrip_hash_and_no_overwrite(tmp_path,capsys):
    p=tmp_path/'input.json';p.write_bytes(m.encoded(fixture()));h=m.sha(p)
    out=tmp_path/(m.NAME+'_fixture.json')
    m.main(['analyze','--summary',str(p),'--summary-sha',h,'--output',str(out)])
    a=json.loads(out.read_bytes());assert a['upstream_authority']==fixture()['authority']
    m.main(['verify','--analysis',str(out),'--analysis-sha',m.sha(out)])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['verified'] is True
    with pytest.raises(FileExistsError):m.main(['analyze','--summary',str(p),'--summary-sha',h,'--output',str(out)])
    with pytest.raises(ValueError):m.analyze(str(p),'0'*64)


def test_duplicate_and_nonfinite(tmp_path):
    for body in (b'{"x":1,"x":2}',b'{"x":NaN}'):
        p=tmp_path/'bad.json';p.write_bytes(body)
        with pytest.raises(ValueError):m.read(str(p),m.sha(p))
