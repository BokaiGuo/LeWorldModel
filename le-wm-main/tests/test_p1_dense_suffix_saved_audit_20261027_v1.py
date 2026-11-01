"""Pure NumPy/mock and readonly actual fixtures, no simulator/GPU/tmp."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import p1_dense_suffix_saved_audit_20261027_v1 as m

PREPARED=m.ROOT/'results/p1_dense_suffix_physical_main_20261027_v1_prepare01.json'
REPORT=m.ROOT/'results/p1_dense_suffix_physical_main_20261027_v1_main01/report.json'
PREPARE_SHA='0ad44a85d15a14ede21f1dcc79fdf041f549aa27e5a31e7885c9f1057f5e7e4f'
REPORT_SHA='76fb65359a9ebf9e42a25a89aaab48ef2ae6918640e7b5e18ddcfcda5675d9a6'


@pytest.fixture(scope='module')
def actual():
    return m.load(m.pure.Reader(),PREPARED,PREPARE_SHA,REPORT,REPORT_SHA)


def test_numpy_bank_fixed_persistent_compass_not_sparse():
    bank=m.bank();assert len(bank)==17 and bank[0]['logits']==[0.]*10
    assert [p['index'] for p in bank]==list(range(17))
    assert [p['direction'] for p in bank[1:9]]==[list(d) for d in m.DIRECTIONS]
    for p in bank[1:]:
        x=np.asarray(p['logits']);assert np.array_equal(x,x.astype(np.float32))
        assert np.linalg.norm(x)<=p['radius']<=1.5<.75*np.sqrt(5)
        assert np.allclose(np.linalg.norm(x),p['radius'],atol=1e-6,rtol=0)
        assert np.array_equal(x.reshape(5,2),np.tile(x[:2],(5,1)))
    for start in (1,9):
        for i in range(4):assert np.array_equal(bank[start+i]['logits'],-np.asarray(bank[start+4+i]['logits']))


def test_contract_maxima_not_required_actual_counts():
    c=m.contract();assert c['branches']==102 and c['png_cap']==306
    assert c['step_cap']==1020 and c['render_cap']==1122 and not c['P1_admission']


def mock_fixture():
    plans=[];rows=[];goals={}
    for root in ('root0','root1'):
        goals[root]=[0.]*7
        for role in m.ROLES:
            for probe in m.bank():
                arm=role+'_dense_'+str(probe['index'])
                identity=dict(root_sha256=root,arm=arm,selection_sha256=root+arm)
                plans.append(dict(**identity,role=role,probe=probe))
                cp=dict(step=5,state=[0.]*7,rgb_sha256='a'*64)
                state=[0.,0.,probe['radius'],0.,probe['radius']/10,0.,0.]
                rows.append(dict(**identity,status='completed',full_horizon=True,checkpoints=[dict(step=0),cp],
                    body_dynamics={'5':{'mock_only':True}},final_state=state,returned_steps=10,
                    pose_diagnostics=m.pure.metrics(goals[root],state),official_evaluation=dict(success=False)))
    return plans,rows,goals


def test_mock102_all_probe_responses_and_object_arithmetic():
    p,r,g=mock_fixture();a,l=m.responses(p,r,g)
    assert sum(x['planned'] for x in l)==102 and sum(x['full_completed'] for x in l)==102
    assert a['groups'][0]['responses'][-1]['response']['object_xy_delta']==[1.5,0.]
    assert a['groups'][0]['responses'][-1]['response']['wrapped_object_angle_delta']==pytest.approx(.15)


def test_mock_partial_zero_reference_does_not_remove_planned_or_fake_pairs():
    p,r,g=mock_fixture();r[0].update(status='failed',full_horizon=False,attempted_steps=7,returned_steps=7)
    a,l=m.responses(p,r,g)
    assert l[0]['planned']==17 and l[0]['full_completed']==16 and l[0]['paired_available']==0
    assert len(a['groups'][0]['responses'])==17 and all(v['response'] is None for v in a['groups'][0]['responses'])
    assert len(l[0]['partial_layer'])==1 and l[1]['paired_available']==17


@pytest.mark.parametrize('defect',['root','order','bank','body5','state5','rgb5','missing_row'])
def test_mock_identity_bank_prefix_fail_closed(defect):
    p,r,g=mock_fixture()
    if defect=='root':r[1]['root_sha256']='other'
    elif defect=='order':p[1],p[2]=p[2],p[1]
    elif defect=='bank':p[1]['probe']['radius']=.1
    elif defect=='body5':r[1]['body_dynamics']['5']['mock_only']=False
    elif defect=='state5':r[1]['checkpoints'][1]['state'][0]=1.
    elif defect=='rgb5':r[1]['checkpoints'][1]['rgb_sha256']='b'*64
    elif defect=='missing_row':r.pop()
    with pytest.raises(ValueError):m.responses(p,r,g)


def test_actual102_including_all6_guard_failures(actual):
    p,r,packets=actual
    for plan,row in zip(p['inputs']['plans'],r['rows']):m.row_check(plan,row,packets[plan['root_sha256']])
    a,l=m.responses(p['inputs']['plans'],r['rows'],{k:v['goal'] for k,v in packets.items()})
    assert a==json.loads((REPORT.parent/'analysis.json').read_text())
    assert [v['full_completed'] for v in l]==[17,11,17,17,17,17]
    assert sum(v['planned'] for v in l)==102 and sum(v['full_completed'] for v in l)==96
    assert [x['probe_index'] for x in l[1]['partial_layer']]==[4,5,6,12,13,14]
    assert all(v['prefix5_records_checked']==17 for v in l)
    assert all(a['groups'][1]['responses'][i]['response'] is None for i in (4,5,6,12,13,14))


def test_guard_after_tenth_return_still_not_certified_fullH2(actual):
    p,r,packets=actual;row=r['rows'][21]
    assert row['returned_steps']==10 and not m.full(row)
    m.row_check(p['inputs']['plans'][21],row,packets[row['root_sha256']])


def test_actual_partial_journal_exact_not_full_budget(actual):
    p,r,packets=actual
    with (REPORT.parent/'journal.jsonl').open() as f:v=m.journal_check(f,p,r,packets)
    assert v['events']==14056 and v['counts_exact_from_durable_events'] and not v['evidence_gaps']
    assert v['reported_counts']['step']==dict(attempted=1010,completed=1010)
    assert v['reported_counts']['render']==dict(attempted=1112,completed=1112)
    assert v['closed_branches']==102


@pytest.mark.parametrize('defect',['command','partial_angle','partial_reward','guard_count','guard_continuation','partial_body','returned_count'])
def test_actual_partial_tampering(defect,actual):
    p,r,packets=actual;row=deepcopy(r['rows'][21]);plan=p['inputs']['plans'][21]
    if defect=='command':row['trace'][0]['command'][0]+=.01
    elif defect=='partial_angle':row['pose_diagnostics']['independent']['wrapped_angle_error']+=.1
    elif defect=='partial_reward':row['trace'][-1]['reward']+=1
    elif defect=='guard_count':row['guard_receipt']['observations'][-1]['canonical_agent_pixels']=1
    elif defect=='guard_continuation':row['guard_receipt']['observations'][-2]['canonical_agent_pixels']=0
    elif defect=='partial_body':row['body_dynamics']['5']['block']['unwrapped_angle']+=.1
    elif defect=='returned_count':row['returned_steps']=9
    with pytest.raises(ValueError):m.row_check(plan,row,packets[plan['root_sha256']])


@pytest.mark.parametrize('retained',[False,True])
def test_guard_diagnosis_boundary_vs_unknown_pixels(actual,retained):
    p,r,_=actual;row=r['rows'][21];v=m.guard_diagnosis(p['inputs']['plans'][21],row,{10:0} if retained else {})
    assert v['boundary_evidence']=='left of native canvas' and v['entire_default_agent_circle_left_of_canvas']
    assert v['failed_frame_PNG_retained'] is retained
    assert ('unknown' in v['failed_frame_pixel_check']) is (not retained)
    assert v['occlusion_vs_color_change'].startswith('unknown')


def test_guard_inside_canvas_cannot_be_declared_boundary(actual):
    p,r,_=actual;row=deepcopy(r['rows'][21]);row['final_state'][0]=100.
    assert m.guard_diagnosis(p['inputs']['plans'][21],row,{})['boundary_evidence'].startswith('center inside')


def sealed_events(values):
    previous=None;out=[]
    for index,v in enumerate(values):
        event=dict(v,index=index,previous_sha256=previous);previous=m.pure.digest(event)
        out.append(m.pure.core.canonical(dict(event,event_sha256=previous))+'\n')
    return out


def test_mock_event_hash_and_unknown_neural_entry_rejected():
    lines=sealed_events([dict(kind='run_started'),dict(kind='module_predict')])
    bad=lines[0].replace('"index": 0','"index": 1')
    with pytest.raises(ValueError):list(m.events([bad]))
    plans,rows,_=mock_fixture();p=dict(contract=m.contract(),inputs=dict(plans=plans))
    report=dict(prepare_sha256='a'*64,rows=rows,status='partial_or_failed',counts={k:dict(attempted=0,completed=0) for k in m.CAPS},errors=[])
    unknown=sealed_events([dict(kind='run_started',prepare_sha256='a'*64,contract=p['contract']),dict(kind='module_predict')])
    with pytest.raises(ValueError,match='known event'):m.journal_check(unknown,p,report)


def test_event_unknown_fields_not_silently_ignored():
    with pytest.raises(ValueError,match='exact journal field'):
        m.event_fields(dict(kind='step_returned',selection_sha256='a'*64,step=1,hidden_neural_call=True))


def test_mock_all_not_started_denominator_kept():
    plans,rows,goals=mock_fixture()
    rows=[dict(root_sha256=p['root_sha256'],arm=p['arm'],selection_sha256=p['selection_sha256'],status='not_started') for p in plans]
    a,l=m.responses(plans,rows,goals)
    assert sum(v['planned'] for v in l)==102 and sum(v['partial_failed_not_started'] for v in l)==102
    assert all(x['response'] is None for g in a['groups'] for x in g['responses'])
