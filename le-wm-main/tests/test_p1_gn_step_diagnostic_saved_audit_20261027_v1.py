"""Saved actual arrays and in-memory tampering only; no tmp/model/physics."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import p1_gn_step_diagnostic_saved_audit_20261027_v1 as m


@pytest.fixture(scope='module')
def arrays():
    reader=m.Reader();b=m.ACTUAL;r=reader.read(b/'report.json',m.REPORT_SHA)
    binding=reader.read(b/'execution_binding.json');p=reader.read(binding['prepared'],m.PREPARE_SHA)
    events=[json.loads(line) for line in (b/'events.jsonl').read_bytes().splitlines()]
    return (p,r,events,[reader.read(b/f'eval_{i:02d}.json') for i in range(16)],
        [reader.read(b/f'root_{i}.selection.json') for i in range(2)],
        [reader.read(b/f'root_{i}.result.json') for i in range(2)],
        [reader.read(x['task_path']) for x in p['roots']],
        [reader.read(x['component_path']) for x in p['roots']])


def test_actual_complete_arrays_independently_consistent(arrays):
    r=m.check_arrays(*arrays)
    assert r['resources']=={k:dict(attempted=32,completed=32) for k in ('predict','action_encoder')}
    assert r['event_count']==192 and r['full_H2_evaluations']==16 and r['complete_H2_packet_arrays']==32
    assert r['selected_replay_H2_evaluations']==6 and not r['P1_admission']
    assert all(row['selected_scale']==.25 and row['strict_improvement'] for row in r['rows'])


@pytest.mark.parametrize('defect',['raw','logits','history','retained_latent','Q_cost','selection','missing_trial',
    'scale_order','operation_resource','operation_return','packet_array','replay','root','task_goal','ad_scope','extra_event','endpoint'])
def test_in_memory_tampering_fails_closed(arrays,defect):
    p,r,events,evals,selections,results,tasks,components=deepcopy(arrays)
    if defect=='raw':evals[1]['commands'][0][0]+=.01
    elif defect=='logits':evals[1]['logits'][0][0]+=.01
    elif defect=='history':evals[1]['contexts'][0]['action_history'][-1][0]+=.01
    elif defect=='retained_latent':evals[1]['contexts'][1]['latent_history'][0][0]+=.01
    elif defect=='Q_cost':evals[1]['cost']+=.01
    elif defect=='selection':selections[0]['winner_index']=1
    elif defect=='missing_trial':evals.pop(4)
    elif defect=='scale_order':p['scales'][0]=.0625
    elif defect=='operation_resource':next(e for e in events if e['kind']=='module_attempt')['entry']['resource']='jacobian'
    elif defect=='operation_return':next(e for e in events if e['kind']=='module_return')['entry']['completed']=False
    elif defect=='packet_array':next(e for e in events if e['kind']=='native_packet')['context']['latent_history'][-1][0]+=.01
    elif defect=='replay':evals[7]['contexts'][-1]['latent_history'][-1][0]+=.01
    elif defect=='root':p['roots'][1]['root_sha256']=p['roots'][0]['root_sha256']
    elif defect=='task_goal':tasks[0]['task']['goal'][0]+=.01
    elif defect=='ad_scope':r['new_AD_calls']=1
    elif defect=='extra_event':events.append(dict(kind='unmetered_work'))
    else:p['roots'][0]['rejected_logits'][0][0]+=.01
    with pytest.raises(ValueError):m.check_arrays(p,r,events,evals,selections,results,tasks,components)


def test_segment_exact_float32_no_projection_or_raw_interpolation():
    x=np.full((2,10),.4);y=np.full((2,10),-.3)
    x=x.astype(np.float32).astype(float);y=y.astype(np.float32).astype(float)
    for s in m.SCALES:
        expected=(x+s*(y-x)).astype(np.float32)
        assert np.array_equal(m.segment(x,y,s,1.7),expected)
    with pytest.raises(ValueError):m.segment(x,y,.75,1.7)
    with pytest.raises(ValueError):m.segment(x*10,y,.25,1.7)


def test_reader_rejects_external_hash_and_drift(monkeypatch):
    reader=m.Reader()
    with pytest.raises(ValueError):reader.read(m.ACTUAL/'report.json','0'*64)
    reader=m.Reader();reader.bind(m.ACTUAL/'report.json',m.REPORT_SHA)
    monkeypatch.setattr(m,'sha',lambda p:'0'*64)
    with pytest.raises(ValueError):reader.recheck()


def test_full_actual_hash_and_source_audit():
    r=m.audit()
    assert r['status']=='saved_arrays_audited_not_admitted' and r['model_unchanged']
    assert r['actual_report_sha256']==m.REPORT_SHA
    assert r['new_neural_calls']==r['new_physics_steps']==0 and not r['P1_admission']
    assert str(m.ACTUAL/'report.json') in r['bindings_sha256']
