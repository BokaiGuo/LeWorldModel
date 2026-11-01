"""Pure-array/saved-input/mock dispatch tests; no torch, model or physics run."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_direct_fd_cuda_20261027_v1 as m


META=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10)


def task():
    return dict(root_context=dict(META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist()),
        action_mean=[0.,0.],action_std=[1.,1.])


def path(t, raw, j):
    previous=deepcopy(t['root_context']);out=[];r=np.asarray(raw).reshape(20)
    for k in range(2):
        z=(j[:,:10*(k+1)]@r[:10*(k+1)]).astype(np.float32).tolist()
        c=dict(META,latent_history=previous['latent_history'][1:]+[z],
            action_history=np.concatenate((previous['action_history'],m.tokens(t,raw)[k:k+1]))[-2:].tolist())
        out.append(c);previous=c
    return out


def fixture():
    t=task();j=np.arange(192*20,dtype=float).reshape(192,20)/10000
    p=dict(roots=[dict(root_index=i,root_sha256=str(i),task_id='t'+str(i),task=deepcopy(t),
        logit_jacobian=j.tolist(),nominal_terminal=np.zeros(192).tolist()) for i in range(2)],directions=m.directions())
    rows=[]
    for r in p['roots']:
        for d in p['directions']:
            for eps in m.EPS:
                for sign in (-1,1):
                    e=dict(order=len(rows),root_index=r['root_index'],root_sha256=r['root_sha256'],task_id=r['task_id'],
                        direction=d['id'],sign=sign,**m.perturb(d['vector'],eps,sign))
                    rows.append(dict(**e,contexts=path(t,e['commands'],j)))
    return p,rows


class Memory:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


def finish_packet(meter):
    for resource in ('action_encoder','predict'):
        token=meter.begin(resource);meter.end(token)
    meter.emit('native_packet',mode='whole_replay',packet=meter.packet,context={})


def test_directions_are_fixed_packet_basis_equal_weight_and_seeded_unit():
    ds=m.directions();a,b,c,r=[np.asarray(d['vector']) for d in ds]
    assert np.flatnonzero(a).tolist()==[0] and a[0]==1
    assert np.flatnonzero(b).tolist()==[10] and b[10]==1
    np.testing.assert_array_equal(c,((a+b)/np.sqrt(2)).astype(np.float32))
    expected=np.random.Generator(np.random.PCG64(20261027)).standard_normal(20)
    expected=(expected/np.linalg.norm(expected)).astype(np.float32)
    np.testing.assert_array_equal(r,expected)
    np.testing.assert_allclose([np.linalg.norm(x) for x in (a,b,c,r)],1,rtol=1e-7)
    assert ds==m.directions()


@pytest.mark.parametrize('eps',m.EPS)
@pytest.mark.parametrize('index',range(4))
def test_exact_numpy_float32_tanh_and_sign_symmetry(index,eps):
    d=m.directions()[index]['vector']
    minus=m.perturb(d,eps,-1);plus=m.perturb(d,eps,1)
    x=np.asarray(d,np.float32)*np.float32(eps)
    np.testing.assert_array_equal(plus['logits'],x.reshape(2,10))
    np.testing.assert_array_equal(plus['commands'],np.tanh(x).reshape(10,2))
    np.testing.assert_array_equal(minus['commands'],-np.asarray(plus['commands']))


def test_error_metrics_no_scientific_pass_floor_and_zero_denominator_explicit():
    r=m.error_metrics(np.full(192,2.),np.ones(192))
    assert r['absolute_l2']==pytest.approx(np.sqrt(192)) and r['absolute_rmse']==1 and r['relative_l2']==1
    r=m.error_metrics(np.ones(192),np.zeros(192))
    assert r['relative_l2'] is r['cosine'] is None and r['symmetric_relative_l2']==1


def test_complete_mock_fd_analysis_includes_full_vectors_and_epsilon_stability():
    p,rows=fixture();a=m.analyze(p,rows)
    assert a['comparisons']==16 and a['epsilon_stability_pairs']==8
    assert a['scientific_pass_threshold'] is None and a['scientific_verdict']=='not_assigned'
    for r in a['rows']:
        assert [e['epsilon_requested'] for e in r['epsilons']]==list(m.EPS)
        assert all(len(e['fd'])==len(e['saved_jv'])==192 for e in r['epsilons'])
        f0,f1=[np.asarray(e['fd']) for e in r['epsilons']]
        assert r['epsilon_stability_l2']==float(np.linalg.norm(f0-f1))


@pytest.mark.parametrize('corruption',['missing','order','task','root','ordinal','raw','history','layout'])
def test_analysis_refuses_mismatched_perturbation_or_causal_path(corruption):
    p,rows=fixture()
    if corruption=='missing':rows.pop()
    elif corruption=='order':rows[0],rows[1]=rows[1],rows[0]
    elif corruption=='task':rows[0]['task_id']='other'
    elif corruption=='root':rows[0]['root_sha256']='other'
    elif corruption=='ordinal':rows[0]['order']=7
    elif corruption=='raw':rows[0]['commands'][5][0]+=.01
    elif corruption=='history':rows[0]['contexts'][1]['action_history'][0][0]+=.01
    else:rows[0]['contexts'][0]['layout']='wrong'
    with pytest.raises(ValueError):m.analyze(p,rows)


def test_wrong_saved_chain_is_exposed_by_full192_comparison_not_hidden_pass():
    v=np.zeros(20);v[0]=1;j=np.ones((192,20))
    good=m.compare(j,np.zeros(192),v,.001,-np.full(192,float(np.float32(.001))),np.full(192,float(np.float32(.001))))
    bad=m.compare(2*j,np.zeros(192),v,.001,-np.full(192,float(np.float32(.001))),np.full(192,float(np.float32(.001))))
    assert good['absolute_l2']==0 and bad['relative_l2']==.5


def test_meter_exact64_each_module_and_no_extra_dispatch():
    memory=Memory();meter=m.Meter(memory)
    for i in range(32):
        meter.start(dict(order=i));finish_packet(meter);finish_packet(meter)
    assert meter.counts()=={k:dict(attempted=64,completed=64) for k in ('predict','action_encoder')}
    meter.start(dict(order=32))
    with pytest.raises(ValueError,match='cap'):meter.begin('action_encoder')


def test_meter_partial_encoder_completion_survives_predict_failure_no_retry():
    meter=m.Meter(Memory());meter.start(dict(order=0))
    a=meter.begin('action_encoder');meter.end(a);meter.begin('predict')
    assert meter.counts()==dict(action_encoder=dict(attempted=1,completed=1),predict=dict(attempted=1,completed=0))
    with pytest.raises(ValueError):meter.start(dict(order=0))
    with pytest.raises(ValueError):meter.begin('predict')


def test_meter_sync_error_keeps_nonrefundable_attempt():
    def fail():raise RuntimeError('sync fault')
    meter=m.Meter(Memory(),sync=fail);meter.start(dict(order=0));a=meter.begin('action_encoder')
    with pytest.raises(RuntimeError):meter.end(a)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=0)


def test_meter_refuses_grad_enabled_and_wrong_module_order():
    meter=m.Meter(Memory(),no_grad=lambda:False);meter.start(dict(order=0))
    with pytest.raises(ValueError):meter.begin('action_encoder')
    meter=m.Meter(Memory());meter.start(dict(order=0))
    with pytest.raises(ValueError):meter.begin('predict')


def test_storage_exclusive_cap_final_reserve_and_disk_floor(tmp_path,monkeypatch):
    s=m.Storage(tmp_path/'out');s.put('a.json',dict(x=1))
    with pytest.raises(ValueError):m.Storage(tmp_path/'out')
    s.written=m.CAP-m.FINAL_RESERVE
    with pytest.raises(ValueError,match='cap'):s.put('full.json',dict(x=2))
    s.put('report.json',dict(errors=['retained']),final=True)
    s.written=0
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError,match='floor'):s.event(dict(x=1))


def test_cleanup_notes_do_not_erase_the_first_execution_fault():
    exc=RuntimeError('first predictor fault');exc.add_note('hook cleanup secondary fault')
    saved=m.failure('execution',exc)
    assert saved['message']=='first predictor fault'
    assert saved['notes']==['hook cleanup secondary fault']


def test_actual_source_binding_raw_derivatives_not_auditor_receipt():
    p=m.material()
    assert len(p['roots'])==2 and p['predict_cap']==p['action_encoder_cap']==64
    assert p['image_cap']==p['AD_cap']==p['physics_cap']==0
    assert p['scientific_pass_threshold'] is None
    for r in p['roots']:
        assert r['raw_rows']==192 and r['chain_at_zero_is_identity']
        assert np.asarray(r['logit_jacobian']).shape==(192,20)
        assert 'direct_journal.jsonl' in r['original_journal']
    assert not any('saved_audit' in path or 'selfcheck' in path for path in p['input_bindings'])


@pytest.mark.parametrize('corruption',['row','point','forward'])
def test_raw_zero_witness_corruption_is_rejected(corruption):
    directory=m.SAVED/'root_0_full_H2_GN'
    t=m.read(m.SAVED/'task_0.json')['task'];c=m.read(directory/'report.json')
    events=[json.loads(x) for x in (directory/'direct_journal.jsonl').read_text().splitlines()]
    if corruption=='row':next(e for e in events if e['kind']=='jacobian_row')['raw_control_derivative'][0]+=.1
    elif corruption=='point':c['direct_runs'][0][1]['derivative']['at_logits'][0][0]=.1
    else:next(e for e in events if e['kind']=='jacobian_forward_packet')['context']['latent_history'][-1][0]+=.1
    with pytest.raises(ValueError):m.extract_zero(t,c,events)


def test_prepare_and_verify_only_saved_arrays(tmp_path):
    out=tmp_path/'prepared';r=m.prepare(out)
    assert r['predict_cap']==r['action_encoder_cap']==64
    p=m.verify(out,r['prepare_sha256']);assert len(p['roots'])==2
    assert sum(x.stat().st_size for x in out.iterdir())<m.CAP
    with pytest.raises(ValueError):m.prepare(out)
