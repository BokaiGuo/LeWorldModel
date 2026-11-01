"""RAM arrays only: no neural, physics, prepare or disk fixtures."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_observed_suffix_cuda_20261027_v1 as m


def helper(stem):
    spec=importlib.util.spec_from_file_location(stem,Path(__file__).with_name(stem+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


h=helper('test_p1_constructive_blind_planners_cuda_20261027_v1')


def data():
    t=h.inputs()['gn_task']; t['goal']=[1.,2.]+[0.]*190
    z=np.zeros(192,np.float32); prefix=np.full((5,2),.125,np.float32)
    c=m.observed_context(t,z,prefix)
    return dict(task=t,observed_context=c,original_prefix=prefix,original_suffix=np.zeros((5,2),np.float32),
        record=dict(source_slot=0,arm=m.base.ARMS[0],blind_sha256='a'*64,parent_task_id='b'*64,observed_problem_id='c'*64))


class Backend(h.d.LinearBackend):
    def __init__(self,ledger): super().__init__(); self.ledger=ledger
    def step(self,c,a):
        for r in ('action_encoder','predict'):
            e=self.ledger.begin(r); self.ledger.end(e)
        return super().step(c,a)


def run(d=None,backend=Backend):
    d=data() if d is None else d; events=[]; ledger=m.Ledger(events.append)
    b=backend(ledger); result=m.execute_branch(d,b,ledger,events.append)
    return result,b,ledger,events


def test_budget_includes_ad_forward_and_full_pool():
    r,b,l,e=run()
    assert b.calls==5 and len(r['candidates'])==3
    m.base.exact_counts(l.counts(),m.budget())
    assert r['winner']['kind']=='fresh_observed_ridge'
    assert [x['kind'] for x in e].index('fresh_J_saved')<[x['kind'] for x in e].index('ridge_frozen_before_prediction')


def test_history_is_actual_own_prefix_not_imagined_terminal():
    d=data(); c=d['observed_context']; t=d['task']
    assert c['latent_history'][:2]==t['root_context']['latent_history'][1:]
    assert c['latent_history'][-1]==[0.]*192
    assert c['action_history'][0]==t['root_context']['action_history'][-1]
    assert c['action_history'][1]==m.base.encode.core.normalized_tokens(t,d['original_prefix'])[0].astype(float).tolist()


def test_ties_choose_zero_but_still_pay_all_five():
    d=data(); d['task']['goal']=[0.]*192
    r,b,l,e=run(d)
    assert r['selected_index']==0 and b.calls==5
    m.base.exact_counts(l.counts(),m.budget())


def test_original_suffix_can_win():
    d=data(); a=np.zeros((5,2),np.float32); a[0]=[.5,.25]
    d['original_suffix']=a; d['task']['goal']=[.5,.5]+[0.]*190
    r,*_=run(d)
    assert r['selected_index']==1 and r['winner']['terminal_cost']==0


@pytest.mark.parametrize('value',[np.zeros((10,2)),np.full((5,2),2.),np.full((5,2),.1),np.full((5,2),np.nan)])
def test_raw_no_silent_clipping_rounding_or_wrong_horizon(value):
    with pytest.raises(ValueError):m.raw5(value)


def test_independent_replay_mismatch_fails():
    class Broken(Backend):
        def replay(self,c,a):
            r=super().replay(c,a); r['latent_history'][-1][0]+=1; return r
    with pytest.raises(ValueError,match='independent H1'):run(backend=Broken)


def test_saved_nominal_and_J_survive_svd_failure(monkeypatch):
    events=[]; l=m.Ledger(events.append)
    def fail(*a,**k):raise RuntimeError('SVD fault')
    monkeypatch.setattr(np.linalg,'svd',fail)
    with pytest.raises(RuntimeError,match='SVD fault'):m.execute_branch(data(),Backend(l),l,events.append)
    assert any(e['kind']=='zero_nominal_saved' for e in events)
    assert any(e['kind']=='fresh_J_saved' for e in events)
    assert l.counts()['svd']==dict(attempted=1,completed=0)


@pytest.mark.parametrize('resource',['predict','action_encoder'])
def test_sixth_dispatch_rejected_before_call(resource):
    l=m.Ledger(lambda e:None)
    for _ in range(5):l.end(l.begin(resource))
    with pytest.raises(ValueError,match='five-forward'):l.begin(resource)


def test_plan_seal_covers_literal_full10_and_metadata():
    d=data(); r,*_=run(d); p=m.seal_plan(d,r)
    assert p['commands'][:5]==d['original_prefix'].astype(float).tolist()
    assert p['commands'][5:]==r['winner']['commands']
    assert np.array_equal(np.asarray(p['commands']),np.asarray(p['commands'],np.float32))
    s=p.pop('selection_sha256'); assert s==m.base.encode.core.digest(p)
    p['commands'][0][0]=0.; assert s!=m.base.encode.core.digest(p)


def test_physical_selection_direct_report_pointers(monkeypatch):
    from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as physical
    rows=[]; refs=[dict(root_sha256=str(i)*64,blind_sha256=chr(97+i)*64) for i in range(2)]
    for slot in range(2):
        for arm in m.base.ARMS:
            rows.append(dict(source_slot=slot,arm=arm,blind_sha256=refs[slot]['blind_sha256'],commands=np.zeros((10,2)).tolist()))
    report=dict(schema=m.NAME+'.report',status='observed_suffix_plans_sealed_not_admitted',errors=[],rows=rows)
    manifest=dict(schema=physical.NAME+'.selection',sourcepack='/external',sourcepack_sha256='a'*64,
        source_prepared='/external_prepare',source_prepare_sha256='b'*64,
        roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(refs)],arms=list(m.base.ARMS),
        plans=[],selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=180)
    for i,r in enumerate(rows):
        p=dict(root_slot=r['source_slot'],arm=r['arm'],commands=r['commands'],compiler_report='/mock/report.json',
            compiler_report_SHA='c'*64,compiler_schema=report['schema'],compiler_status=report['status'],
            commands_pointer=f'/rows/{i}/commands',source_slot_pointer=f'/rows/{i}/source_slot',
            blind_sha_pointer=f'/rows/{i}/blind_sha256',arm_pointer=f'/rows/{i}/arm')
        p['selection_digest']=physical.source.digest(p); manifest['plans'].append(p)
    monkeypatch.setattr(physical,'sha',lambda p:'c'*64); monkeypatch.setattr(physical,'read',lambda p:report)
    plans,_=physical.selection(manifest,dict(references=refs))
    assert len(plans)==4
    report['status']='partial_or_failed'
    with pytest.raises(ValueError):physical.selection(manifest,dict(references=refs))


def test_scope_and_no_free_computation_claim():
    c=m.contract(); assert c['total']==dict(predict=20,action_encoder=20,jacobian=4,svd=4)
    assert c['ADforward_in_predict_per_branch']==1 and c['no_dedup']
    assert c['P1_admission'] is c['P2_admission'] is c['fair_budget_claim'] is False
    assert c['new_encoder']==c['new_projector']==c['new_physics']==0


def test_oneAD_nested_batched_grad_and_second_call_cap():
    t=h.fake_torch(); old=t.autograd.grad
    t.autograd.functional.jacobian=lambda:t.autograd.grad(is_grads_batched=True)
    a=m.OneAD(t,m.base.ARMS[1],lambda e:None,lambda:None)
    with a:
        assert t.autograd.functional.jacobian()=='returned'
        with pytest.raises(ValueError,match='one batched-grad'):t.autograd.functional.jacobian()
    assert t.autograd.grad is old and a.events[1]['is_grads_batched']
    assert a.counts()=={k:dict(attempted=1,completed=1) for k in ('grad','jacobian')}


def test_oneAD_forbids_unscoped_grad():
    t=h.fake_torch()
    with m.OneAD(t,m.base.ARMS[1],lambda e:None,lambda:None):
        with pytest.raises(ValueError,match='only inside'):t.autograd.grad()


def test_inherited_suffix_gzip_partial_final_and_broken(monkeypatch):
    import zlib
    s=object.__new__(m.Store); s.files={}; s.comp=zlib.compressobj(6,zlib.DEFLATED,31); s.closed=s.broken=False
    def write(name,raw,append=False,final=False):
        assert append or name not in s.files
        assert sum(map(len,s.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        s.files[name]=s.files.get(name,b'')+raw
    monkeypatch.setattr(s,'_write',write)
    s.event(dict(kind='attempt')); assert not m.base.inflate(s.files['events.jsonl.gz'])[1]
    s.close(); assert m.base.inflate(s.files['events.jsonl.gz'])[1]
    with pytest.raises(ValueError):s.event({})
    s=object.__new__(m.Store); s.comp=zlib.compressobj(6,zlib.DEFLATED,31); s.closed=s.broken=False
    def fail(*a,**k):raise OSError('RAM disk fault')
    monkeypatch.setattr(s,'_write',fail)
    with pytest.raises(OSError):s.event({})
    assert s.broken
    with pytest.raises(ValueError):s.close()
