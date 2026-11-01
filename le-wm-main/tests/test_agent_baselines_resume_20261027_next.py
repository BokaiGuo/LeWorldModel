"""RAM-only backend/interface mocks, no neural model, tmp, GPU or simulator."""
from copy import deepcopy
from io import BytesIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_baselines_resume_20261027_next as m

# Reuse immutable NumPy fixture builders, not previous model/experiment runners.
spec=importlib.util.spec_from_file_location('next_suffix_fixture',Path(__file__).with_name('test_moving_history_observed_suffix_cuda_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class RAM:
    def __init__(self):self.files={};self.events=[]
    def _write(self,name,data,**kw):
        assert name not in self.files
        self.files[name]=data
    def event(self,e):self.events.append(deepcopy(e))


def execute(slot=0,arm=None,fail=None):
    b=h.branch(slot,arm);store=RAM();meter=NS(stage=None,expected_input=None,last_native=None,require_native=True)
    modules=m.Modules(NS(),store,'branch_00',meter)
    backend=h.Backend(b,modules,fail);row=dict(index=0,branch_id=b['branch_id'])
    try:m.evaluate(b,backend,meter,store,row)
    finally:row.update(module_ledger=deepcopy(modules.entries),resources=modules.counts())
    return b,store,row,backend


def load(store,a):
    data=store.files[a['file']];assert m.prior.observed.enc.digest(data)==a['sha256']
    return np.load(BytesIO(data),allow_pickle=False)


@pytest.mark.parametrize('slot',[0,1])
@pytest.mark.parametrize('arm',m.prior.ARMS)
def test_full_fixed64_plus_one_independent_replay_native_path(slot,arm):
    b,s,r,backend=execute(slot,arm);m.validate_row(b,r);m.native_arrays(b,r,lambda x:load(s,x))
    assert backend.calls==list(range(65))
    assert len(r['evaluations'])==65 and len(r['module_ledger'])==195
    assert r['resources']=={k:dict(attempted=65,completed=65) for k in m.Modules.ORDER}
    p=m.model_plan(b,r)
    assert p['commands'][:5]==b['prefix_commands'] and p['commands'][5:]==r['suffix_commands']
    assert p['blind_sha256']==b['blind_sha256'] and p['prefix_arm']==arm
    assert p['selection_sha256']==m.core.digest({k:v for k,v in p.items() if k!='selection_sha256'})
    assert sum(map(len,s.files.values()))<2*1024**2
    events=s.events;kinds=[e['kind'] for e in events]
    assert kinds.index('proposal_bank_frozen')<kinds.index('candidate_attempt')
    freeze=kinds.index('selection_frozen_before_replay')
    assert len([e for e in events[:freeze] if e['kind']=='candidate_return'])==64
    assert events[freeze+1]['receipt']['phase']=='independent_winner_replay'


def test_original_nonzero_history_and_no_second_scale():
    b,s,r,_=execute();raw=np.asarray(b['observed_raw_history'],np.float32)
    assert np.array_equal(raw[:5],m.prior.planner.fixed_raw()[5:])
    assert np.array_equal(raw[5:],b['prefix_commands'])
    expected=m.prior.core.normalized_tokens(b,raw)
    assert np.array_equal(b['observed_context']['action_history'],expected)
    ae=load(s,r['module_ledger'][0]['inputs'][0])[0]
    assert np.array_equal(ae[:2],expected)


@pytest.mark.parametrize('bad',['cost','raw','proposal','mean','std','elite','selected','replay','history','denominator'])
def test_full_candidate_update_release_tampering_rejected(bad):
    b,_,r,_=execute()
    if bad=='cost':r['evaluations'][0]['cost']+=1
    elif bad=='raw':r['evaluations'][0]['commands'][0][0]+=.01
    elif bad=='proposal':r['iterations'][0]['params'][0][0]+=.01
    elif bad=='mean':r['iterations'][1]['mean_before'][0]+=.01
    elif bad=='std':r['iterations'][0]['std_after'][0]+=.01
    elif bad=='elite':r['iterations'][0]['elite'].reverse()
    elif bad=='selected':r['selected_index']=(r['selected_index']+1)%64
    elif bad=='replay':r['evaluations'][-1]['context']['latent_history'][-1][0]+=.01
    elif bad=='history':r['evaluations'][0]['context']['action_history'][0][0]+=.01
    else:r['evaluations'].pop()
    with pytest.raises(ValueError):m.validate_row(b,r)


@pytest.mark.parametrize('stage',[0,1,15,16,63,64])
def test_failed_predict_preserves_every_attempt_and_partial_array(stage):
    b=h.branch();s=RAM();meter=NS(stage=None,expected_input=None,last_native=None,require_native=True)
    modules=m.Modules(NS(),s,'branch_00',meter);backend=h.Backend(b,modules,stage)
    row=dict(index=0,branch_id=b['branch_id'])
    with pytest.raises(RuntimeError,match='original forward failed'):m.evaluate(b,backend,meter,s,row)
    assert backend.calls==list(range(stage+1))
    assert len(row['evaluations'])==stage+1 and row['evaluations'][-1]['status']=='attempted'
    assert modules.entries[-1]['resource']=='predict' and not modules.entries[-1]['completed']
    assert len(modules.entries[-1]['inputs'])==2 and modules.failed
    assert modules.counts()['predict']==dict(attempted=stage+1,completed=stage)
    assert modules.counts()['action_encoder']==dict(attempted=stage+1,completed=stage+1)
    assert s.files and len([e for e in s.events if e['kind']=='candidate_return'])==stage


@pytest.mark.parametrize('target',['context','action','bridge','output','stage'])
def test_saved_native_array_bridge_rejects_mutation(target):
    b,s,r,_=execute()
    if target=='stage':r['module_ledger'][1]['stage']=1
    else:
        entry=r['module_ledger'][0]['inputs'][0] if target=='action' else r['module_ledger'][1]['inputs'][0 if target=='context' else 1]
        if target=='output':entry=r['module_ledger'][2]['output']
        a=load(s,entry).copy()
        if target=='output':a[-1,0]+=.01  # Returned successor is last projected row, not earlier history outputs.
        else:a.flat[0]+=.01
        new=m.prior.save_array(s,'tampered.npy',a);entry.clear();entry.update(new)
    with pytest.raises(ValueError):m.native_arrays(b,r,lambda x:load(s,x))


def test_projection_is_declared_proposal_change_not_dispatch_clip():
    cap=m.contract()['packet_logit_cap'];a=np.full((16,10),100.)
    out=m.project(a,cap)
    assert np.allclose(np.linalg.norm(out,axis=1),cap)
    assert not m.contract()['projection_density_correction']
    assert len(set(m.contract()['seeds']))==4


def test_stable_ties_are_oldest_candidate_not_last_or_best_replay():
    b=h.branch();s=RAM();meter=NS(stage=None,expected_input=None,last_native=None,require_native=False)
    class Constant:
        def step(self,context,raw):
            incoming=m.prior.planner.incoming(b,context,raw)
            return dict(context,latent_history=deepcopy(context['latent_history'][1:])+[context['latent_history'][-1]],action_history=incoming['action_history'][-2:])
    row=dict(branch_id=b['branch_id']);m.evaluate(b,Constant(),meter,s,row);m.validate_row(b,row)
    assert row['selected_index']==0


def test_invalid_prepare_no_output_before_gate(monkeypatch):
    def bad(*a):raise ValueError('external prepare SHA mismatch')
    monkeypatch.setattr(m,'verify',bad)
    monkeypatch.setattr(m,'Store',lambda *a:pytest.fail('output before verify'))
    with pytest.raises(ValueError,match='SHA mismatch'):m.run('/RAM/p','a'*64,'/RAM/o',m.AUTH)


def test_MAIN_authorization_precedes_any_preparation(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verification before authorization'))
    with pytest.raises(ValueError,match='MAIN'):m.run('/RAM/p','a'*64,'/RAM/o','wrong')


def test_cli_help_no_producer_or_model(capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    assert e.value.code==0 and 'coverage' in capsys.readouterr().out


def test_known_task_not_same_compute_or_official_paper():
    c=m.contract()
    assert c['total']['predict']==260 and c['per_branch']['predict']==65
    assert c['old_suffix_evaluations']==c['nominal_evaluations']==c['padding']==c['AD']==c['SVD']==c['new_physics']==0
    assert c['P1_admission'] is c['equal_total_compute'] is c['official_reproduction'] is False
