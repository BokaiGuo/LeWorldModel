"""Saved-array/metadata fixtures only: never a CPU neural forward."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p3p4_goal_patch_replay_20261027_v1 as m


@pytest.fixture(scope='module')
def actual():return m.validate_metric(m.PROJECT)


def test_actual_common_bank(actual):
    plans,p,bindings,cost=actual
    assert len(plans)==8 and len(bindings)==7
    assert [v['horizon'] for v in plans if v['arm']=='goal_patch_Q']==[2,3,3,2]
    assert sum(2*v['horizon'] for v in plans)==36
    assert cost['metric']['actual_encoder_calls']==80
    assert cost['planner']['attempted_predict_samples']==552
    assert all(v['selection_sha256']==m.digest({k:x for k,x in v.items() if k!='selection_sha256'}) for v in plans)


class TensorMetadata:
    def __init__(self,kind='predict',device='cuda'):
        self.device=SimpleNamespace(type=device);self.shape=(1,3,192 if kind=='predict' else 10)


def fake_model(fail=False):
    def callback(*a,**k):
        if fail:raise RuntimeError('fixture dispatch failed')
        return 'metadata-only'
    return SimpleNamespace(predict=callback,action_encoder=SimpleNamespace(forward=callback),encoder=SimpleNamespace(forward=callback))


@pytest.mark.parametrize('kind',['predict','action_encoder'])
def test_predispatch_path_cap(kind):
    p=dict(root_index=0,arm='old_Q',horizon=2);journal=[];model=fake_model()
    with m.ReplayCap(model,[p],journal.append) as guard:
        callback=model.predict if kind=='predict' else model.action_encoder.forward
        with guard.scope(p,'whole'):
            assert callback(TensorMetadata(kind))=='metadata-only'
            callback(TensorMetadata(kind))
            with pytest.raises(ValueError):callback(TensorMetadata(kind))
        assert len(guard.events)==2 and guard.blocked==1
    assert [e['completed'] for e in journal if e['kind']==kind]==[False,True,False,True]


def test_failed_attempt_not_refunded():
    p=dict(root_index=0,arm='old_Q',horizon=2);journal=[]
    with m.ReplayCap(fake_model(True),[p],journal.append) as guard:
        with guard.scope(p,'whole'):
            for _ in range(2):
                with pytest.raises(RuntimeError):guard.model.predict(TensorMetadata())
            with pytest.raises(ValueError):guard.model.predict(TensorMetadata())
    assert len(guard.events)==2 and not any(e['completed'] for e in guard.events)
    assert all('error' in e for e in guard.events)


def test_scope_device_and_visual_guard():
    p=dict(root_index=0,arm='old_Q',horizon=2);model=fake_model();original=model.predict
    with m.ReplayCap(model,[p],lambda e:None) as guard:
        with pytest.raises(ValueError):model.predict(TensorMetadata())
        with guard.scope(p,'whole'):
            with pytest.raises(ValueError):model.predict(TensorMetadata(device='cpu'))
            with pytest.raises(ValueError):model.encoder.forward(None)
        assert guard.events==[] and guard.forbidden==1
    assert model.predict is original


class ContextGuard:
    def scope(self,*args):return nullcontext()


def test_double_replay_no_context_alias(actual):
    p=actual[0][0];calls=[];journal=[]
    def replay(context,actions):
        calls.append((deepcopy(context),np.asarray(actions).shape))
        return deepcopy(p['context'])
    result=m.replay_plan(p,replay,ContextGuard(),journal.append)
    assert result['status']=='completed' and len(calls)==1+p['horizon']
    assert calls[0][0]==calls[1][0]==p['root_context']
    assert result['segments'][-1]==result['whole_context']==p['context']


@pytest.mark.parametrize('key',['context','old_Q_cost','goal_patch_Q_cost'])
def test_failed_witness_retained(actual,key):
    p=deepcopy(actual[0][0]);saved=deepcopy(p['context'])
    if key=='context':p[key]['latent_history'][0][0]+=1
    else:p[key]+=1
    row=m.replay_plan(p,lambda c,a:deepcopy(saved),ContextGuard(),lambda e:None)
    assert row['status']=='failed' and 'whole_context' in row and 'segments' in row


def fixture_ledger(plans):
    events=[];rows=[];journal=[]
    for p in plans:
        rows.append(m.replay_plan(p,lambda c,a:deepcopy(p['context']),ContextGuard(),lambda e:None))
        for path in ('whole','segmented'):
            for _ in range(p['horizon']):
                for kind in ('action_encoder','predict'):
                    e=dict(kind=kind,index=len(events),root_index=p['root_index'],arm=p['arm'],path=path,context_tokens=3,completed=True)
                    events.append(e);journal.extend([dict(e,completed=False),deepcopy(e)])
    return events,rows,journal


def test_exact_ledger(actual):
    e,r,j=fixture_ledger(actual[0]);m.validate_ledger(e,actual[0],r,j)


@pytest.mark.parametrize('mutation',['missing_event','missing_arm','wrong_root','wrong_horizon','partial','wrong_Q','missing_journal','visual'])
def test_ledger_rejects(actual,mutation):
    e,r,j=fixture_ledger(actual[0])
    if mutation=='missing_event':e.pop()
    elif mutation=='missing_arm':r.pop()
    elif mutation=='wrong_root':r[0]['root_sha256']='wrong'
    elif mutation=='wrong_horizon':r[0]['horizon']=3
    elif mutation=='partial':r[0]['status']='failed'
    elif mutation=='wrong_Q':r[0]['goal_patch_Q_cost']+=1
    elif mutation=='missing_journal':j.pop()
    elif mutation=='visual':j.append(dict(kind='forbidden_visual_encoder'))
    with pytest.raises(ValueError):m.validate_ledger(e,actual[0],r,j)


@pytest.mark.parametrize('target',['bank','projection','goal','horizon','root','journal'])
def test_source_mutation_rejected(actual,monkeypatch,target):
    original=m.read_json
    def altered(path):
        value=deepcopy(original(path))
        if str(path).endswith('report.json') and 'goal_patch_metric' in str(path):
            if target=='bank':value['rows'][0]['selection']['candidates'].pop()
            elif target=='projection':value['rows'][0]['metric']['projection'][0][0]+=1
            elif target=='goal':value['rows'][0]['goal_vectors'][0][0]+=1
            elif target=='horizon':value['rows'][0]['selection']['winners']['old_Q']['horizon']=3
            elif target=='root':value['rows'][0]['root_sha256']='wrong'
            elif target=='journal':value['compile_journal_sha256']='wrong'
        return value
    monkeypatch.setattr(m,'read_json',altered)
    with pytest.raises((ValueError,KeyError)):m.validate_metric(m.PROJECT)


def test_prepare_verify_exclusive_and_tamper(tmp_path,actual):
    out=tmp_path/'prepared';m.prepare(m.PROJECT,out);sha=m.io.sha((out/'prepare.json').read_bytes())
    p=m.validate_prepared(out,sha);assert p['plans']==actual[0]
    with pytest.raises(ValueError):m.prepare(m.PROJECT,out)
    with pytest.raises(ValueError):m.validate_prepared(out,'bad')
    # Fixture mutation only, never a sealed project asset.
    (out/'protocol.json').write_text('{}')
    with pytest.raises(ValueError):m.validate_prepared(out,sha)
