"""Array/metadata tests only: no CUDA/model/simulator execution."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p1_observed_direct_cuda_20261027_v1 as m

PACK=m.ROOT/'results/p1_observed_history_source_20261027_v1_main01'
PREP=m.ROOT/'results/p1_observed_history_source_20261027_v1_prepare01.json'
PACK_SHA='0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546'
PREP_SHA='ecc5f4a488290b6d0eade4d3ff2034d256729de94c4d3a14b12d6eebad783385'


@pytest.mark.parametrize('method',('full_H2_GN','full_H2_gradient'))
def test_fixed_budget(method):
    c=m.config(method);b=m.screen.budget(c)
    assert (c['B1'],c['B2'],len(c['initial_logits']),c['extra_line_search_trials'])==(4,8,1,0)
    assert (b['predict'],b['action_encoder'],b['jacobian'],b['vjp_directions'])==(70,70,8,1536)
    assert c['damping']==.002 and np.array_equal(c['initial_logits'],np.zeros((1,2,10)))


def test_actual_source_audit_without_neural():
    material=m.material(PACK,PACK_SHA,PREP,PREP_SHA)
    s=material['summary']
    assert s['validated_rows']==2 and s['actual_acquisition_counts']['physics_step']['completed']==208
    assert s['novel_root_count']==0 and s['independent_sample_increment']==0
    assert all(x['root_equals_parent'] and x['rgb_at_0_5_10_identical'] and x['state_at_0_5_10_identical'] for x in s['roots'])


@pytest.mark.parametrize('defect',['terminal','rows','counts','checkpoint','state','raw_step','flags','metadata'])
def test_source_receipt_corruption_fails_before_encoder(defect):
    p=m.read(PREP);r=m.read(PACK/'sourcepack.json');events=m.observed.source.read_journal(PACK/'journal.jsonl')
    if defect=='terminal':events[-1]['kind']='not_finished'
    elif defect=='rows':r['rows']=r['rows'][:1]
    elif defect=='counts':r['counts']['physics_step']['completed']=200
    elif defect=='checkpoint':r['rows'][0]['checkpoints'][1]['sha256']='0'*64
    elif defect=='state':next(e for e in events if e['kind']=='operation_completed' and e['resource']=='public_step')['receipt']['state'][0]+=1
    elif defect=='raw_step':next(e for e in events if e['kind']=='operation_started' and e['resource']=='public_step')['raw_action'][0]=1
    elif defect=='flags':next(e for e in events if e['kind']=='operation_completed' and e['resource']=='public_step')['receipt']['terminated']=True
    else:r['rows'][0]['metadata']['root_sha256']='0'*64
    with pytest.raises(ValueError):m.audit_source(r,p,events,PACK)


def test_bridge_true_observed_history_and_no_privileged_encoder_input():
    material=m.material(PACK,PACK_SHA,PREP,PREP_SHA);packet=material['packets'][0]
    template=m.template(material,'a'*64,'b'*64,'c'*64)
    calls=[]
    def encode(rgb):
        assert rgb.dtype==np.uint8 and rgb.shape==(224,224,3)
        calls.append(rgb.copy());return np.full(192,len(calls),np.float32)
    encoded=m.encode_observed(packet,template,encode)
    assert len(calls)==4 and np.array_equal(encoded['task']['root_context']['latent_history'],np.tile([[1.],[2.],[3.]],(1,192)))
    assert encoded['task']['root_context']['action_history']==packet['normalized_action_history'].tolist()
    assert encoded['task']['goal']==[4.]*192 and not encoded['P1_admission']
    assert encoded['task_id']==m.digest(encoded['task'])


def test_bad_bridge_template_rejected_before_encoding():
    material=m.material(PACK,PACK_SHA,PREP,PREP_SHA);t=m.template(material,'a'*64,'b'*64,'c'*64);t['privileged_pose']=[]
    with pytest.raises(ValueError):m.encode_observed(material['packets'][0],t,lambda _:pytest.fail('encoder'))


def test_space_gate_before_any_cuda_import_or_output(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'verify',lambda *a:{})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP-1))
    with pytest.raises(ValueError,match='256MiB'):m.run(tmp_path/'p','a'*64,tmp_path/'out')
    assert not list(tmp_path.iterdir())


def test_shared_lossless_journal_cap_floor_and_exclusive(tmp_path,monkeypatch):
    directory=tmp_path/'root_0';sink=m.SharedJournal(tmp_path,directory)
    value=dict(kind='raw_derivative',row=list(range(20)));sink(value)
    assert m.read_lines(directory/'direct_journal.jsonl')==[value]
    with pytest.raises(FileExistsError):m.SharedJournal(tmp_path,directory)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError,match='floor'):sink(value)


def test_total_output_cap_checked_before_journal_append(tmp_path,monkeypatch):
    sink=m.SharedJournal(tmp_path,tmp_path/'component')
    monkeypatch.setattr(m,'used_bytes',lambda _:m.CAP)
    with pytest.raises(ValueError,match='56MiB'):sink(dict(kind='do_not_write'))
    assert sink.path.stat().st_size==0


def test_native_angle_risk_diagnostic_not_silently_wrapped():
    state=np.zeros(7);goal=np.zeros(7);state[4]=5;goal[4]=-2
    d=m.angle_diagnostic(state,goal)
    assert d['native_terminal'] and not d['wrapped_terminal'] and d['discrepancy']
    assert d['native_angle']<0<d['wrapped_angle']


def test_no_old_padding_loader_or_environment_dispatch_in_new_driver():
    import inspect
    code=inspect.getsource(m)
    assert 'load_raw_source(' not in code and 'encode_source(' not in code
    assert 'gym.make(' not in code and 'env.reset(' not in code and 'env.step(' not in code


def test_small_actual_prepare_and_verify_without_neural(tmp_path):
    path=tmp_path/'small_prepare.json'
    value=m.prepare(path,pack=PACK,source_sha=PACK_SHA,source_prepare=PREP,source_prepare_sha=PREP_SHA)
    assert path.stat().st_size<128*1024
    assert list(tmp_path.iterdir())==[path]  # No copied seal/source/weights.
    assert value['direct_resources']==m.TOTAL and value['independent_sample_increment']==0
    assert m.verify(path,m.sha(path))==value
    with pytest.raises(ValueError):m.verify(path,'0'*64)


class AnalyticBackend:
    """Pure array autoregression; not a real model or authenticated CUDA run."""
    @contextmanager
    def guard(self,meter):yield
    def rollout(self,t,raw,meter,mode):
        previous=deepcopy(t['root_context']);result=[]
        for token in m.screen.core.normalized_tokens(t,raw):
            meter.call('action_encoder',lambda:None)
            value=meter.call('predict',lambda:.7*np.asarray(previous['latent_history'][-1])+np.pad(token,(0,182)))
            previous=dict(previous,latent_history=previous['latent_history'][1:]+[value.tolist()],
                          action_history=previous['action_history'][1:]+[token.tolist()])
            result.append(deepcopy(previous))
        return result
    def jacobian(self,t,inc,meter):
        terminal=self.rollout(t,inc['commands'],meter,'AD')[-1]['latent_history'][-1]
        b=np.zeros((192,10));b[:10]=np.eye(10)
        std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
        j=np.concatenate((.7*b/std,b/std),axis=1)
        for _ in range(192):meter.call('vjp_directions',lambda:None)
        return terminal,j*(1-np.asarray(inc['commands']).reshape(20)**2)


def test_four_B8_components_through_sealed_core_with_array_backend():
    material=m.material(PACK,PACK_SHA,PREP,PREP_SHA);total={k:0 for k in m.TOTAL};images=[]
    def encode(rgb):images.append(rgb);return np.zeros(192)
    for packet in material['packets']:
        t=m.encode_observed(packet,m.template(material,'a'*64,'b'*64,'c'*64),encode)['task']
        for method in m.METHODS:
            r=m.screen.run_direct(t,m.config(method),AnalyticBackend(),lambda e:None)
            assert r['status']=='direct_component_completed_not_admitted' and r['error'] is None
            assert len(r['direct_runs'][0])==9 and not r['P1_admission']
            for resource,cap in r['budget'].items():
                receipts=[e for e in r['extended_journal'] if e['resource']==resource]
                assert len(receipts)==cap and all(e['completed'] for e in receipts)
                total[resource]+=sum(e['samples'] for e in receipts)
    assert total==m.TOTAL and len(images)==8


def test_import_is_pure():
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(n in sys.modules for n in ("torch","gymnasium","pymunk","stable_worldmodel"))'],check=True)
