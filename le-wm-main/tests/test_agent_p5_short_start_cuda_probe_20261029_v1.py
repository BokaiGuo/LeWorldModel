"""Synthetic descriptors/arrays only; real Store/Trace/journal, no torch/model."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_short_start_cuda_probe_20261029_v1 as m
from test_agent_p5_short_start_native_abi_20261029_v1 import fixture as original_fixture


class Tensor:
    def __init__(self,a):
        self.value=a;self.shape=a.shape;self.device='cuda:0';self.dtype='float32';self.requires_grad=False;self.is_cuda=True
    def detach(self):return self
    def contiguous(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


class Module:
    def __init__(self):self.pre=[];self.post=[];self.fail=False;self.bad=False
    def register_forward_pre_hook(self,h,with_kwargs):
        self.pre.append(h);return NS(remove=lambda:self.pre.remove(h))
    def register_forward_hook(self,h,with_kwargs):
        self.post.append(h);return NS(remove=lambda:self.post.remove(h))
    def __call__(self,*args):
        for h in self.pre:h(self,args,{})
        if self.fail:raise RuntimeError('fixture native dispatch failed')
        v,u,p=(x.value for x in args);t=u.shape[1]
        out=(Tensor(v.reshape(1,t,256,384)+np.float32(.25)),None,Tensor(p+np.float32(.5)))
        if self.bad:out[0].device='cpu'
        for h in self.post:h(self,args,{},out)
        return out


class FixtureBackend:
    """No real CUDA. Fake tensors exercise actual new hook and Store code."""
    fail_stage=None;bad_stage=None
    def __init__(self,manifest,trace):
        self.trace=trace;self.before=manifest['record']['expected_model'];self.strict=manifest['record']['expected_strict_loads']
        self.sources={};self.attention_ops={};self.models={k:Module() for k in m.codec.KEYS}
        torch=NS(is_tensor=lambda x:isinstance(x,Tensor),float32='float32',isfinite=lambda x:np.isfinite(x.value),
            is_inference_mode_enabled=lambda:True,backends=NS(cuda=NS(matmul=NS(allow_tf32=False)),cudnn=NS(allow_tf32=False)))
        trace.install(self.models,torch)
    def predict(self,args):
        model=self.models['predictor'];stage=len(self.trace.calls)+1
        model.fail=stage==self.fail_stage;model.bad=stage==self.bad_stage
        model(*(Tensor(x) for x in args))
    def receipt(self):return self.before
    def memory(self):return {'fixture':True}
    def check_sources(self,manifest):return None


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'inner';(root/'results').mkdir(parents=True)
    source=root/'saved';source.mkdir();r,p,c=original_fixture(source)
    record=m.abi.resolve_actual0(r,p,c,source,0);args,extraction=m.abi.materialize_actual0(record)
    manifest=dict(record=record,extraction=extraction,bindings={str(q):m.sha(q) for q in source.iterdir()})
    monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.setattr(m,'verify',lambda prepared,expected:manifest)
    return NS(root=root,m=manifest,args=args,out=root/'results'/(m.NAME+'_fixture'),prepared=root/'results'/'fixture.json')


def perform(case,backend=FixtureBackend):
    v=m.execute(case.m,case.prepared,'a'*64,case.out,backend_factory=backend)
    return v,m.read(case.out/'report.json',v['sha256'])


def test_real_store_hooks_journal_full_IO_and_causal_consume(case):
    v,r=perform(case)
    assert r['status']=='branch0_T1_T2_native_ABI_recorded'
    assert r['tensor_bytes']==2457720 and len(r['calls'])==2
    assert len(list(case.out.glob('tensor_*.npy')))==10
    events=list(m.codec.journal(case.out/'journal.jsonl'))
    assert len(events)==13 and b'\n\n' not in (case.out/'journal.jsonl').read_bytes()
    assert events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished'
    receipt=m.consume(case.out,v['sha256'])
    assert receipt['status']=='branch0_T1_T2_saved_native_ABI_verified' and not receipt['P1_admission']
    c=r['calls'][1];actual,kw=m.codec.restore(c['inputs'],lambda rec:m.abi.load_array(dict(directory=str(case.out),array=rec)))
    assert not kw and m.exact(actual[0][:,:1],case.args[0]) and m.exact(actual[2][:,:1],case.args[2])
    original=m.abi.load_array(case.m['record']['refs']['visual'])
    assert not m.exact(actual[0][:,1:2],original[:,1:2])


@pytest.mark.parametrize('mode',('dispatch','badreturn'))
def test_failure_keeps_full_denominator_and_observed_vs_validated_returns(case,mode):
    class Broken(FixtureBackend):
        fail_stage=2 if mode=='dispatch' else None
        bad_stage=2 if mode=='badreturn' else None
    v,r=perform(case,Broken)
    assert r['status']=='partial_or_failed' and len(r['stages'])==2 and r['stages'][1]['prediction'] is None
    assert r['counts']['attempted']['api']==2 and r['counts']['validated']['api']==1
    assert r['counts']['returned']['api']==(1 if mode=='dispatch' else 2)
    assert m.consume(case.out,v['sha256'])['status']=='blocked_partial_probe'
    assert not (case.out/'prediction_seal.json').exists()


def test_second_stage_uses_both_new_predictions_and_original_packets(case):
    v,a,p=case.args;out=(v.reshape(1,1,256,384)+np.float32(9),None,p+np.float32(7))
    next_v,next_a,next_p=m.second_inputs(case.m,case.args,out)
    assert m.exact(next_v[:,1:].reshape(out[0].shape),out[0]) and m.exact(next_p[:,1:],out[2])
    assert m.exact(next_a,m.abi.load_array(case.m['record']['refs']['action'])[:,:2])
    assert m.exact(v,next_v[:,:1]) and m.exact(p,next_p[:,:1])


@pytest.mark.parametrize('bad',('CPU','full3','encoder','dtype','grad'))
def test_actual_hook_rejects_invalid_native_dispatch(case,bad):
    store=m.Store(case.out);trace=m.Trace(store);b=FixtureBackend(case.m,trace)
    args=[Tensor(x) for x in case.args];model=b.models['predictor']
    if bad=='CPU':args[0].device='cpu'
    elif bad=='full3':args[0].shape=(1,3,1,16,16,384)
    elif bad=='encoder':model=b.models['official_encoder']
    elif bad=='dtype':args[0].dtype='float64'
    else:args[0].requires_grad=True
    with pytest.raises(ValueError):model(*args)
    assert trace.guard.attempted['api']==0
    assert not trace.close()


def test_two_exact_newline_store_events_roundtrip_and_hashchain(case):
    store=m.Store(case.out);store.emit('one',value=1);store.emit('two',value=2)
    raw=(case.out/'journal.jsonl').read_bytes();assert raw.count(b'\n')==2 and not raw.endswith(b'\n\n')
    assert [e['kind'] for e in m.codec.journal(case.out/'journal.jsonl')]==['one','two']
    (case.out/'journal.jsonl').write_bytes(raw.replace(b'"value":1',b'"value":2',1))
    with pytest.raises(ValueError):list(m.codec.journal(case.out/'journal.jsonl'))


@pytest.mark.parametrize('fault',('artifact','journal','extra','model','stage','counts','metadata','causal','event','sources','fused','seal'))
def test_own_consumer_rejects_tampering(case,fault):
    v,r=perform(case)
    if fault=='artifact':
        q=case.out/'tensor_000.npy';q.write_bytes(q.read_bytes()+b'\n')
    elif fault=='journal':
        q=case.out/'journal.jsonl';q.write_bytes(q.read_bytes()+b'\n')
    elif fault=='extra':(case.out/'unpaid.npy').write_bytes(b'wrong')
    elif fault=='model':r['model_after']={}
    elif fault=='stage':r['stages'][1]['frames']=[5,10]
    elif fault=='counts':r['counts']['attempted']['api']=3
    elif fault=='metadata':r['calls'][1]['native_outputs'][0]['device']='cpu'
    elif fault=='causal':r['calls'][1]['inputs']=r['calls'][0]['inputs']
    elif fault=='event':r['calls'][0]['units']['api']=2
    elif fault=='sources':r['imported_sources']={'wrong':'/missing_source'}
    elif fault=='fused':r['attention_ops']={'aten._scaled_dot_product_flash_attention.default':1}
    else:r['prediction_seal']['sha256']='0'*64
    (case.out/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,FileNotFoundError)):m.consume(case.out,m.sha(case.out/'report.json'))


def test_store_exclusive_paths_and_array_bound(case):
    s=m.Store(case.out)
    with pytest.raises(ValueError):m.Store(case.out)
    with pytest.raises(ValueError):s.put('../escape',b'no')
    with pytest.raises(ValueError):s.array(np.zeros(1,np.float64))
    with pytest.raises(ValueError):s.array(np.full(1,np.nan,np.float32))
    with pytest.raises(ValueError):s.array(np.zeros(m.TENSOR_CAP//4+1,np.float32))
    s.put('single',b'1')
    with pytest.raises(ValueError):s.put('single',b'2')


def test_conflicting_binding_rejected():
    bindings={'a':'1'};m.merge(bindings,{'a':'1','b':'2'})
    with pytest.raises(ValueError):m.merge(bindings,{'a':'2'})


def test_run_clock_starts_before_verify(case,monkeypatch):
    monkeypatch.delenv('PYTHONPATH',raising=False);seen=[]
    monkeypatch.setattr(m.time,'monotonic',lambda:123.)
    monkeypatch.setattr(m,'verify',lambda *args:seen.append('verify') or case.m)
    monkeypatch.setattr(m,'execute',lambda *args,**kw:seen.append(kw['started']) or {'status':'fixture'})
    m.run('unused','0'*64,'unused',m.AUTH)
    assert seen==['verify',123.]
    with pytest.raises(ValueError):m.run('unused','0'*64,'unused','wrong')


def test_import_and_CLI_without_neural_or_physics():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,check=True)
    r=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert '{prepare,verify,run,consume}' in r.stdout


def test_actual_catalog_constants_and_no_science_claim():
    assert m.CATALOG_SHA=='1c49b5eba91619a6798a9d387c524b6e7d8658c196a9ac75f1ea9a65226c2678'
    assert len(m.ACTUAL0_SHA)==3 and all(len(h)==64 for h in m.ACTUAL0_SHA)
    p=m.protocol();assert p['new_labels']==p['new_physics']==p['new_encoder']==0
    assert p['caps']['predictor']==dict(api=2,batch_samples=2,frame_samples=3) and not p['P1_admission']
