"""Small NumPy/RAM mocks only, no prepare, neural, physics or temporary files."""
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace as NS
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v1 as m


class Pixels:
    shape=(125,224,224,3);dtype=np.dtype('uint8')
    def __init__(self):self.requests=[]
    def __getitem__(self,indices):
        self.requests.append(list(indices));return np.asarray([np.full((224,224,3),i,np.uint8) for i in indices])


def h5():
    return dict(pixels=Pixels(),ep_offset=np.arange(5,dtype=np.int64)*25,ep_len=np.full(5,25,np.int32),
        action=np.linspace(-.4,.4,250,dtype=np.float32).reshape(125,2),
        episode_idx=np.repeat(np.arange(5),25),step_idx=np.tile(np.arange(25),5))


@pytest.mark.parametrize('i',m.EPISODES)
def test_precise_slices_episode_boundaries_and_receipts(i):
    f=h5();a,d=m.slice_episode(f,i);off=25*i
    assert f['pixels'].requests==[[off+s for s in m.STEPS]]
    assert a['offset']==off and a['length']==25 and a['next_offset']==off+25
    assert a['action_global_slice']==[off,off+20]
    assert [p['global_slice'] for p in a['packet_slices']]==[[off+s,off+s+5] for s in (0,5,10,15)]
    assert np.array_equal(d['actions'],f['action'][off:off+20])
    assert a['actions']==m.array_receipt(d['actions'])
    for row,frame in zip(a['frames'],d['frames']):assert row['png_sha256']==m.enc.digest(m.png_bytes(frame))


@pytest.mark.parametrize('bad',['short','offset','previous','episode','step','actionnan','actioncap','dtype','rows'])
def test_no_replacement_cross_episode_or_silent_action_repair(bad):
    f=h5()
    if bad=='short':f['ep_len'][1]=20
    elif bad=='offset':f['ep_offset'][2]+=1
    elif bad=='previous':f['ep_len'][0]-=1
    elif bad=='episode':f['episode_idx'][25+7]=2
    elif bad=='step':f['step_idx'][25+8]=9
    elif bad=='actionnan':f['action'][25+5,0]=np.nan
    elif bad=='actioncap':f['action'][25+5,0]=1.01
    elif bad=='dtype':f['action']=f['action'].astype(float)
    elif bad=='rows':f['action']=f['action'][:-1]
    with pytest.raises(ValueError):m.slice_episode(f,1)
    assert not f['pixels'].requests


def inputs():
    z=[np.full(192,i/8,np.float32) for i in range(5)]
    a=np.linspace(-.4,.4,40,dtype=np.float32).reshape(20,2)
    return z,a


class Backend:
    def __init__(self,ledger):self.ledger=ledger;self.calls=[]
    def step(self,c,a):
        self.calls.append((deepcopy(c),a.copy()))
        for key in ('action_encoder','predict'):self.ledger.end(self.ledger.begin(key))
        z=np.asarray(c['latent_history'][-1],np.float32)+np.float32(a.sum()/40)
        after=dict(c,latent_history=deepcopy(c['latent_history'][1:])+[z.astype(float).tolist()],
            action_history=[deepcopy(c['action_history'][-1]),m.base.d2p.normalized_history(a)[0].astype(float).tolist()])
        return after
    def replay(self,c,a):
        for packet in a.reshape(-1,5,2):c=self.step(c,packet)
        return c


def test_observed_context_and_fixed_five_calls_no_cached_H2_prefix():
    zs,a=inputs();events=[];l=m.PredictLedger(events.append);b=Backend(l)
    result=m.diagnose(zs,a,b,events.append)
    assert len(b.calls)==5
    assert [r['kind'] for r in result['outputs']]==['factual','zero','sign_flipped','factual']
    assert [r['horizon'] for r in result['outputs']]==[1,1,1,2]
    c=m.context(zs,a)
    assert c['latent_history']==np.asarray(zs[:3]).astype(float).tolist()
    assert c['action_history']==m.base.d2p.normalized_history(a[:10]).astype(float).tolist()
    assert all(b.calls[i][0]==c for i in (0,1,2,3))
    assert np.array_equal(b.calls[0][1],a[10:15]) and not b.calls[1][1].any()
    assert np.array_equal(b.calls[2][1],-a[10:15])
    assert np.array_equal(b.calls[3][1],a[10:15]) and np.array_equal(b.calls[4][1],a[15:20])
    assert b.calls[4][0]['latent_history'][-1]==result['outputs'][0]['context']['latent_history'][-1]
    assert l.counts()=={k:dict(attempted=5,completed=5) for k in ('predict','action_encoder')}


def test_targets_score_only_never_inputs_or_choice():
    z,a=inputs();b1=Backend(m.PredictLedger(lambda e:None));r1=m.diagnose(z,a,b1,lambda e:None)
    z[3]=np.full(192,100,np.float32);z[4]=np.full(192,-100,np.float32)
    b2=Backend(m.PredictLedger(lambda e:None));r2=m.diagnose(z,a,b2,lambda e:None)
    for (c,x),(d,y) in zip(b1.calls,b2.calls):assert c==d and np.array_equal(x,y)
    assert r1['outputs'][0]['fidelity_mse']!=r2['outputs'][0]['fidelity_mse']
    assert r1['selection'] is r2['selection'] is r1['P1_admission'] is False


def test_all_four_diagnostics_exact20_budget():
    l=m.PredictLedger(lambda e:None);b=Backend(l)
    for _ in range(4):m.diagnose(*inputs(),b,lambda e:None)
    assert l.counts()=={k:dict(attempted=20,completed=20) for k in ('predict','action_encoder')}
    with pytest.raises(ValueError,match='predispatch20'):l.begin('action_encoder')
    assert m.contract()['prediction_tokens']['total_latent_input_tokens']==60
    assert m.contract()['prediction_tokens']['total_action_packet_tokens']==60


def test_history_tamper_rejected():
    class Broken(Backend):
        def step(self,c,a):
            out=super().step(c,a);out['action_history'][0][0]+=1;return out
    with pytest.raises(ValueError,match='history'):m.diagnose(*inputs(),Broken(m.PredictLedger(lambda e:None)),lambda e:None)


def test_failed_call_not_refunded_or_retried():
    l=m.PredictLedger(lambda e:None)
    class Broken(Backend):
        def step(self,c,a):
            self.ledger.begin('action_encoder');raise RuntimeError('mock forward')
    with pytest.raises(RuntimeError):m.diagnose(*inputs(),Broken(l),lambda e:None)
    assert l.counts()['action_encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):l.begin('predict')


def test_no_AD_restore_even_after_failure():
    def original():return 1
    t=NS(autograd=NS(grad=original),Tensor=NS(),func=NS())
    t.autograd.functional=NS();events=[];errors=[]
    with pytest.raises(ValueError,match='prohibits AD'):
        with m.no_ad(t,events.append,errors):t.autograd.grad()
    assert t.autograd.grad is original and not errors and events==[dict(kind='forbidden_AD')]


def test_lossless_NPY_file_vs_raw_hash(monkeypatch):
    files={}
    s=NS(output=m.Path('/RAM'),_write=lambda name,b:files.__setitem__(name,b))
    monkeypatch.setattr(m,'sha',lambda path:m.enc.digest(files[m.Path(path).name]))
    z=inputs()[0][0];r=m.save_array(s,'z.npy',z)
    assert np.array_equal(np.load(BytesIO(files['z.npy']),allow_pickle=False),z)
    assert r['sha256']==m.enc.digest(files['z.npy']) and r['raw_sha256']==m.enc.digest(z.tobytes())
    assert r['sha256']!=r['raw_sha256']


def test_prepare_verify_rebuild_and_stat_tamper(monkeypatch):
    p=dict(file_version=dict(st_mtime_ns=1));monkeypatch.setattr(m,'sha',lambda p:'a'*64)
    monkeypatch.setattr(m,'read',lambda path:p);monkeypatch.setattr(m,'material',lambda:(deepcopy(p),[]))
    assert m.verify('/RAM','a'*64)==(p,[])
    with pytest.raises(ValueError):m.verify('/RAM','b'*64)
    monkeypatch.setattr(m,'material',lambda:(dict(file_version=dict(st_mtime_ns=2)),[]))
    with pytest.raises(ValueError):m.verify('/RAM','a'*64)


def test_scope_no_training_or_counterfactual_truth_claim():
    c=m.contract()
    assert c['episodes']==[0,1,2,3] and c['frames']==[0,5,10,15,20]
    assert c['encoder']==c['projector']==c['predict']==c['action_encoder']==20
    assert c['selection'] is c['training'] is c['normalizer']['fit'] is c['P1_admission'] is False
    assert c['normalizer']['training_provenance_verified'] is False
    assert c['AD']==c['physics']==0 and c['reserve_bytes']==200*1024**2


def test_native_dtype_required():
    z,a=inputs();z[0]=z[0].astype(float)
    with pytest.raises(ValueError):m.context(z,a)


def test_twenty_encodes_actual_dispatch_no_dedup(monkeypatch):
    payload=[m.slice_episode(h5(),i)[1] for i in m.EPISODES];files={};events=[]
    s=NS(output=m.Path('/RAM'),_write=lambda n,b:files.__setitem__(n,b),event=lambda e:events.append(deepcopy(e)))
    monkeypatch.setattr(m,'sha',lambda path:m.enc.digest(files[m.Path(path).name]))
    l=m.enc.Ledger(5,s.event);rows=[dict(index=i,status='not_started') for i in range(20)]
    tensor=lambda shape:NS(device=NS(type='cuda'),shape=shape)
    def cb(a):
        l.dispatch('encoder',tensor((1,3,224,224)),lambda:None)
        l.dispatch('projector',tensor((1,192)),lambda:None)
        return np.full(192,a[0,0,0]/255,np.float32)
    latents=m.encode_all(payload,cb,l,s,rows)
    assert len(latents)==4 and all(len(zs)==5 for zs in latents)
    assert l.counts()=={k:dict(attempted=20,completed=20) for k in ('encoder','projector')}
    assert len(files)==40 and all(r['status']=='encoded' for r in rows)
    for i,r in enumerate(rows):
        z=np.load(BytesIO(files[r['latent']['file']]),allow_pickle=False)
        assert np.array_equal(z,latents[i//5][i%5])


def test_encoder_partial_preserves_full_denominator(monkeypatch):
    payload=[m.slice_episode(h5(),i)[1] for i in m.EPISODES];files={}
    s=NS(output=m.Path('/RAM'),_write=lambda n,b:files.__setitem__(n,b),event=lambda e:None)
    monkeypatch.setattr(m,'sha',lambda path:m.enc.digest(files[m.Path(path).name]))
    l=m.enc.Ledger(5,s.event);rows=[dict(index=i,status='not_started') for i in range(20)]
    def cb(a):
        l.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        raise RuntimeError('mock projector failure')
    with pytest.raises(RuntimeError):m.encode_all(payload,cb,l,s,rows)
    assert len(rows)==20 and rows[0]['status']=='attempted' and rows[1]['status']=='not_started'
    assert l.counts()['encoder']==dict(attempted=1,completed=1) and l.counts()['projector']==dict(attempted=0,completed=0)
    assert len(files)==1
