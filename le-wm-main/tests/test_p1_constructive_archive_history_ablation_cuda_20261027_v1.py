"""RAM contract tests only, never instantiate CPU/CUDA neural or simulator."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_archive_history_ablation_cuda_20261027_v1 as m


def data():
    return ([[np.full(192,100*i+j,np.float32) for j in range(5)] for i in range(4)],
        [np.arange(40,dtype=np.float32).reshape(20,2)/10+i for i in range(4)])


def test_all32_and96tokens_teacher_forcing_exact_intervention_positions():
    zs,raw=data(); before=deepcopy((zs,raw)); reqs,labels=m.make_requests(zs,raw)
    assert len(reqs)==32 and len(labels)==8 and sum(len(r['tokens']) for r in reqs)==96
    for i in range(8):
        f,s,a,sa=reqs[4*i:4*i+4]
        assert [r['condition'] for r in (f,s,a,sa)]==list(m.CONDITIONS)
        slot=i//2; start=i%2
        assert np.array_equal(f['latent_history'],np.stack(zs[slot][start:start+3]))
        assert np.array_equal(f['raw_packets'],raw[slot][start*5:(start+3)*5].reshape(3,5,2))
        for r in (f,s,a,sa):
            assert set(r)==m.frozen.REQUEST_KEYS
            assert m.bits(r['latent_history'][-1],f['latent_history'][-1])
            assert m.bits(r['raw_packets'][-1],f['raw_packets'][-1])
            assert m.bits(r['tokens'][-1],f['tokens'][-1])
        assert m.bits(f['latent_history'],a['latent_history'])
        assert m.bits(f['raw_packets'],s['raw_packets']) and m.bits(f['tokens'],s['tokens'])
        assert np.all(s['latent_history']==s['latent_history'][-1])
        assert np.all(sa['latent_history']==sa['latent_history'][-1])
        assert not a['raw_packets'][:2].any() and not sa['raw_packets'][:2].any()
        assert a['tokens'][:2].any()  # normalized raw-zero != normalized-zero
        assert m.bits(a['tokens'],m.frozen.normalized(a['raw_packets']))
        assert m.bits(labels[i]['target'],zs[slot][start+3])
    for xs,ys in zip(before,(zs,raw),strict=True):
        for x,y in zip(xs,ys,strict=True): assert np.array_equal(x,y)


def test_targets_only_evaluator_and_z20_never_input():
    zs,raw=data(); reqs,labels=m.make_requests(zs,raw)
    labels[0]['target'][:]=-999
    assert reqs[0]['latent_history'][-1,0]==2
    assert reqs[4]['latent_history'][-1,0]==3
    assert all(not np.any(r['latent_history']==4) for r in reqs[:8])


def test_frozen_F_inputs_exactly_match_v2_builder():
    zs,raw=data(); original,_=m.frozen.requests(zs,[dict(actions=a) for a in raw],m.EPISODES)
    ours,_=m.make_requests(zs,raw)
    for i in range(8):
        old,new=original[3*i],ours[4*i]
        for key in ('tokens','raw_packets','latent_history'): assert m.bits(old[key],new[key])


@pytest.mark.parametrize('kind',m.CONDITIONS)
def test_all_conditions_route_original_F_numerical_forward_no_embedding_mask(monkeypatch,kind):
    reqs,_=m.make_requests(*data()); req=next(r for r in reqs if r['condition']==kind); calls=[]
    def frozen_forward(model,torch,device,r): calls.append(deepcopy(r)); return 'native-F'
    monkeypatch.setattr(m.frozen,'native_predict',frozen_forward)
    assert m.native_predict(None,None,'cuda:0',req)=='native-F'
    assert calls[0]['condition']=='F' and req['condition']==kind
    for key in ('tokens','raw_packets','latent_history'): assert m.bits(calls[0][key],req[key])


def backend(r):
    assert set(r)==m.frozen.REQUEST_KEYS
    embedding=np.ones((3,192),np.float32)
    return dict(prediction=r['latent_history'].copy(),embedding_before=embedding,embedding_after=embedding.copy())


class RAMStore:
    def __init__(self): self.files={}; self.events=[]
    def _write(self,name,raw): assert name not in self.files; self.files[name]=raw
    def event(self,e): self.events.append(deepcopy(e))


@pytest.fixture
def ram(monkeypatch):
    s=RAMStore()
    def save(store,name,a):
        b=BytesIO(); np.save(b,a,allow_pickle=False); store._write(name,b.getvalue())
        return dict(file=name,**m.frozen.array_receipt(a))
    monkeypatch.setattr(m.frozen,'save_array',save); return s


def setup_requests():
    reqs,labels=m.make_requests(*data()); refs=[backend(reqs[i]) for i in range(0,32,4)]
    return reqs,labels,refs


def test_all32_durable_and_exact8_F_before_scoring(ram):
    reqs,labels,refs=setup_requests(); rows=[dict(status='not_started') for _ in reqs]
    outputs=m.execute(reqs,refs,backend,ram,rows,m.frozen.Limits())
    assert len(outputs)==32 and len(ram.files)==32*7
    assert all(r['status']=='saved' for r in rows)
    assert sum('F_exact' in r and all(r['F_exact'].values()) for r in rows)==8
    scored=m.score(outputs,labels)
    assert scored['window_denominator']==8 and scored['episode_denominator']==4
    assert scored['mean_fidelity_mse']==dict(F=1.,S=1.,A=1.,SA=1.)
    assert all(r['interaction_SA_minus_S_minus_A_plus_F']==0. for r in scored['windows'])
    assert not scored['physical_counterfactual_truth'] and not scored['P1_admission']


@pytest.mark.parametrize('fault',['one_ulp','signed_zero','embedding'])
def test_F_no_allclose_fallback_partial_saved_and_no_retry(ram,fault):
    reqs,labels,refs=setup_requests(); rows=[dict(status='not_started') for _ in reqs]; calls=[]
    def broken(r):
        calls.append(r); out=backend(r)
        if fault=='one_ulp': out['prediction'][0,0]=np.nextafter(out['prediction'][0,0],np.float32(1))
        if fault=='signed_zero': out['prediction'][0,0]=-np.float32(0)
        if fault=='embedding':
            out['embedding_before'][0,0]+=1; out['embedding_after'][0,0]+=1
        return out
    with pytest.raises(ValueError,match='bitwise'): m.execute(reqs,refs,broken,ram,rows,m.frozen.Limits())
    assert len(calls)==1 and rows[0]['status']=='saved_factual_reproduction_failed'
    assert all(r['status']=='not_started' for r in rows[1:])
    assert 'request_00.prediction.npy' in ram.files


def test_label_or_reference_cannot_reach_callback(ram):
    reqs,labels,refs=setup_requests(); reqs[0]['target']=labels[0]['target']; calls=[]
    with pytest.raises(ValueError): m.execute(reqs,refs,lambda r:calls.append(r),ram,[{}]*32,m.frozen.Limits())
    assert not calls and not ram.files


def simulate(l):
    l.dispatch('action_encoder',lambda:None)
    def predict(): l.dispatch('predictor',lambda:None); l.dispatch('pred_proj',lambda:None)
    l.dispatch('predict',predict)


def test_32_API_not64_and96token_budget():
    ledger=m.Ledger(lambda e:None,m.frozen.Limits())
    for _ in range(32): simulate(ledger)
    assert len(ledger.entries)==128
    assert all(v==dict(attempted=32,completed=32,attempted_tokens=96,completed_tokens=96) for v in ledger.counts().values())
    with pytest.raises(ValueError): simulate(ledger)


def test_failed_ledger_and_journal_not_refunded():
    calls=[]
    def sink(e): calls.append(e); raise RuntimeError('journal mock')
    l=m.Ledger(sink,m.frozen.Limits()); dispatch=[]
    with pytest.raises(RuntimeError): l.dispatch('action_encoder',lambda:dispatch.append(1))
    assert not dispatch and l.failed and l.counts()['action_encoder']['attempted']==1
    with pytest.raises(ValueError): simulate(l)


@pytest.mark.parametrize('bad',['wall','memory','disk'])
def test_pre_dispatch_caps(bad):
    clock=[0]; limit=m.frozen.Limits(clock=lambda:clock[0]); called=[]
    if bad=='wall': clock[0]=600
    if bad=='memory': limit.memory=lambda:(m.contract()['allocated_bytes']+1,0)
    if bad=='disk': limit.free=lambda:m.FLOOR-1
    l=m.Ledger(lambda e:None,limit)
    with pytest.raises(ValueError): l.dispatch('action_encoder',lambda:called.append(1))
    assert not called and not l.entries


def test_source_fixed_SHA_rejects_before_saved_neural(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda p:(_ for _ in ()).throw(AssertionError('must not read')))
    with pytest.raises(ValueError): m.load_source('RAM','0'*64)


def test_array_file_and_entry_hash_drift_in_ram(monkeypatch):
    a=np.arange(192,dtype=np.float32); s=m.Saved('/RAM',{'z.npy':'expected'})
    b=BytesIO(); np.save(b,a,allow_pickle=False); monkeypatch.setattr(s,'raw',lambda name:b.getvalue())
    rec=dict(file='z.npy',sha256='expected',raw_sha256=m.enc.digest(a.tobytes()),shape=[192],dtype='<f4')
    assert m.bits(s.array('z.npy',(192,),rec),a)
    rec['raw_sha256']='0'*64
    with pytest.raises(ValueError): s.array('z.npy',(192,),rec)


def test_initial_real_disk_and_primary_error_preserved_no_output(monkeypatch):
    reports=[]; probes=[]
    class Output:
        def iterdir(self): return []
    class Broken:
        output=Output()
        def __init__(self,*args): pass
        def put(self,name,value,final=False):
            if name=='report.json': reports.append(deepcopy(value))
            else: raise RuntimeError('FIRST binding mock')
        def event(self,e): raise RuntimeError('SECOND journal mock')
        def close(self): raise RuntimeError('THIRD close mock')
    p=dict(contract=m.contract(),source_report='source',source_sha256=m.SOURCE_SHA)
    monkeypatch.setattr(m,'verify',lambda *a:(p,{}))
    monkeypatch.setattr(m,'Store',Broken)
    def disk_usage(path): probes.append(Path(path)); return NS(free=m.FLOOR+1024**3)
    monkeypatch.setattr(m.shutil,'disk_usage',disk_usage)
    with pytest.raises(ValueError): m.run('RAM','sha',str(m.ROOT/'results'/'not_created_history'))
    assert probes and all(p==m.ROOT/'results' for p in probes)
    r=reports[0]; assert r['first_error']['message']=='FIRST binding mock'
    assert [e['stage'] for e in r['errors']]==['execution','error_journal','journal_finalize']
    assert r['limits']['minimum_observed_free_bytes']==m.FLOOR+1024**3
    assert len(r['requests'])==32 and r['F_exact_windows']==0 and r['F_expected_windows']==8


def test_original_source_frozen_and_no_top_level_torch():
    import ast
    assert m.sha(m.frozen.__file__)==m.SOURCE_CODE_SHA
    tree=ast.parse(Path(m.__file__).read_text())
    assert not any('torch' in ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
    c=m.contract(); assert c['counts']['encoder']==c['counts']['projector']==0
    assert c['H1_calls']==32 and c['tokens']['latent_input']==96 and not c['counterfactual_physics']
    assert not c['selection'] and not c['training'] and not c['P1_admission']
