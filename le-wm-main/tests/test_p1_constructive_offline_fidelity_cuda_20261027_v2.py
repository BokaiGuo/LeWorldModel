"""RAM-only synthetic contract tests: no Torch/model/environment, no temp files."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v2 as m


class Pixels:
    dtype=np.dtype('uint8')
    def __init__(self,n): self.shape=(n,224,224,3)
    def __getitem__(self,indices):
        return np.stack([np.full((224,224,3),i%251,np.uint8) for i in indices])


def dataset():
    n=7; length=25; rows=n*length
    return dict(pixels=Pixels(rows),action=np.linspace(-2.1,2.1,rows*2,dtype=np.float32).reshape(rows,2),
        ep_len=np.full(n,length,np.int64),ep_offset=np.arange(n,dtype=np.int64)*length,
        episode_idx=np.repeat(np.arange(n,dtype=np.int64),length),step_idx=np.tile(np.arange(length,dtype=np.int64),n))


def inputs():
    latents=[[np.full(192,100*e+j,np.float32) for j in range(5)] for e in range(4)]
    payload=[dict(actions=(np.arange(40,dtype=np.float32).reshape(20,2)+e)/10) for e in range(4)]
    return latents,payload


class RAMStore:
    def __init__(self): self.files={}; self.events=[]
    def _write(self,name,data):
        assert name not in self.files; self.files[name]=data
    def event(self,e): self.events.append(deepcopy(e))


@pytest.fixture
def ram(monkeypatch):
    store=RAMStore()
    def save(s,name,a):
        b=BytesIO(); np.save(b,a,allow_pickle=False); s._write(name,b.getvalue())
        return dict(file=name,**m.array_receipt(a))
    monkeypatch.setattr(m,'save_array',save)
    return store


@pytest.mark.parametrize('n,expected',[(4,(0,1,2,3)),(7,(0,2,4,6)),(18685,(0,6228,12456,18684))])
def test_indices(n,expected): assert m.episode_indices(n)==expected


@pytest.mark.parametrize('n',[0,3,True,4.0])
def test_indices_fail_closed(n):
    with pytest.raises(ValueError): m.episode_indices(n)


@pytest.mark.parametrize('index',[0,2,4,6])
def test_exact_slices_last_episode_and_unbounded_raw(index):
    f=dataset(); r,p=m.slice_episode(f,index)
    np.testing.assert_array_equal(p['actions'],f['action'][index*25:index*25+20])
    assert r['next_or_end_offset']==(index+1)*25
    assert r['frame_global_indices']==[index*25+s for s in m.STEPS]
    if index in (0,6): assert abs(p['actions']).max()>1
    assert len(r['frames'])==5 and len(r['packets'])==4


@pytest.mark.parametrize('fault',['short','previous','next','labels','time','nan','dtype','notfixed','bool'])
def test_bad_slice_no_substitution(fault):
    f=dataset(); index=2
    if fault=='short': f['ep_len'][index]=20
    if fault=='previous': f['ep_len'][index-1]-=1
    if fault=='next': f['ep_offset'][index+1]-=1
    if fault=='labels': f['episode_idx'][50]=99
    if fault=='time': f['step_idx'][60]=3
    if fault=='nan': f['action'][50,0]=np.nan
    if fault=='dtype': f['action']=f['action'].astype(np.float64)
    if fault=='notfixed': index=1
    if fault=='bool': index=True
    with pytest.raises(ValueError): m.slice_episode(f,index)


def test_no_result_based_dedup():
    _,p=inputs(); frames=np.zeros((5,224,224,3),np.uint8)
    for d in p: d['frames']=frames
    p[1]['actions']=p[0]['actions'].copy()
    c=m.correlations(p)
    assert len(c)==6 and c[0]['equal_pixel_steps']==list(m.STEPS) and c[0]['action_max_abs_difference']==0
    assert len(p)==4


def test_request_timeline_whitelist_teacher_forcing():
    z,p=inputs(); reqs,labels=m.requests(z,p,[0,6228,12456,18684])
    assert len(reqs)==24 and len(labels)==8 and sum(len(r['tokens']) for r in reqs)==72
    assert all(set(r)==m.REQUEST_KEYS for r in reqs)
    for slot in range(4):
        f,zr,mask,f15,z15,m15=reqs[slot*6:slot*6+6]
        np.testing.assert_array_equal(f['latent_history'],np.stack(z[slot][:3]))
        np.testing.assert_array_equal(f15['latent_history'],np.stack(z[slot][1:4]))
        np.testing.assert_array_equal(f['raw_packets'],p[slot]['actions'][:15].reshape(3,5,2))
        np.testing.assert_array_equal(f15['raw_packets'],p[slot]['actions'][5:20].reshape(3,5,2))
        np.testing.assert_array_equal(f['tokens'],mask['tokens'])
        np.testing.assert_array_equal(f['tokens'][:2],zr['tokens'][:2])
        assert np.all(zr['raw_packets'][-1]==0) and np.any(zr['tokens'][-1]!=0)
        assert not any(np.array_equal(row,z[slot][4]) for row in f15['latent_history'])
    labels[0]['target'][:]=-999
    assert reqs[0]['latent_history'][-1,0]==2 and reqs[3]['latent_history'][-1,0]==3


def test_normalization_exact_time_major_no_divisor_or_clipping():
    raw=np.arange(30,dtype=np.float32).reshape(3,5,2)
    expect=(raw.reshape(3,10)-np.tile(m.base.d2p.ACTION_MEAN,5))/np.tile(m.base.d2p.ACTION_STD,5)
    np.testing.assert_array_equal(m.normalized(raw),expect)
    assert raw.max()==29


@pytest.mark.parametrize('kind',m.CONDITIONS)
def test_mask_only_last_embedding_and_no_input_mutation(kind):
    x=np.arange(576,dtype=np.float32).reshape(1,3,192)+1; original=x.copy()
    y=m.mask_outgoing(x,kind)
    np.testing.assert_array_equal(x,original); np.testing.assert_array_equal(x[:,:2],y[:,:2])
    np.testing.assert_array_equal(y[:,-1],np.zeros((1,192),np.float32) if kind=='M' else x[:,-1])
    assert not np.shares_memory(x,y)


def mock_backend(r):
    assert set(r)==m.REQUEST_KEYS
    before=np.repeat(r['tokens'][:,0,None],192,axis=1).astype(np.float32)
    return dict(prediction=r['latent_history'].copy(),embedding_before=before,
        embedding_after=m.mask_outgoing(before[None],r['condition'])[0])


def test_all24_durable_before_score_and_exact_saved_arrays(ram):
    z,p=inputs(); reqs,labels=m.requests(z,p,[0,2,4,6]); rows=[dict(status='not_started') for _ in reqs]
    outputs=m.execute_requests(reqs,mock_backend,ram,rows,m.Limits())
    assert len(outputs)==24 and all(r['status']=='saved' for r in rows)
    assert len(ram.files)==24*7
    for i,r in enumerate(reqs):
        a=np.load(BytesIO(ram.files[f'request_{i:02d}.tokens.npy']),allow_pickle=False)
        np.testing.assert_array_equal(a,r['tokens'])
    result=m.score(outputs,labels)
    assert result['window_denominator']==8 and result['episode_denominator']==4
    assert result['P1_admission'] is False and result['counterfactual_truth'] is False
    assert result['mean_fidelity_mse']==dict(F=1.,Z=1.,M=1.)
    assert all(r['persistence_mse']==1. and r['actual_latent_change_l2']==pytest.approx(np.sqrt(192)) for r in result['windows'])


def test_labels_cannot_enter_request(ram):
    z,p=inputs(); reqs,_=m.requests(z,p,[0,2,4,6]); reqs[0]['target']=z[0][3]
    called=[]
    with pytest.raises(ValueError): m.execute_requests(reqs,lambda r:called.append(r),ram,[{} for _ in reqs],m.Limits())
    assert called==[] and ram.files=={}


@pytest.mark.parametrize('fault',['missing','ordering','repeat'])
def test_fixed_score_denominator(fault):
    z,p=inputs(); reqs,labels=m.requests(z,p,[0,2,4,6])
    out=[dict(slot=r['slot'],window=r['window'],condition=r['condition'],prediction=r['latent_history']) for r in reqs]
    if fault=='missing': out.pop()
    if fault=='ordering': out[0],out[1]=out[1],out[0]
    if fault=='repeat': out[0]=deepcopy(out[1])
    with pytest.raises(ValueError): m.score(out,labels)


def test_partial_attempts_preserved_no_retry(ram):
    z,p=inputs(); reqs,_=m.requests(z,p,[0,2,4,6]); rows=[dict(status='not_started') for _ in reqs]; calls=[]
    def fail(r):
        calls.append(r)
        if len(calls)==2: raise RuntimeError('first CUDA error mock')
        return mock_backend(r)
    with pytest.raises(RuntimeError,match='first CUDA error'): m.execute_requests(reqs,fail,ram,rows,m.Limits())
    assert len(calls)==2 and rows[0]['status']=='saved' and rows[1]['status']=='attempted'
    assert all(r['status']=='not_started' for r in rows[2:])
    assert 'request_01.tokens.npy' in ram.files and 'request_01.prediction.npy' not in ram.files


@pytest.mark.parametrize('fault',['wall','allocated','reserved','disk'])
def test_limits_refuse_before_callback(fault):
    clock=[0]; limits=m.Limits(clock=lambda:clock[0]); called=[]
    if fault=='wall': clock[0]=600
    if fault=='allocated': limits.memory=lambda:(m.ALLOCATED+1,0)
    if fault=='reserved': limits.memory=lambda:(0,m.RESERVED+1)
    if fault=='disk': limits.free=lambda:m.FLOOR-1
    ledger=m.PredictLedger(lambda e:None,limits)
    with pytest.raises(ValueError): ledger.dispatch('action_encoder',lambda:called.append(1))
    assert called==[] and ledger.entries==[]


def simulate_request(ledger):
    ledger.dispatch('action_encoder',lambda:None)
    def predict():
        ledger.dispatch('predictor',lambda:None)
        ledger.dispatch('pred_proj',lambda:None)
    ledger.dispatch('predict',predict)


def test_nested_native_calls_24_not48_72tokens():
    events=[]; ledger=m.PredictLedger(events.append,m.Limits())
    for _ in range(24): simulate_request(ledger)
    assert len(events)==192
    assert all(v==dict(attempted=24,completed=24,attempted_tokens=72,completed_tokens=72) for v in ledger.counts().values())
    with pytest.raises(ValueError): simulate_request(ledger)
    assert len(ledger.entries)==96


def test_module_fault_and_no_retry():
    ledger=m.PredictLedger(lambda e:None,m.Limits())
    def fail(): raise RuntimeError('kernel mock')
    with pytest.raises(RuntimeError): ledger.dispatch('action_encoder',fail)
    with pytest.raises(ValueError): ledger.dispatch('predict',lambda:None)
    assert ledger.failed and ledger.counts()['action_encoder']['completed']==0


@pytest.mark.parametrize('fault',['hash','content'])
def test_verify_rejects_source_drift_without_disk(monkeypatch,fault):
    monkeypatch.setattr(m,'sha',lambda p:'expected' if fault=='content' else 'wrong')
    monkeypatch.setattr(m,'read',lambda p:dict(source='old'))
    monkeypatch.setattr(m,'material',lambda:(dict(source='new'),[]))
    with pytest.raises(ValueError): m.verify('RAM','expected')


def test_no_neural_import_in_prepare_module():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('torch' in ast.unparse(n) for n in top)
    assert m.sha(m.old.__file__)==m.V1_SHA
    c=m.contract(); assert c['counts']['predict']==24 and c['tokens']['latent_input']==72
    assert c['paired_render_B'] is False and c['scientific_threshold'] is None


class FakeTensor:
    """Shape/byte carrier, NOT a neural/torch implementation."""
    device=SimpleNamespace(type='cuda')
    dtype=np.dtype('float32')
    def __init__(self,a): self.a=np.asarray(a); self.shape=self.a.shape
    def to(self,device): return self
    def __getitem__(self,key): return FakeTensor(self.a[key])
    def __setitem__(self,key,value): self.a[key]=value
    def clone(self): return FakeTensor(self.a.copy())
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class FakeTorch:
    float32=np.dtype('float32')
    is_grad_enabled=staticmethod(lambda:False)
    isfinite=staticmethod(lambda x:np.isfinite(x.a))
    equal=staticmethod(lambda x,y:np.array_equal(x.a,y.a))
    from_numpy=staticmethod(FakeTensor)
    @staticmethod
    def inference_mode():
        from contextlib import nullcontext
        return nullcontext()


class FakeModel:
    def __init__(self):
        # Return predeclared arrays: only interface routing, no neural arithmetic.
        self.encoder=SimpleNamespace(forward=lambda x:x)
        self.projector=SimpleNamespace(forward=lambda x:x)
        self.action_encoder=SimpleNamespace(forward=lambda x:FakeTensor(np.ones((1,3,192),np.float32)))
        self.predictor=SimpleNamespace(forward=lambda x,a:FakeTensor(np.full((1,3,192),2,np.float32)))
        self.pred_proj=SimpleNamespace(forward=lambda x:FakeTensor(np.full((3,192),3,np.float32)))
    def predict(self,z,a):
        hidden=self.predictor.forward(z,a)
        return FakeTensor(self.pred_proj.forward(FakeTensor(hidden.a.reshape(3,192))).a.reshape(1,3,192))


class ModuleCall:
    def __init__(self,module): self.forward=module.forward
    def __call__(self,*a,**kw): return self.forward(*a,**kw)


def routed_model():
    model=FakeModel()
    for k in ('encoder','projector','action_encoder','predictor','pred_proj'):
        setattr(model,k,ModuleCall(getattr(model,k)))
    return model


@pytest.mark.parametrize('kind',m.CONDITIONS)
def test_native_interface_routes_projection_and_mask_without_torch(kind):
    model=routed_model(); events=[]; errors=[]; ledger=m.PredictLedger(events.append,m.Limits())
    z,p=inputs(); reqs,_=m.requests(z,p,[0,2,4,6]); r=next(x for x in reqs if x['condition']==kind)
    original=model.predict
    with m.prediction_guard(model,FakeTorch,ledger,errors):
        result=m.native_predict(model,FakeTorch,'cuda:0',r)
    assert not errors and model.predict==original
    assert [e['resource'] for e in ledger.entries]==list(m.PredictLedger.ORDER)
    assert all(e['completed'] for e in ledger.entries)
    assert np.all(result['prediction']==3)  # pred_proj, NOT predictor's 2
    assert np.all(result['embedding_after'][:2]==1)
    assert np.all(result['embedding_after'][-1]==(0 if kind=='M' else 1))


def test_prediction_guard_blocks_visual_and_restores_on_error():
    model=routed_model(); errors=[]; ledger=m.PredictLedger(lambda e:None,m.Limits()); original=model.encoder.forward
    with pytest.raises(ValueError,match='no fresh visual'):
        with m.prediction_guard(model,FakeTorch,ledger,errors): model.encoder.forward(None)
    assert model.encoder.forward==original and ledger.failed and not errors


@pytest.mark.parametrize('bad',['cpu','shape','dtype','grad'])
def test_prediction_guard_rejects_wrong_contract(bad,monkeypatch):
    model=routed_model(); ledger=m.PredictLedger(lambda e:None,m.Limits()); x=FakeTensor(np.ones((1,3,10),np.float32))
    if bad=='cpu': x.device=SimpleNamespace(type='cpu')
    if bad=='shape': x.shape=(3,10)
    if bad=='dtype': x.dtype=np.dtype('float64')
    if bad=='grad': monkeypatch.setattr(FakeTorch,'is_grad_enabled',staticmethod(lambda:True))
    with pytest.raises(ValueError):
        with m.prediction_guard(model,FakeTorch,ledger,[]): model.action_encoder(x)
    assert ledger.entries==[]


def test_close_error_does_not_replace_first_execution_error(monkeypatch):
    class Output:
        def iterdir(self): return []
    class BrokenStore:
        output=Output()
        reports=[]
        def __init__(self,path): pass
        def put(self,name,value,final=False):
            if name=='report.json': self.reports.append(deepcopy(value))
            else: raise RuntimeError('FIRST binding fsync mock')
        def event(self,e): raise RuntimeError('SECOND journal mock')
        def close(self): raise RuntimeError('THIRD close mock')
    monkeypatch.setattr(m,'verify',lambda *a:(dict(episode_indices=[0,2,4,6],contract=m.contract()),[]))
    monkeypatch.setattr(m,'Store',BrokenStore)
    with pytest.raises(ValueError,match='partial retained'): m.run('RAM','sha','RAMoutput')
    report=BrokenStore.reports[0]
    assert report['first_error']['message']=='FIRST binding fsync mock'
    assert [e['stage'] for e in report['errors']]==['execution','error_journal','journal_finalize']
    assert report['status']=='partial_or_failed' and len(report['requests'])==24 and len(report['images'])==20
    assert all(r['status']=='not_started' for r in report['requests'])
    assert all(v['attempted']==v['completed']==0 for v in report['resources'].values())


def test_store_caps_and_broken_journal_in_ram(monkeypatch):
    # Exercise real Store byte-cap/floor checks, no actual file creation.
    class EmptyOutput:
        def iterdir(self): return []
        def __truediv__(self,name): return SimpleNamespace(is_symlink=lambda:False)
    store=object.__new__(m.Store); store.output=EmptyOutput()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR+1))
    with pytest.raises(ValueError): store._write('too_big',b'ab')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError): store._write('cap',b'x'*(m.CAP-m.FINAL+1))
    store.closed=store.broken=False
    import zlib
    store.comp=zlib.compressobj(6,zlib.DEFLATED,31)
    monkeypatch.setattr(store,'_write',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('disk mock')))
    with pytest.raises(RuntimeError): store.event(dict(kind='attempt'))
    assert store.broken
    with pytest.raises(ValueError): store.event(dict(kind='retry'))
    with pytest.raises(ValueError): store.close()
