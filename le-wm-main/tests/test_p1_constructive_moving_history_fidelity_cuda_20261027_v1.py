"""RAM-only protocol tests: no model, environment, fixture files or CUDA."""
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_moving_history_fidelity_cuda_20261027_v1 as m


class RAM:
    def __init__(self): self.files={}; self.events=[]
    def _write(self,name,raw,**kw):
        assert name not in self.files
        self.files[name]=raw
    def event(self,e): self.events.append(deepcopy(e))


@pytest.fixture
def ram(monkeypatch):
    store=RAM()
    def save(store,name,a):
        b=BytesIO(); np.save(b,a,allow_pickle=False); store._write(name,b.getvalue())
        return dict(file=name,sha256=m.low.digest(b.getvalue()),shape=list(a.shape),dtype=str(a.dtype))
    monkeypatch.setattr(m,'save_array',save)
    return store


def payload():
    return [dict(slot=i,task_id=str(i),root_history=np.stack([np.full(192,j+i,np.float32) for j in range(3)]),
        raw_history=m.moving.fixed_raw(),commands=np.arange(20,dtype=np.float32).reshape(10,2)/20,
        goal_latent=np.full(192,9+i,np.float32)) for i in range(2)]


def execute(ram,fail=None):
    data=payload(); prefixes=[np.full(192,4+i,np.float32) for i in range(2)]
    calls=[]; rows=[dict(index=i,status='not_started') for i in range(8)]
    def backend(r):
        calls.append(deepcopy(r))
        if len(calls)==fail: raise ValueError('first native failure')
        z=np.full((3,192),len(calls),np.float32)
        return dict(prediction=z,embedding_before=z.copy(),embedding_after=z.copy())
    results=m.evaluate(data,prefixes,backend,ram,rows,m.numerics.Limits())
    return data,prefixes,calls,rows,results


def test_fixed_eight_fresh_calls_and_imagined_prefix(ram):
    data,prefix,calls,rows,res=execute(ram)
    assert [r['kind'] for r in calls]==list(m.KINDS)*2
    assert all(set(r)==m.REQUEST_KEYS for r in calls)
    assert all(r['status']=='saved' for r in rows)
    np.testing.assert_array_equal(calls[1]['latent_history'][-1],prefix[0])
    np.testing.assert_array_equal(calls[3]['latent_history'][-1],np.full(192,3,np.float32))
    assert not np.array_equal(calls[3]['latent_history'][-1],prefix[0])
    assert not np.array_equal(calls[3]['latent_history'][-1],res[0]['predictions'][0][-1])
    np.testing.assert_array_equal(calls[0]['latent_history'],calls[2]['latent_history'])
    np.testing.assert_array_equal(calls[1]['raw_packets'],calls[3]['raw_packets'])


def test_actual_history_shift_and_no_clip(ram):
    data,_,calls,_,_=execute(ram)
    np.testing.assert_array_equal(calls[0]['raw_packets'][:2].reshape(10,2),data[0]['raw_history'])
    expected=np.concatenate([data[0]['raw_history'][5:],data[0]['commands']]).reshape(3,5,2)
    np.testing.assert_array_equal(calls[1]['raw_packets'],expected)
    np.testing.assert_array_equal(calls[1]['tokens'],m.numerics.normalized(expected))
    raw=np.full((5,2),2,np.float32)
    r=m.request(0,m.KINDS[0],data[0]['root_history'],data[0]['raw_history'],raw)
    assert np.all(r['tokens'][-1]>9)


def test_all_float32_outputs_durable_in_ram(ram):
    execute(ram)
    assert len(ram.files)==8*7
    for name,b in ram.files.items():
        a=np.load(BytesIO(b),allow_pickle=False)
        assert a.dtype==np.float32
        if name.endswith('last_native192.npy'): assert a.shape==(192,)


def test_scoring_fixed_denominators_and_persistence(ram):
    data,prefix,_,_,res=execute(ram); s=m.score(res,data,prefix)
    assert s['slot_denominator']==2 and s['predict_call_denominator']==8 and s['P1_admission'] is False
    es=s['rows'][0]['evaluations']
    assert [e['persistence_mse'] for e in es]==[4,25,4,49]
    assert es[0]['mse']==9 and es[2]['mse']==1
    with pytest.raises(ValueError): m.score(res[:1],data,prefix)


def test_partial_preserves_prior_outputs_and_no_retry(ram):
    with pytest.raises(ValueError,match='first native'): execute(ram,fail=3)
    assert 'request_01.last_native192.npy' in ram.files
    assert 'request_02.tokens.npy' in ram.files
    assert 'request_02.last_native192.npy' not in ram.files


@pytest.mark.parametrize('which',['dtype','shape','finite','slot','kind'])
def test_request_rejects_bad_input(which):
    d=payload()[0]; z=d['root_history']; h=d['raw_history']; a=d['commands'][:5]; slot=0; kind=m.KINDS[0]
    if which=='dtype': z=z.astype(float)
    if which=='shape': h=h[:5]
    if which=='finite': a=a.copy(); a[0,0]=np.nan
    if which=='slot': slot=2
    if which=='kind': kind='planner'
    with pytest.raises(ValueError): m.request(slot,kind,z,h,a)


def nested(ledger):
    ledger.dispatch('action_encoder',lambda:None)
    return ledger.dispatch('predict',lambda:(ledger.dispatch('predictor',lambda:None),ledger.dispatch('pred_proj',lambda:None)))


def test_predict_exact_api_cap_and_tokens():
    events=[]; ledger=m.Predict(events.append,m.numerics.Limits())
    for _ in range(8): nested(ledger)
    assert len(events)==64
    assert all(c==dict(attempted=8,completed=8,attempted_tokens=24,completed_tokens=24) for c in ledger.counts().values())
    with pytest.raises(ValueError): nested(ledger)


def test_predict_failure_attempt_without_return():
    ledger=m.Predict(lambda e:None,m.numerics.Limits())
    def bad(): raise RuntimeError('native')
    with pytest.raises(RuntimeError): ledger.dispatch('action_encoder',bad)
    assert ledger.counts()['action_encoder']['attempted']==1
    assert ledger.counts()['action_encoder']['completed']==0
    with pytest.raises(ValueError): nested(ledger)


def test_visual_two_not_four_images():
    l=m.Visual(lambda e:None,m.numerics.Limits())
    for i in range(2):
        l.image=i
        for kind,shape in [('encoder',(1,3,224,224)),('projector',(1,192))]:
            l.dispatch(kind,SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=shape),lambda:None)
    assert l.counts()=={k:dict(attempted=2,completed=2) for k in ('encoder','projector')}
    l.image=2
    with pytest.raises(ValueError): l.dispatch('encoder',None,lambda:None)


@pytest.mark.parametrize('bad',['disk','memory','wall'])
def test_limit_predispatch_no_native(bad):
    tick=[0]; limits=m.numerics.Limits(clock=lambda:tick[0],free=lambda:m.FLOOR if bad!='disk' else m.FLOOR-1,
        memory=lambda:(3221225473,0) if bad=='memory' else (0,0))
    if bad=='wall': tick[0]=600
    ledger=m.Predict(lambda e:None,limits)
    with pytest.raises(ValueError): nested(ledger)
    assert ledger.entries==[]


def events():
    raw=np.concatenate([m.moving.fixed_raw(),np.zeros((10,2),np.float32)])
    out=[]
    for k in ('operation_attempted','operation_completed'):
        out.extend(dict(kind=k,resource='public_step',slot=0,phase='replay0',step=i+1,raw_action=a.astype(float).tolist()) for i,a in enumerate(raw))
    for i,a in enumerate(raw[:10]):
        out.append(dict(kind='warmup_return',slot=0,phase='replay0',step=i+1,raw_action=a.astype(float).tolist(),raw_float32_bits=a.tobytes().hex()))
        out.append(dict(kind='native_return',slot=0,phase='replay0',step=i+1))
    return out


@pytest.mark.parametrize('corrupt',[None,'bits','missing','zero'])
def test_completed_actual_action_evidence(corrupt):
    es=events()
    if corrupt=='bits': es[-2]['raw_float32_bits']='00'
    if corrupt=='missing': es.pop(20)
    if corrupt=='zero': es[0]['raw_action']=[0,0]
    if corrupt:
        with pytest.raises(ValueError): m.executed_actions(es,0)
    else:
        h,a=m.executed_actions(es,0)
        assert m.bits(h,m.moving.fixed_raw()) and a.shape==(10,2)


def test_chain_full_and_partial_rejection():
    es=[]; previous=None
    for i,kind in enumerate(('run_started','run_finished')):
        e=dict(index=i,kind=kind,previous_sha256=previous); previous=m.digest_json(e); e['event_sha256']=previous; es.append(e)
    raw=b''.join(m.canonical(e) for e in es)
    assert m.source_chain(raw,previous)==es
    for bad in (raw[:-1],raw.replace(b'run_started',b'run_changed')):
        with pytest.raises(ValueError): m.source_chain(bad,previous)


def test_run_disk_probe_real_from_initialization(monkeypatch):
    seen=[]
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:(seen.append(p) or SimpleNamespace(free=m.FLOOR-1)))
    monkeypatch.setattr(m,'verify',lambda *a:({},[]))
    monkeypatch.setattr(m,'Store',lambda p:pytest.fail('must not create output'))
    with pytest.raises(ValueError,match='floor'): m.run('unused','unused','/tmp/never_created')
    assert str(seen[0])=='/tmp'


def test_contract_and_moving_loader_only():
    import inspect
    code=inspect.getsource(m.load_saved)
    assert 'moving.load_task(task_path,task_sha)' in code and 'low.load_task(' not in code
    c=m.contract()
    assert c['counts']['predict']==8 and c['counts']['encoder']==2
    assert c['physics']==c['AD']==0 and c['P1_admission'] is False
    assert m.CAP==2097152 and m.FLOOR==209715200


def test_close_error_does_not_replace_first_error(monkeypatch):
    class Broken(RAM):
        output=SimpleNamespace(iterdir=lambda:[])
        def put(self,name,value,**kw):
            if name=='binding.json': raise RuntimeError('original write failure')
            self.report=value
        def close(self): raise RuntimeError('secondary close failure')
    store=Broken()
    monkeypatch.setattr(m,'verify',lambda *a:({'arguments':{},'slots':[],'contract':m.contract()},[]))
    monkeypatch.setattr(m,'Store',lambda p:store)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='partial retained'): m.run('unused','unused','/tmp/never_created')
    assert store.report['status']=='partial_or_failed'
    assert 'original write failure' in str(store.report['first_error'])
    assert 'secondary close failure' in str(store.report['errors'][1])
    assert len(store.report['requests'])==8 and len(store.report['images'])==2
    assert all(r['status']=='not_started' for r in store.report['requests'])


@pytest.mark.parametrize('mode',['cap','floor','traversal'])
def test_writer_rejects_without_files(mode,monkeypatch):
    class FakePath:
        def __truediv__(self,name): return self
        def is_symlink(self): return False
        def iterdir(self): return []
        def open(self,*a): pytest.fail('rejection must precede writes')
    store=object.__new__(m.Store); store.output=FakePath()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR if mode=='floor' else m.FLOOR+m.CAP*2))
    name='../escape' if mode=='traversal' else 'test'
    raw=b'x'*(m.CAP-m.FINAL+1) if mode=='cap' else b'x'
    with pytest.raises(ValueError): store._write(name,raw)


def test_nonfactual_embedding_change_rejected(ram):
    def backend(r):
        z=np.zeros((3,192),np.float32)
        return dict(prediction=z,embedding_before=z,embedding_after=z+1)
    with pytest.raises(ValueError,match='no embedding mask'):
        m.evaluate(payload(),[np.zeros(192,np.float32)]*2,backend,ram,
            [dict(status='not_started') for _ in range(8)],m.numerics.Limits())


def test_native_adapter_whitelist_and_fixed_factual_mapping(monkeypatch):
    got=[]
    monkeypatch.setattr(m.numerics,'native_predict',lambda model,torch,device,r:got.append(r))
    d=payload()[0]; r=m.request(0,m.KINDS[1],d['root_history'],d['raw_history'],d['commands'][:5])
    m.native_predict(None,None,None,r)
    assert got[0]['condition']=='F' and got[0]['window']==15
    assert 'goal' not in got[0]
    with pytest.raises(ValueError): m.native_predict(None,None,None,dict(r,goal=d['goal_latent']))
