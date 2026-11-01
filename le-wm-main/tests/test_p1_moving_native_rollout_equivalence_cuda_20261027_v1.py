"""RAM array tests; official rollout AST on artificial tensors, never a neural model."""
import ast
import base64
from copy import deepcopy
import gzip
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_native_rollout_equivalence_cuda_20261027_v1 as m
from test_p1_moving_teacher_horizon_localization_cuda_20261027_v1 import fixture as horizon_fixture,census_fixture
from test_p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2 import RAM,Tensor as BaseTensor,Module


class Tensor(BaseTensor):
    @property
    def shape(self):return self.a.shape
    def size(self,dim):return self.a.shape[dim]
    def __getitem__(self,key):return Tensor(self.a[key],self.device)
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape),self.device)
    def unbind(self,dim):return [Tensor(np.take(self.a,i,axis=dim),self.device) for i in range(self.a.shape[dim])]


def official_fixture():
    dev=NS(type='cuda')
    torch=NS(float32='float32',tensor=lambda a,**kw:Tensor(a,kw['device']),is_grad_enabled=lambda:False,
        cuda=NS(synchronize=lambda d:None))
    torch.stack=lambda values,dim=0:Tensor(np.stack([v.a for v in values],axis=dim),dev)
    torch.cat=lambda values,dim=0:Tensor(np.concatenate([v.a for v in values],axis=dim),dev)
    torch.split=lambda x,sizes,dim:tuple(Tensor(a,dev) for a in np.split(x.a,np.cumsum(sizes)[:-1],axis=dim))
    def rearrange(x,pattern,**kw):
        if pattern=='b s ... -> (b s) ...':return Tensor(x.a.reshape((x.a.shape[0]*x.a.shape[1],)+x.a.shape[2:]),dev)
        assert pattern=='(b s) ... -> b s ...'
        return Tensor(x.a.reshape((kw['b'],kw['s'])+x.a.shape[1:]),dev)
    # Execute EXACT installed rollout function, only replacing tensor primitives with RAM arrays.
    tree=ast.parse(m.OFFICIAL.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LeWM')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='rollout')
    namespace=dict(torch=torch,rearrange=rearrange)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(m.OFFICIAL),'exec'),namespace)
    class Model:
        def __init__(self):
            self.action_encoder=Module(lambda a:Tensor(np.zeros((1,a.shape[1],192),np.float32),dev))
            self.predictor=Module(lambda z,a:Tensor(z.a+a.a,dev))
            self.pred_proj=Module(lambda z:Tensor(z.a+np.float32(.25),dev))
            self.encoder=Module(lambda x:x);self.projector=Module(lambda x:x)
        def predict(self,z,a):return self.pred_proj(self.predictor(z,a).reshape(3,192)).reshape(1,3,192)
        def encode(self,*a):raise AssertionError('official cached path MUST NOT encode')
    Model.rollout=namespace['rollout']
    return Model(),torch,dev


def fixture():
    hp,unused=horizon_fixture();events,rows,files=census_fixture();queries=m.make_queries(hp,events)
    old=RAM();old.output=Path('/RAM/reference');references=[]
    def step(z,tokens,number):
        ae=dict(resource='action_encoder',inputs=[old.array(f'old{number}_ae_in.npy',tokens[None])],
            output=old.array(f'old{number}_ae_out.npy',np.zeros((1,3,192),np.float32)))
        api=dict(resource='predict_API')
        pred=dict(resource='predictor',inputs=[old.array(f'old{number}_p_in.npy',z[None]),ae['output']],
            output=old.array(f'old{number}_p_out.npy',z[None]))
        projected=z+np.float32(.25)
        proj=dict(resource='pred_proj',inputs=[old.array(f'old{number}_q_in.npy',z)],
            output=old.array(f'old{number}_q_out.npy',projected))
        return [ae,api,pred,proj],old.array(f'old{number}_final.npy',projected[-1]),projected[-1].copy()
    for n,q in enumerate(queries):
        z=np.asarray(q['emb'],np.float32).reshape(3,192);tokens=np.asarray(q['action_sequence'],np.float32).reshape(-1,10)
        entries=[];finals=[]
        for s in range(len(tokens)-2):
            e,f,p=step(z,tokens[s:s+3],f'{n}_{s}');entries.append(e);finals.append(f)
            z=np.concatenate((z[1:],p[None]))
        references.append(dict(identity=deepcopy(q['identity']),kind=q['kind'],module_entries=entries,finals=finals))
    p=dict(schema=m.NAME+'.prepare',queries=queries,references=references,
        goals=[dict(identity=e['identity'],goal=e['goal']) for e in hp['evaluators']],contract=m.contract(),
        expected_model_hash=hp['expected_model_hash'])
    return p,old


def execute(monkeypatch,fail_at=None,change_ae=False,label_error=False):
    p,old=fixture();s=RAM();s.output=Path('/RAM/native');l=m.Ledger(s);rows=m.initial_rows(p)
    model,torch,dev=official_fixture();previous=model.predict
    monkeypatch.setattr(m,'verify_class',lambda model:None)  # Artificial tensor class only, never an authority bypass in source.
    if change_ae:model.action_encoder.fn=lambda a:Tensor(np.full((1,a.shape[1],192),np.float32(.1),np.float32),dev)
    loads=[]
    def load_old(e):
        assert 'predictions_seal.json' in s.files and l.counts()=={k:dict(attempted=v,completed=v) for k,v in m.contract()['resources'].items()}
        loads.append(e['file'])
        if label_error:raise FileNotFoundError('artificial reference missing')
        return old.load(e)
    pixels=lambda receipts:Tensor(np.zeros((1,1,3,3,224,224),np.float32),dev)
    with m.NativeRollout(model,torch,dev,l,s,pixels) as native:
        def forward(q):
            if l.current==fail_at:raise RuntimeError('artificial failure')
            return native(q)
        seal=m.execute(p,s,l,forward,rows,load_new=s.load,load_old=load_old)
    assert model.predict==previous and 'predict' not in model.__dict__ and 'encode' not in model.__dict__
    assert all(not x.pre and not x.post for x in (model.action_encoder,model.predictor,model.pred_proj,model.encoder,model.projector))
    return p,old,s,l,rows,loads,seal


def test_direct_official_ast_cached_rollout_budget_shapes_and_all_exact(monkeypatch):
    p,old,s,l,rows,loads,seal=execute(monkeypatch)
    assert len(l.entries)==156 and loads and len(rows)==24 and all(r['status']=='compared' for r in rows)
    assert l.counts()=={k:dict(attempted=v,completed=v) for k,v in m.contract()['resources'].items()}
    for n,(q,row) in enumerate(zip(p['queries'],rows,strict=True)):
        pred=s.load(row['predicted_emb']);assert pred.shape==(1,1,5 if n%2==0 else 4,192)
        es=[e for e in l.entries if e['call_index']==n];m.validate_trace(q,pred,es,s.load)
        assert row['comparison']['full_predicted_emb']['bitwise'] and row['comparison']['final']['maxULP']==0
        call=m.core.saved.parse(s.files[f'call_{n:02d}_input.json'])
        assert call['pixels_shape']==[1,1,3,3,224,224]
        assert s.load(call['action_sequence']).shape==(1,1,4 if n%2==0 else 3,10)
    events=[m.core.saved.parse(x) for x in s.files['events.jsonl'].splitlines()]
    at=next(e['index'] for e in events if e['kind']=='all_predictions_sealed')
    assert all(e['index']>at for e in events if e['kind']=='rollout_compared')


def test_report_differences_without_tolerance_or_success_gate(monkeypatch):
    p,old,s,l,rows,loads,seal=execute(monkeypatch,change_ae=True)
    assert all(r['status']=='compared' for r in rows)
    assert all(not r['comparison']['final']['bitwise'] and r['comparison']['final']['maxabs']>0 for r in rows)
    assert all(not r['comparison']['criterion_API_called'] for r in rows)


@pytest.mark.parametrize('n',[0,1,10,23])
def test_partial_keeps24_queries_and_no_reference_decode(monkeypatch,n):
    p,old=fixture();s=RAM();l=m.Ledger(s);rows=m.initial_rows(p);model,torch,dev=official_fixture()
    monkeypatch.setattr(m,'verify_class',lambda model:None)
    pixels=lambda receipts:Tensor(np.zeros((1,1,3,3,224,224),np.float32),dev)
    with m.NativeRollout(model,torch,dev,l,s,pixels) as native:
        def forward(q):
            if l.current==n:raise RuntimeError('partial')
            return native(q)
        with pytest.raises(RuntimeError):m.execute(p,s,l,forward,rows,load_old=lambda e:pytest.fail('future reference'))
    assert len(rows)==len(m.core.saved.parse(s.files['initial_rows.json']))==24
    assert 'predictions_seal.json' not in s.files
    assert sum(r['status']=='predicted' for r in rows)==n and rows[n]['status']=='attempted'


def test_no_encoder_image_or_cpu_fallback_and_restore(monkeypatch):
    model,torch,dev=official_fixture();s=RAM();l=m.Ledger(s)
    with pytest.raises(ValueError):m.verify_class(model)
    with pytest.raises(ValueError):m.NativeRollout(model,torch,NS(type='cpu'),l,s,None)
    monkeypatch.setattr(m,'verify_class',lambda model:None)
    with pytest.raises(ValueError):
        with m.NativeRollout(model,torch,dev,l,s,None):model.encode({})
    assert 'encode' not in model.__dict__
    with pytest.raises(ValueError):
        with m.NativeRollout(model,torch,dev,l,s,None):model.encoder(None)
    assert not model.encoder.pre


@pytest.mark.parametrize('fault',['future','episode','tokens','emb','goal','RGBmissing'])
def test_illegal_query_no_future_dummy_or_shape_downgrade(fault):
    p,old=fixture();q=deepcopy(p['queries'][1])
    if fault=='future':q['rgb_history'][-1].update(step=10)
    if fault=='episode':q['rgb_history'][0]['path']=q['rgb_history'][0]['path'].replace('e00_O','e00_R')
    if fault=='tokens':q['action_sequence']=[[[[0.]*10]*2]]
    if fault=='emb':q['emb']=np.zeros((1,3,192),np.float32).tolist()
    if fault=='goal':q['goal']=[0.]*192
    if fault=='RGBmissing':q['rgb_history'].pop()
    with pytest.raises(ValueError):m.validate_query(q)


def test_missing_real_frame_or_changed_root_frame_refuses_make_queries():
    hp,_=horizon_fixture();events,rows,files=census_fixture()
    ix=next(i for i,e in enumerate(events) if e['kind']=='fresh_observation' and e['receipt']['phase']=='warmup' and e['receipt']['step']==5)
    events.pop(ix)
    with pytest.raises(ValueError):m.make_queries(hp,events)
    events,rows,files=census_fixture()
    next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan' and e['receipt']['step']==0)['receipt']['pixel_sha256']='f'*64
    with pytest.raises(ValueError):m.make_queries(hp,events)


def test_float32_ulp_signed_zero_adjacent_and_nonfinite():
    x=np.array([1.,-1.,0.],np.float32);y=x.copy();y[0]=np.nextafter(y[0],np.float32(2))
    c=m.compare(x,y);assert c['maxULP']==1 and c['maxabs']==float(y[0])-1 and not c['bitwise']
    c=m.compare(np.array([-0.],np.float32),np.array([0.],np.float32))
    assert c['numeric_equal'] and c['maxULP']==0 and not c['bitwise']
    with pytest.raises(ValueError):m.compare(x,x.astype(float))
    with pytest.raises(ValueError):m.compare(x,np.full(3,np.nan,np.float32))


def test_single_ae4_not_ae3_twice_and_no_25th_rollout():
    plan=m.operation_plan();assert len(plan)==156
    assert sum(k=='action_encoder' for call,k,s in plan)==24 and sum(k=='predict_API' for call,k,s in plan)==36
    l=m.Ledger(RAM());l.current=0;l.begin('rollout_API')
    with pytest.raises(ValueError):l.begin('predict_API')
    l.entries=[{}]*156
    with pytest.raises(ValueError):l.begin('rollout_API')


@pytest.mark.parametrize('fault',[None,'double','trailing','truncated','sha','bomb'])
def test_prepare_envelope_lossless_and_caps(fault):
    p,old=fixture();raw=m.encode_prepare(p)
    if fault is None:assert len(raw)<=262144 and m.decode_prepare(raw)==p;return
    e=m.core.saved.parse(raw);z=base64.b64decode(e['payload'])
    if fault=='double':z+=gzip.compress(b'{}')
    if fault=='trailing':z+=b'junk'
    if fault=='truncated':z=z[:-1]
    if fault=='sha':e['semantic_sha256']='0'*64
    if fault=='bomb':z=gzip.compress(b'x'*(m.SEMANTIC_CAP+1))
    e['payload']=base64.b64encode(z).decode()
    with pytest.raises(ValueError):m.decode_prepare(m.canonical(e))


def test_exact_saved_consume_all_traces_shapes_and_costs(monkeypatch):
    p,old,s,l,rows,loads,seal=execute(monkeypatch)
    r=dict(schema=m.NAME+'.report',status='native_rollout_comparison_recorded_not_admitted',prepared='/RAM/prepare',prepare_sha256='a'*64,
        errors=[],blocked=[],source_verified=True,model_unchanged=True,model_hash_before=p['expected_model_hash'],model_hash_after=p['expected_model_hash'],
        contract=m.contract(),fixed_episode_denominator=12,fixed_rollout_denominator=24,rows=rows,resources=l.counts(),ledger=l.entries,
        predictions_seal_sha256=seal,artifact_sha256={k:m.core.saved.sha_bytes(v) for k,v in s.files.items()})
    s.files['report.json']=m.canonical(r)
    def read(path,*a):
        path=Path(path);return old.files[path.name] if path.parent==m.REFERENCE.parent else s.files[path.name]
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p));monkeypatch.setattr(m,'sha',lambda path:m.core.saved.sha_bytes(s.files[Path(path).name]))
    monkeypatch.setattr(m.core.saved,'read_bytes',read);monkeypatch.setattr(Path,'read_bytes',lambda path:read(path))
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([s.output/k for k in s.files]));monkeypatch.setattr(Path,'is_symlink',lambda path:False)
    result=m.consume(s.output,m.core.saved.sha_bytes(s.files['report.json']));assert len(result['rows'])==24
    s.files['extra.npy']=b'not declared'
    with pytest.raises(ValueError,match='artifacts'):m.consume(s.output,m.core.saved.sha_bytes(s.files['report.json']))
