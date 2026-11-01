"""RAM arithmetic/ABI tests only. No model, simulator, actual prepare or results."""
from copy import deepcopy
from io import BytesIO
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_teacher_cross_library_20261028_v1 as m


class MemoryStore:
    array=m.Store.array
    def __init__(self):self.data={};self.events=[]
    def write(self,name,raw,**kw):
        assert name not in self.data
        self.data[name]=raw
    def event(self,e):self.events.append(deepcopy(e))
    def load(self,r):
        raw=self.data[r['file']];assert hashlib.sha256(raw).hexdigest()==r['file_sha256']
        plain=gzip.decompress(raw);assert hashlib.sha256(plain).hexdigest()==r['npy_sha256']
        a=np.load(BytesIO(plain),allow_pickle=False)
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['array_sha256']
        assert list(a.shape)==r['shape'] and str(a.dtype)==r['dtype']
        return a


def fixture_anchors():
    anchors=[]
    for task,(slot,condition) in enumerate(m.probe.ORDER):
        for role in m.ROLES:
            prefix=np.zeros((5,2),'<f4') if role=='Z' else np.tile(np.asarray([.1,-.1],'<f4'),(5,1))
            a=dict(index=len(anchors),task=task,root_slot=slot,condition=condition,role=role,
                task_id=f'task{task}',blind_sha256='a'*64,task_file='/original/task',task_sha256='b'*64,
                root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                    latent_history=np.arange(576,dtype=np.float32).reshape(3,192).tolist(),
                    action_history=m.kernel.normalized_history(np.tile(np.asarray([.1,0],'<f4'),(10,1))).tolist()),
                goal=np.zeros(192).tolist(),q=np.eye(192).tolist(),prefix_commands=prefix.tolist(),png='fixture.png',
                png_sha256='c'*64,pixel_sha256='d'*64)
            a['anchor_id']=m.digest(a);anchors.append(a)
    return anchors


class AnalyticBackend:
    """Deterministic RAM fixture, NOT neural/AD/physical evidence."""
    def __init__(self,row,store):self.row=row;self.store=store
    def module(self,resource,inputs,output,scope):
        n=len([k for k in self.store.data if k.endswith('_out.npy.gz')]);stem=f'm{n}'
        return dict(index=n,resource=resource,scope=scope,completed=True,
            inputs=[self.store.array(stem+f'_in{i}',np.asarray(a,np.float32)) for i,a in enumerate(inputs)],
            output=self.store.array(stem+'_out',np.asarray(output,np.float32)))
    def __call__(self,stage,context,raw):
        assert stage==m.stages()[len(self.row['forwards'])]
        hist=m.incoming(context,raw);z=np.asarray(context['latent_history'][-1],np.float32).copy()
        z[:10]+=raw.reshape(-1)
        after=dict(context,latent_history=context['latent_history'][1:]+[z.tolist()],
            action_history=hist['action_history'][1:])
        scope=dict(anchor=self.row['index'],stage=stage)
        aeout=np.zeros((1,3,192),np.float32)
        predictor=np.zeros((1,3,192),np.float32);predictor[0,-1]=z
        mods=[self.module('action_encoder',[np.asarray(hist['action_history'])[None]],aeout,scope),
              self.module('predictor',[np.asarray(hist['latent_history'])[None],aeout],predictor,scope),
              self.module('pred_proj',[predictor.reshape(3,192)],predictor.reshape(3,192),scope)]
        f=dict(index=len(self.row['forwards']),stage=stage,expected=hist,raw_commands=raw.tolist(),modules=mods,completed=True)
        if stage=='jacobian':
            f['AD_API']=[dict(resource='jacobian',completed=True,is_grads_batched=False),dict(resource='grad',completed=True,is_grads_batched=True)]
        self.row['forwards'].append(f)
        if stage=='jacobian':return np.eye(192,10)
        if stage=='nominal':return z.astype(float)
        return after


@pytest.fixture(scope='module')
def compiled():
    s=MemoryStore();a=fixture_anchors();z=[np.linspace(-1,1,192,dtype=np.float32)+np.float32(i/8) for i in range(8)]
    rows=[dict(index=i,status='not_started') for i in range(8)]
    m.compile_library(a,z,lambda r,b:AnalyticBackend(r,s),s,rows)
    images=[]
    for anchor,latent in zip(a,z,strict=True):
        i=anchor['index'];pixels=np.zeros((224,224,3),np.uint8);pre=np.zeros((1,1,3,224,224),np.float32)
        enout=np.zeros((1,257,192),np.float32);enout[:,0]=latent
        backend=AnalyticBackend({},s)
        en=backend.module('encoder',[pre.reshape(1,3,224,224)],enout,dict(image_index=i))
        proj=backend.module('projector',[enout[:,0]],latent[None],dict(image_index=i))
        images.append(dict(index=i,anchor_id=anchor['anchor_id'],status='completed',modules=[en,proj],
            pixels=s.array(f'i{i}_pixels',pixels),preprocessed=s.array(f'i{i}_pre',pre),latent=s.array(f'i{i}_z',latent)))
    return s,a,rows,images


def test_fixed_contract():
    c=m.contract();assert c['total']==dict(encoder=8,projector=8,predict=96,predictor_samples=96,action_encoder=96,pred_proj=96,
        jacobian=8,AD_forward_in_predict=8,mathematical_VJP=1536,svd=8,bounded_ridge=8)
    assert c['P1_admission'] is c['C2_non_greedy_premise'] is c['equal_FLOPs'] is False
    assert c['future_physical_denominator']==40 and c['retries']==0


def test_all96_and_freeze_order(compiled):
    s,a,rows,images=compiled
    assert sum(len(r['forwards']) for r in rows)==96
    assert sum(len(r['evaluations']) for r in rows)==80
    assert sum(len(r['math']) for r in rows)==8
    ev=s.events
    assert max(i for i,e in enumerate(ev) if e['kind']=='suffix_frozen')<min(i for i,e in enumerate(ev) if e['kind']=='common_bank_frozen')
    assert [e['task'] for e in ev if e['kind']=='common_bank_frozen']==list(range(4))


def test_actual_resource_axes_separate_ad_samples_and_helper(compiled):
    s,a,rows,images=compiled
    modules=[e for im in images for e in im['modules']]+[e for r in rows for f in r['forwards'] for e in f['modules']]
    actual=m.actual_resources(modules,rows)
    assert actual['predict']==actual['predictor_samples']==dict(attempted=96,completed=96)
    assert actual['jacobian']==actual['AD_grad_API']==actual['AD_forward_in_predict']==dict(attempted=8,completed=8)
    assert actual['mathematical_VJP']==dict(attempted=1536,completed=1536)
    assert actual['SVD_successful_helper_calls']==8 and actual['SVD_failed_helper_internal_attempts']==0
    broken=deepcopy(rows);broken[0]['math'][0]['completed']=False
    partial=m.actual_resources(modules,broken)
    assert partial['SVD_successful_helper_calls']==7 and partial['SVD_failed_helper_internal_attempts']=='unknown'


def test_compiler_has_no_privileged_input():
    a=fixture_anchors()[1];a.update(native_state='not allowed',reward=1000,probe_response='hidden')
    b=m.blind_anchor(a,np.zeros(192,np.float32))
    assert set(b)=={'context','goal','q','raw_history'}
    assert not {'native_state','reward','probe_response'} & set(b)


@pytest.mark.parametrize('i',range(8))
def test_teacher_history_is_real5_10_15(i):
    a=fixture_anchors()[i];z=np.full(192,np.float32(-i));b=m.blind_anchor(a,z)
    assert b['context']['latent_history'][:2]==a['root_context']['latent_history'][1:]
    assert b['context']['latent_history'][-1]==z.tolist()
    raw=np.concatenate((np.tile(np.asarray([0,.1],'<f4'),(5,1)),m.probe.raw(a['prefix_commands'],5)))
    assert m.probe.raw(b['raw_history'],10).tobytes()==raw.tobytes()
    assert np.array_equal(b['context']['action_history'],m.kernel.normalized_history(raw))
    assert not np.array_equal(b['context']['action_history'],m.kernel.normalized_history(raw/100))


def test_inverse_is_bitwise_sign_flip_and_no_reverse():
    p=np.linspace(-.3,.3,10);u=m.raw_suffix(p);v=m.raw_suffix(-p)
    b=m.common_bank(u,v)
    assert [x.dtype for x in b]==[np.float32]*5
    assert b[2].tobytes()==(-u).tobytes() and b[4].tobytes()==(-v).tobytes()
    assert b[2].tobytes()!=(-u[::-1]).tobytes()
    assert b[0].tobytes()==np.zeros((5,2),'<f4').tobytes()


def test_duplicate_actions_not_deduplicated():
    z=np.zeros((5,2),'<f4');b=m.common_bank(z,z)
    assert len(b)==5 and all(np.array_equal(x,z) for x in b)
    assert b[2].tobytes()!=b[0].tobytes()  # Signed zero authority survives.


@pytest.mark.parametrize('bad',[np.ones(9),np.full(10,np.inf),np.ones(10)*100])
def test_domain_fail_closed(bad):
    with pytest.raises(ValueError):m.raw_suffix(bad)


@pytest.mark.parametrize('dtype',[np.uint8,np.float32,np.float64])
def test_three_level_hashes(dtype):
    s=MemoryStore();a=np.arange(12).reshape(3,4).astype(dtype);r=s.array('one',a)
    assert len({r[k] for k in ('file_sha256','npy_sha256','array_sha256')})==3
    assert np.array_equal(s.load(r),a)
    with pytest.raises(AssertionError):s.array('one',a)


def test_all_native_bridges_and_saved_arithmetic(compiled,monkeypatch):
    s,a,rows,images=compiled
    monkeypatch.setattr(m,'rgb',lambda *args:np.zeros((224,224,3),np.uint8))
    plans=m.validate_saved(dict(anchors=a),dict(rows=rows,images=images),s.load)
    assert len(plans)==40
    assert [(x['task'],x['role'],x['library_index']) for x in plans]==[(t,r,i) for t in range(4) for r in m.ROLES for i in range(5)]
    for p in plans:
        h=p['plan_digest'];assert m.digest({k:v for k,v in p.items() if k!='plan_digest'})==h
        assert len(p['commands'])==10 and p['selection_uses_physics'] is False


@pytest.mark.parametrize('mutation',['cost','tokens','candidate','J','nominal','AD','stage','history','partial','missing','image','role'])
def test_corruption_rejected(compiled,monkeypatch,mutation):
    s,a,original,im=compiled;rows=deepcopy(original);images=deepcopy(im);a=deepcopy(a)
    monkeypatch.setattr(m,'rgb',lambda *args:np.zeros((224,224,3),np.uint8))
    if mutation=='cost':rows[0]['evaluations'][0]['cost']+=1
    elif mutation=='tokens':rows[0]['forwards'][0]['expected']['action_history'][0][0]+=1
    elif mutation=='candidate':rows[0]['evaluations'][1]['raw_commands'][0][0]=.5
    elif mutation=='J':rows[0]['arrays']['jacobian']=rows[0]['arrays']['nominal']
    elif mutation=='nominal':rows[0]['arrays']['nominal']=rows[1]['arrays']['nominal']
    elif mutation=='AD':rows[0]['forwards'][1]['AD_API'][1]['is_grads_batched']=False
    elif mutation=='stage':rows[0]['forwards'][0]['stage']='replay_0'
    elif mutation=='history':rows[0]['blind_input']['context']['latent_history'][0][0]+=1
    elif mutation=='partial':rows[0]['status']='failed_partial'
    elif mutation=='missing':rows.pop()
    elif mutation=='image':images[0]['latent']=images[1]['latent']
    else:a[0]['role']='S'  # This must change the anchor seal, not only its label.
    with pytest.raises((ValueError,AssertionError)):
        if mutation=='role':
            assert m.digest({k:v for k,v in a[0].items() if k!='anchor_id'})==a[0]['anchor_id']
        m.validate_saved(dict(anchors=a),dict(rows=rows,images=images),s.load)


def test_full_capacity_not_only_compressed_volume():
    c=m.dense_capacity();assert c['fits'] and c['storage_bound_bytes']<m.CAP
    assert c['storage_bound_bytes']>m.JOURNAL_CAP+m.REPORT_CAP
    assert c['original_array_bytes_bound']>8*257*192*4


def test_no_worker_run_without_authorization(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *args:pytest.fail('must fail before verify/output/model'))
    with pytest.raises(ValueError,match='authorization'):m.run('x','a'*64,'x',None)


def test_cli_help_no_neural_import():
    import os
    e=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    out=subprocess.run([sys.executable,'-m',m.__name__,'--help'],capture_output=True,text=True,env=e)
    assert out.returncode==0 and 'build-memory' in out.stdout


@pytest.mark.parametrize('name',['../evil','/tmp/escape','','..'])
def test_flat_paths_reject_traversal(name):
    with pytest.raises(ValueError):m.flat('/tmp',name)


def test_primary_exception_is_not_replaced_by_finalization_source():
    # Static seam: each finalizer guarded independently; no unguarded digest/write finally.
    import inspect
    source=inspect.getsource(m.run)
    assert "err('execution',e)" in source and "err('final_source_or_wall',e)" in source
    assert "err('final_report',e)" in source and "first_error=errors[0]" in source


def test_no_torch_import_for_prepare_module():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and
        ('torch' in (getattr(n,'module',None) or '') or any(x.name=='torch' for x in n.names)) for n in tree.body)


def test_no_retry_full_failure_denominator_and_first_error(monkeypatch):
    import builtins
    class Sink:
        def __init__(self,*args):self.objects={}
        def put(self,name,value,**kwargs):self.objects[name]=deepcopy(value)
        def event(self,e):raise OSError('secondary journal')
        def census(self):return {}
        def check(self):return None
        index=0;previous=None
    sink=Sink();monkeypatch.setattr(m,'Store',lambda *a:sink)
    calls=[]
    def verify(*args):
        calls.append(1)
        if len(calls)>1:raise OSError('secondary source')
        return dict(expected_model_hash='a'*64)
    monkeypatch.setattr(m,'verify',verify)
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name=='torch':raise RuntimeError('FIRST CUDA import denied in RAM fixture')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',importing)
    with pytest.raises(ValueError,match='partial preserved'):m.run('not_actual','a'*64,'not_actual',m.AUTH)
    r=sink.objects['report.json']
    assert r['first_error']['message']=='FIRST CUDA import denied in RAM fixture'
    assert [e['phase'] for e in r['errors']]==['execution','error_journal','final_source_or_wall']
    assert len(r['rows'])==8 and len(r['library_slots'])==40
    assert r['commands_released'] is False and len(calls)==2


def test_partial_bytes_charged_on_write_error_without_disk(monkeypatch):
    from types import SimpleNamespace
    s=m.Store.__new__(m.Store);s.output=Path('/RAM/not_created');s.used=0;s.files={};s.check=lambda *args:None
    monkeypatch.setattr(m,'flat',lambda *args:Path('/RAM/not_created/new'))
    size=[0]
    monkeypatch.setattr(m.os,'open',lambda *args:77)
    monkeypatch.setattr(m.os,'fstat',lambda fd:SimpleNamespace(st_mode=0o100600,st_dev=1,st_ino=2,st_size=size[0]))
    monkeypatch.setattr(m.os,'close',lambda fd:None)
    class Partial:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def write(self,raw):size[0]+=3;raise OSError('partial write')
    monkeypatch.setattr(m.os,'fdopen',lambda *args,**kwargs:Partial())
    with pytest.raises(OSError,match='partial write'):s.write('new',b'123456')
    assert s.used==3 and s.files['new']==(1,2,3)


def test_own_native_trace_hard_caps_before_dispatch(monkeypatch):
    from types import SimpleNamespace
    class Hook:
        def remove(self):return None
    class Module:
        def register_forward_pre_hook(self,fn):self.pre=fn;return Hook()
        def register_forward_hook(self,fn):return Hook()
    model=SimpleNamespace(**{k:Module() for k in ('encoder','projector','action_encoder','predictor','pred_proj')})
    trace=m.NativeTrace(model,None,MemoryStore(),lambda:None,[1,257,192])
    with trace:
        trace.scope=dict(anchor=0,stage='nominal')
        trace.entries=[dict(resource='predictor',scope=dict(anchor=-1,stage=str(i)),completed=True) for i in range(96)]
        with pytest.raises(ValueError,match='predispatch8/96'):model.predictor.pre(model.predictor,())
        trace.entries=[dict(resource='encoder',scope=dict(image_index=-i-1),completed=True) for i in range(8)]
        trace.scope=dict(image_index=0)
        with pytest.raises(ValueError,match='predispatch8/96'):model.encoder.pre(model.encoder,())

