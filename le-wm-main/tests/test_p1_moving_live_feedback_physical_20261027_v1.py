"""RAM-only artificial transitions/tensors: no simulator, CUDA, model or tmp."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as m
from experiments.spectral_reachability import p1_moving_live_suffix_cuda_20261027_v1 as suffix
from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as enc
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as movingenc
from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv, fixture as oldfixture, REQUEST


class RAM:
    def __init__(self, path='/RAM_LIVE'):
        self.path=Path(path); self.files={}; self.events=[]; self.previous=None; self.index=0
    def check(self,size=4096):
        assert sum(len(v) for v in self.files.values())+size < m.CAP-m.FINAL
    def write(self,name,value,append=False,final=False):
        self.check(len(value)); assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+value
        return hashlib.sha256(self.files[name]).hexdigest()
    def put(self,name,value,final=False):return self.write(name,m.raw(value),final=final)
    def emit(self,kind,**fields):
        e=dict(index=self.index,kind=kind,previous_sha256=self.previous,**deepcopy(fields))
        e['event_sha256']=m.source.digest(e);self.previous=e['event_sha256'];self.events.append(e);self.index+=1
        self.write('journal.jsonl',m.raw(e),append=True)
    def namespace(self,name):return self.path/name
    def frame(self,name,a):
        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');return self.write(name,b.getvalue())
    def array(self,name,a):
        b=io.BytesIO();np.save(b,a,allow_pickle=False);h=self.write(name,b.getvalue())
        return dict(path=str(self.path/name),file_sha256=h,array_sha256=m.array_sha(a),shape=list(a.shape),dtype=str(a.dtype))
    def load_array(self,r):
        payload=self.files[str(Path(r['path']).relative_to(self.path))]
        assert hashlib.sha256(payload).hexdigest()==r['file_sha256']
        a=np.load(io.BytesIO(payload),allow_pickle=False);assert m.array_sha(a)==r['array_sha256'];return a
    def hash(self,path):return hashlib.sha256(self.files[str(Path(path).relative_to(self.path))]).hexdigest()


def fixture():
    _,d=oldfixture();layers=[];tasks={}
    context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                 latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.zeros((2,10)).tolist())
    # Obtain metadata only, no source material/model loaded.
    mb=suffix.modelbinding()
    for slot,arm in m.LAYERS:
        layers.append(dict(source_slot=slot,arm=arm,task_id=str(slot+1)*64,
                           task_export='/approved/task'+str(slot),task_sha256=suffix.planner.EXPORT_SHAS[slot],
                           selection_sha256=str(slot+3)*64,commands=np.full((10,2),.05,np.float32).tolist(),
                           imagined_prefix=np.full(192,2,np.float32).tolist()))
        tasks[slot]=dict(gn_task=dict(root_context=deepcopy(context),goal=np.zeros(192,np.float32).tolist(),
                                     action_mean=mb['action_mean'],action_std=mb['action_std']),
                         modelbinding=mb,model_bindings=enc.MODELS)
    return dict(contract=m.contract(),layers=layers,bindings={}),dict(d,tasks=tasks)


class Tensor:
    def __init__(self,a,device):self.a=np.asarray(a,np.float32);self.device=device;self.dtype='float32';self.shape=self.a.shape
    def __getitem__(self,key):return Tensor(self.a[key],self.device)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


class FakeSuffix:
    def __init__(self):self.requests=[];self.models=[];self.fail=False
    def compile_suffix(self,request,sink,*,model,device,check_limits):
        check_limits();self.requests.append(deepcopy(request));self.models.append(model)
        # Exercise frozen packet whitelist and all causal seals, not neural mathematics.
        suffix.validate_packet(request)
        if self.fail:raise RuntimeError('artificial suffix failure')
        def eventpair(kind1,kind2,entry):
            sink.event(dict(kind=kind1,entry=dict(entry,completed=False)))
            sink.event(dict(kind=kind2,entry=dict(entry,completed=True)))
        for i,stage in enumerate(suffix.frozen.STAGES):
            eventpair('predict_API_attempt','predict_API_return',dict(index=i,stage=stage))
            for j,resource in enumerate(('action_encoder','predict','pred_proj')):
                eventpair('module_attempt','module_return',dict(index=i*3+j,stage=stage,resource=resource))
        for i,resource in enumerate(('jacobian','svd')):
            entry=dict(index=i,resource=resource)
            if i==0:entry.update(shape=[192,10],mathematical_VJP=192)
            eventpair('math_attempt','math_return',entry)
        for i,resource in enumerate(('jacobian','grad')):
            eventpair('AD_API_attempt','AD_API_completed',dict(index=i,resource=resource))
        return dict(packet_sha256=m.packet_digest(request),episode_observation_seal=request['episode_observation_seal'],
                    source_verified=True,model_unchanged=True,artifact_sha256={},dispatch_raw5=request['oldrawsuffix5'])
    def verify_suffix_seal(self,request,response):return response['dispatch_raw5']


@contextmanager
def visual_mock(store,*,load_failure=False):
    """Real Handler and real visual Ledger; only tensors/modules are artificial."""
    device=NS(type='cuda');holder=NS(ledger=None,loads=0,sync=0)
    torch=ModuleType('torch');torch.float32='float32';torch.device=lambda _:device
    torch.is_grad_enabled=lambda:False;torch.inference_mode=nullcontext;torch.autocast=lambda **_:nullcontext()
    torch.backends=NS(cuda=NS(matmul=NS(allow_tf32=True)),cudnn=NS(allow_tf32=True))
    def sync(_):holder.sync+=1
    torch.cuda=NS(synchronize=sync,memory_allocated=lambda _:0,memory_reserved=lambda _:0)
    attention=ModuleType('torch.nn.attention');attention.SDPBackend=NS(MATH='math');attention.sdpa_kernel=lambda _:nullcontext()
    compat=ModuleType('experiments.checkpoint_compat');preflight=ModuleType('experiments.gpu_preflight')
    preflight.require_cuda_allocation=lambda _:None
    class Model:
        def eval(self):return self
        def requires_grad_(self,v):assert v is False;return self
        def encode(self,value):
            x=Tensor(np.zeros((1,3,224,224)),device);z=Tensor(np.ones((1,192)),device)
            holder.ledger.dispatch('encoder',x,lambda:None)
            holder.ledger.dispatch('projector',z,lambda:None)
            return dict(emb=Tensor(np.ones((1,1,192)),device))
    model=Model()
    def load(*a,**kw):
        holder.loads+=1
        if load_failure:raise RuntimeError('artificial load failure')
        assert kw['device'] is device;return model
    compat.load_compatible_pretrained=load
    @contextmanager
    def guard(model,torch,ledger,errors):holder.ledger=ledger;yield
    actualopen=Image.open
    def imageopen(path):return actualopen(io.BytesIO(store.files[str(Path(path).relative_to(store.path))]))
    modules={'torch':torch,'torch.nn':ModuleType('torch.nn'),'torch.nn.attention':attention,
             'experiments.checkpoint_compat':compat,'experiments.gpu_preflight':preflight}
    with patch.dict(sys.modules,modules),patch.object(enc,'guard',guard),patch.object(enc,'frozen',lambda _:None),\
         patch.object(enc,'tensor_hash',lambda _:enc.MODEL_TENSOR_SHA),patch.object(p0,'image_transform',lambda:None),\
         patch.object(p0,'_image_tensor',lambda *a,**kw:Tensor(np.zeros((1,1,3,224,224)),device)),\
         patch.object(Image,'open',imageopen):
        yield dict(model=model,device=device,check_limits=lambda:None),holder


def execute(*,base=ArrayEnv,p=None,d=None,store=None,compiler=None):
    fp,fd=fixture();p=fp if p is None else p;d=fd if d is None else d
    store=RAM() if store is None else store;compiler=FakeSuffix() if compiler is None else compiler;envs=[]
    def factory(meter):
        e=m.metered_class(base,meter)();envs.append(e);return e
    with visual_mock(store) as (runtime,holder),patch.object(m,'sha',store.hash),\
         patch.object(m.source,'sample_state',lambda *a:REQUEST.copy()):
        result=m.execute(p,d,store,m.Handler(compiler),factory,authorization=m.AUTH,runtime=runtime,
                         final_model_check=lambda:dict(model_unchanged=True),final_evaluator=lambda _:dict(matched=True))
    return result,store,envs,compiler


def test_complete_real_handler_routing_and_census():
    r,s,envs,c=execute()
    assert r['status']=='live_feedback_calibration_recorded',r['first_error']
    assert len(envs)==12 and len(c.requests)==8 and r['full_horizons']==12
    assert r['summary']==dict(native_success=0,completed10=12,denominator=12)
    assert r['counts']['physics_step']==dict(attempted=2448,completed=2448)
    assert r['counts']['reset']==dict(attempted=24,completed=24)
    assert r['counts']['set_state']==dict(attempted=48,completed=48)
    assert r['counts']['goal_setter']==dict(attempted=36,completed=36)
    assert r['counts']['goal_bind']==dict(attempted=12,completed=12)
    assert r['counts']['public_step']==dict(attempted=240,completed=240)
    assert all(r['counts'][k]==dict(attempted=v,completed=v) for k,v in m.contract()['neural_caps'].items())
    assert sum(x['PNG'] for x in r['rows'])==264
    assert sum(len(v) for v in s.files.values()) < m.CAP-m.FINAL
    assert not r['P1_admission'] and not r['P4_admission']
    for row in r['rows']:
        es=[e for e in s.events if e.get('episode_id')==row['episode_id']]
        seal=next(e['index'] for e in es if e['kind']=='suffix_sealed')
        six=next(e['index'] for e in es if e['kind']=='authorized_dispatch' and e['step']==6)
        assert row['encoded5_event_index'] < seal < six
    for p in c.requests:
        z=p['context3']['latent_history'][-1]
        assert z==[1.]*192 if p['mode']=='R' else z==[2.]*192
        assert '/RAM' not in json.dumps(p) and 'state' not in p and 'latent' not in p


@pytest.mark.parametrize('fault',['truncate','throw','substitute','render','guard'])
def test_physical_failure_or_stop_retains_all_denominators(fault):
    class Fault(ArrayEnv):
        def __init__(self):super().__init__(fault=fault,at=11)
    r,s,envs,c=execute(base=Fault)
    assert len(r['rows'])==12 and len(envs)==12 and all(len(e.commands)==11 for e in envs)
    assert not c.requests and not any(x['full_horizon'] for x in r['rows'])
    assert r['status']==('live_feedback_calibration_recorded' if fault=='truncate' else 'partial_or_failed')
    assert all(not x['native_success'] for x in r['rows'])


def test_encoder_failure_no_suffix_or_fallback():
    with patch.object(m.Handler,'encode_observation',side_effect=RuntimeError('encoder failure')):
        r,s,envs,c=execute()
    assert all(len(e.commands)==15 for e in envs) and not c.requests
    assert r['status']=='partial_or_failed' and len(r['rows'])==12


def test_suffix_failure_no_dispatch_no_old_fallback():
    c=FakeSuffix();c.fail=True;r,s,envs,_=execute(compiler=c)
    assert r['status']=='partial_or_failed'
    assert [len(e.commands) for e in envs]==[20,15,15]*4
    assert all('suffix failure' in x['first_error']['message'] for x in r['rows'] if x['mode']!='O')


def test_source_root_drift_rejected_before_control():
    p,d=fixture();d['references'][0]['history'][1,2,2,0]^=1
    r,s,envs,_=execute(p=p,d=d)
    assert all(len(e.commands)==10 for e in envs[:6]) and all(x['full_horizon'] for x in r['rows'][6:])


def test_constructor_implicit_reset_is_counted():
    class Constructor(ArrayEnv):
        def __init__(self):super().__init__();self.reset(seed=0)
    r,_,_,_=execute(base=Constructor)
    assert r['status']=='live_feedback_calibration_recorded',r['first_error']
    assert r['counts']['reset']['completed']==36 and r['counts']['physics_step']['completed']==2472


def test_dispatch_seal_same_episode_and_single_use():
    a=np.zeros((5,2),np.float32);seal=dict(episode_id='e',remaining_public_steps=5,commands_sha256=m.array_sha(a))
    seal['seal_sha256']=m.source.digest(seal)
    with pytest.raises(ValueError):m.DispatchGate('other',seal,a)
    gate=m.DispatchGate('e',seal,a)
    with pytest.raises(ValueError):gate.take('e',7)
    for i in range(6,11):assert np.array_equal(gate.take('e',i),a[i-6])
    with pytest.raises(ValueError):gate.take('e',11)


def test_missing_stage_visual_ledger_never_charged_as_compilation():
    s=RAM();meter=m.Meter(s);sink=m.Sink(s,meter,'e','encode5')
    ledger=enc.Ledger(1,sink.event);ledger.image=0
    tensor=NS(device=NS(type='cuda'),shape=(1,3,224,224))
    ledger.dispatch('encoder',tensor,lambda:None);sink.complete()
    assert meter.counts['encoder']==dict(attempted=1,completed=1)
    assert all(meter.counts[k]['attempted']==0 for k in ('predict','predictor','ad_forward','jacobian'))


def test_failed_event_attempt_never_refunded():
    s=RAM();meter=m.Meter(s);sink=m.Sink(s,meter,'e','suffix')
    sink.event(dict(kind='module_attempt',entry=dict(index=0,resource='predict',stage='jacobian')))
    sink.event(dict(kind='module_error',entry=dict(index=0,resource='predict',stage='jacobian')))
    assert meter.counts['predictor']==dict(attempted=1,completed=0)
    assert meter.counts['ad_forward']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):sink.complete()


@pytest.mark.parametrize('kind',['encoder','public_step','physics_step'])
def test_global_deadline_checked_before_every_dispatch(kind):
    s=RAM();called=[]
    def expired():raise ValueError('whole job load+execution expired')
    meter=m.Meter(s,expired)
    with pytest.raises(ValueError):meter.call(kind,lambda:called.append(1))
    assert not called and meter.counts[kind]['attempted']==0


@pytest.mark.parametrize('where',['file','directory'])
def test_actual_store_fsync_failure_propagates_without_disk(where,monkeypatch):
    store=object.__new__(m.Store);store.path=Path('/RAM_ONLY')
    class File(io.BytesIO):
        def fileno(self):return 11
    opened=[]
    monkeypatch.setattr(Path,'rglob',lambda *a:[])
    monkeypatch.setattr(Path,'is_symlink',lambda *a:False)
    monkeypatch.setattr(Path,'open',lambda *a,**kw:File())
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:NS(free=m.FLOOR+m.CAP))
    monkeypatch.setattr(m.os,'open',lambda path,flags:opened.append(path) or 12)
    monkeypatch.setattr(m.os,'close',lambda fd:None)
    def fsync(fd):
        if fd==(11 if where=='file' else 12):raise OSError('mock fsync '+where)
    monkeypatch.setattr(m.os,'fsync',fsync)
    with pytest.raises(OSError,match='mock fsync'):store.write('e_suffix/array.npy',b'array')
    if where=='directory':assert opened==[Path('/RAM_ONLY/e_suffix')]


@pytest.mark.parametrize('where',['report','journal','closed'])
def test_durable_failure_never_success(where):
    s=RAM();put,emit=s.put,s.emit
    def failput(name,value,final=False):
        if where=='report' and final:raise OSError('fsync report')
        return put(name,value,final)
    def failemit(kind,**kw):
        if (where=='journal' and kind=='run_finished') or (where=='closed' and kind=='episode_closed'):raise OSError('fsync journal')
        return emit(kind,**kw)
    s.put=failput;s.emit=failemit;r,_,_,_=execute(store=s)
    assert r['status']=='partial_or_failed'


def test_native_terminal_not_completion_is_success():
    row={};body=ArrayEnv();goal=np.array([100.,100.,150.,150.,0.,0.,0.])
    m.native_transition(body,goal,dict(state=goal),0.,True,False,row)
    assert row['native_success'] and row['terminated']
    with pytest.raises(ValueError):m.native_transition(body,goal,dict(state=goal),0.,False,False,{})


@pytest.mark.parametrize('load_failure',[False,True])
def test_real_run_entry_load_once_shared_handler_runtime(monkeypatch,load_failure):
    p,d=fixture();s=RAM();c=FakeSuffix();p.update(manifest='/manifest',backend_source='/suffix',backend_sha256=m.BACKEND_SHA,
                                             modelbinding=d['tasks'][0]['modelbinding'])
    manifest=dict(sourcepack='sp',source_prepared='pr',source_prepare_sha256='h',historical_physical='/old')
    envs=[]
    def factory(meter):e=m.metered_class(ArrayEnv,meter)();envs.append(e);return e
    def hashvalue(path):
        if str(path)=='/suffix':return m.BACKEND_SHA
        if str(path)=='/old':return m.PHYSICAL_SHA
        return s.hash(path)
    def finalrows(rows):
        # Historical evaluation is separate from compiler input; use same mock prefix.
        return dict(rows=[dict(root_slot=r['source_slot'],arm=r['prefix_arm'],checkpoints=r['checkpoints'])
                          for r in rows if r['mode']=='O'])
    read=m.read
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'Store',lambda _:s)
    monkeypatch.setattr(m.kernel,'load_source',lambda *a:deepcopy(d))
    monkeypatch.setattr(movingenc,'load_task',lambda path,h:dict(gn_task=d['tasks'][int(path[-1])]['gn_task']))
    monkeypatch.setattr(m,'sha',hashvalue);monkeypatch.setattr(m,'native_factory',factory)
    monkeypatch.setattr(m.source,'sample_state',lambda *a:REQUEST.copy())
    monkeypatch.setattr(m,'read',lambda path:manifest if str(path)=='/manifest' else finalrows(rows_holder['rows']))
    rows_holder={};original_execute=m.execute
    def runexecute(*a,**kw):
        original_evaluator=kw['final_evaluator']
        def evaluate(rows):rows_holder['rows']=rows;return original_evaluator(rows)
        kw['final_evaluator']=evaluate
        assert isinstance(a[3],m.Handler) and a[3].suffix is c
        assert a[1]['tasks'][0]['modelbinding']==p['modelbinding']
        return original_execute(*a,**kw)
    monkeypatch.setattr(m,'execute',runexecute)
    monkeypatch.setattr(m.importlib,'import_module',lambda name:c if name.endswith(m.BACKEND) else pytest.fail(name))
    with visual_mock(s,load_failure=load_failure) as (runtime,holder):
        if load_failure:
            with pytest.raises(RuntimeError,match='load failure'):m.run('/prepare','hash','/out',authorization=m.AUTH)
            assert not envs and 'startup_failure.json' in s.files
        else:
            r=m.run('/prepare','hash','/out',authorization=m.AUTH)
            assert r['status']=='live_feedback_calibration_recorded',r['first_error']
            assert len(envs)==12 and c.models==[runtime['model']]*8
            assert r['wall_seconds']>=r['execution_wall_seconds']
            assert 'model loading' in r['wall_scope']
        assert holder.loads==1
