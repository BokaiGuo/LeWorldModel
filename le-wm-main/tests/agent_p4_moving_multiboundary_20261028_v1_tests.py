"""RAM fake arrays/environments only. No torch model, CUDA or native physics."""
from copy import deepcopy
from types import SimpleNamespace as NS,ModuleType
from contextlib import nullcontext
import sys
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p4_moving_multiboundary_20261028_v1 as m
import test_p1_moving_live_feedback_physical_20261027_v1 as h
import test_p1_moving_live_suffix_cuda_20261027_v1 as math_helpers


@pytest.fixture(scope='module')
def model_metadata():return m.suffix.modelbinding()


@pytest.fixture(autouse=True)
def cached_metadata(monkeypatch,model_metadata):
    # Static hashes only; never read or deserialize model weights.
    monkeypatch.setattr(m.suffix,'modelbinding',lambda:deepcopy(model_metadata))
    monkeypatch.setattr(m.suffix.shutil,'disk_usage',lambda p:NS(free=m.old.FLOOR+10*m.old.CAP))


class RAM(h.RAM):
    def png_bytes(self,receipt):
        assert self.hash(receipt['path'])==receipt['file_sha256']
        return self.files[str(__import__('pathlib').Path(receipt['path']).relative_to(self.path))]


class FakeRolling:
    def __init__(self,fail_at=None):self.requests=[];self.fail_at=fail_at
    def compile_suffix(self,p,sink,**runtime):
        m.validate_request(p);self.requests.append(deepcopy(p))
        if p['boundary']==self.fail_at:raise RuntimeError('RAM compile failure')
        def pair(k1,k2,e):
            sink.event(dict(kind=k1,entry=dict(e,completed=False)))
            sink.event(dict(kind=k2,entry=dict(e,completed=True)))
        for i,stage in enumerate(m.F.STAGES):
            pair('predict_API_attempt','predict_API_return',dict(index=i,stage=stage))
            for j,resource in enumerate(('action_encoder','predict','pred_proj')):
                pair('module_attempt','module_return',dict(index=i*3+j,stage=stage,resource=resource))
        for i,resource in enumerate(('jacobian','svd')):
            e=dict(index=i,resource=resource)
            if i==0:e.update(shape=[192,10],mathematical_VJP=192)
            pair('math_attempt','math_return',e)
        for i,resource in enumerate(('jacobian','grad')):
            pair('AD_API_attempt','AD_API_completed',dict(index=i,resource=resource))
        commands=np.full((5,2),.075 if p['mode']=='R' else -.075,np.float32).tolist()
        successor=m.roll(p['context3'],np.full(192,3+p['boundary'],np.float32),
            np.concatenate((np.asarray(p['actualrawhistory10'],np.float32)[5:],commands)),p['modelbinding'])
        return dict(dispatch_raw5=commands,predicted_successor=successor,artifact_sha256={})
    def verify_suffix_seal(self,p,r):return r['dispatch_raw5']


def execute(*,base=h.ArrayEnv,compiler=None,store=None):
    p,data=h.fixture();p['contract']=m.contract();store=store or RAM();compiler=compiler or FakeRolling();envs=[]
    def factory(meter):e=m.old.metered_class(base,meter)();envs.append(e);return e
    with h.visual_mock(store) as (runtime,holder),patch.object(m.old,'sha',store.hash),patch.object(m,'sha',store.hash),\
         patch.object(m.source,'sample_state',lambda *a:h.REQUEST.copy()):
        result=m.execute(p,data,store,m.old.Handler(compiler),factory,authorization=m.AUTH,runtime=runtime,
            final_model_check=lambda:dict(model_unchanged=True))
    return result,store,envs,compiler


def test_real_Handler_three_boundaries_20_steps_equal_execution_budget():
    r,s,envs,c=execute()
    assert r['status']=='known_moving_multiboundary_recorded',r['errors']
    assert len(envs)==8 and len(c.requests)==24 and len(r['compile_slots'])==24 and r['completed20']==8
    assert all(len(e.commands)==30 and len(e.resets)==2 and e.closed for e in envs)
    assert r['counts']['public_step']==dict(attempted=240,completed=240)
    assert r['counts']['physics_step']==dict(attempted=2432,completed=2432)
    assert all(r['counts'][k]==dict(attempted=v,completed=v) for k,v in m.contract()['neural_caps'].items())
    assert all([x['step'] for x in row['encodings']]==[5,10,15,20] for row in r['rows'])
    assert all(x['status']=='sealed' for x in r['compile_slots'])
    assert r['independent_tasks']==2 and r['fixed_denominator']==8 and not r['P4_admission']
    for row in r['rows']:
        events=[e for e in s.events if e.get('episode_id')==row['episode_id']]
        for b in m.BOUNDARIES:
            enc=next(e['index'] for e in events if e['kind']=='boundary_encoded' and e['actual_step']==b)
            seal=next(e['index'] for e in events if e['kind']=='boundary_sealed' and e['boundary']==b)
            first=next(e['index'] for e in events if e['kind']=='authorized_dispatch' and e['step']==b+1)
            assert enc<seal<first
    assert sum(e['kind']=='boundary_encoded' for e in s.events)==32


def test_history_is_own_rolling_raw_and_I_never_observed_latents():
    r,s,envs,c=execute()
    for p in c.requests:
        row=next(x for x in r['rows'] if x['episode_id']==p['episode_id']);step=p['boundary']
        raw=np.concatenate((m.old.kernel.moving.warmup_commands(),np.asarray(row['executed_raw'],np.float32)[:step]))[-10:]
        assert np.array_equal(raw,p['actualrawhistory10'])
        assert p['context3']['action_history']==m.normalized(p['modelbinding'],raw)
        z=p['context3']['latent_history'][-1]
        assert z==[1.]*192 if p['mode']=='R' else z==[2. if step==5 else 3+step-5]*192
        assert not {'state','snapshot','env','success','goal7','witness'} & p.keys()
        if step>5:assert not np.array_equal(raw[:5],m.F.planner.fixed_raw()[5:])
    # At15 R's full history consists of its own three encoded boundaries.
    assert all(p['context3']['latent_history']==[[1.]*192]*3 for p in c.requests if p['mode']=='R' and p['boundary']==15)


@pytest.mark.parametrize('boundary',[5,10,15])
def test_compile_failure_no_old_action_fallback_fixed_failure_denominator(boundary):
    r,s,envs,c=execute(compiler=FakeRolling(fail_at=boundary))
    assert r['status']=='partial_or_failed' and len(r['rows'])==8 and len(r['compile_slots'])==24
    assert all(len(e.commands)==10+boundary for e in envs)
    assert all(x['native_success'] is False for x in r['rows'])
    assert sum(x['status']=='failed_pending' for x in r['compile_slots'])==8
    assert sum(x['status']=='not_reached' for x in r['compile_slots'])==8*(15-boundary)//5


@pytest.mark.parametrize('fault',['truncate','throw','substitute','guard','render'])
def test_physical_abort_never_advances_partial_packet(fault):
    class Broken(h.ArrayEnv):
        def __init__(self):super().__init__(fault=fault,at=18)
    r,s,envs,c=execute(base=Broken)
    assert len(r['rows'])==8 and len(c.requests)==8 and all(len(e.commands)==18 for e in envs)
    assert all(x['status']=='not_reached' for x in r['compile_slots'] if x['boundary']>5)
    assert not any(row['full_horizon'] for row in r['rows'])


def test_encoder_failure_at10_stops_before_second_compile():
    encode=m.old.Handler.encode_observation
    def broken(self,p,sink,**kw):
        if '_b10' in p['episode_id']:raise RuntimeError('RAM encoder failed at actual10')
        return encode(self,p,sink,**kw)
    with patch.object(m.old.Handler,'encode_observation',broken):r,s,envs,c=execute()
    assert len(c.requests)==8 and all(len(e.commands)==20 for e in envs)
    assert r['status']=='partial_or_failed'


def request():
    p=h.fixture()[0];task=h.fixture()[1]['tasks'][0];raw10=np.full((10,2),.05,np.float32)
    context=m.roll(task['gn_task']['root_context'],np.ones(192,np.float32),raw10,task['gn_task'])
    return dict(schema=m.NAME+'.compile',episode_id=m.episode_id(0,m.old.ARMS[0],'R'),source_slot=0,
        prefix_arm=m.old.ARMS[0],mode='R',boundary=15,context3=context,actualrawhistory10=raw10.tolist(),
        diagnosticraw5=p['layers'][0]['commands'][5:],goal=task['gn_task']['goal'],modelbinding=task['modelbinding'],
        observation=dict(boundary=15,png_sha256='a'*64,pixel_sha256='b'*64,encoding_sha256='c'*64,
            observed_index=10,encoded_index=11,observed_latent_sha256=m.old.packet_digest([1.]*192)))


@pytest.mark.parametrize('damage',['state','success','future_rgb','raw','tokens','precision','boundary','sequence','latent','model'])
def test_new_whitelist_rejects_privilege_and_history_drift(damage):
    p=request()
    if damage in ('state','success','future_rgb'):p[damage]=[0]
    elif damage=='raw':p['actualrawhistory10'][0][0]=.1
    elif damage=='tokens':p['context3']['action_history'][0][0]=999
    elif damage=='precision':p['context3']['latent_history'][0][0]+=1e-12
    elif damage=='boundary':p['boundary']=20
    elif damage=='sequence':p['observation']['encoded_index']=10
    elif damage=='latent':p['observation']['observed_latent_sha256']='d'*64
    else:p['modelbinding']['action_mean'][0]=0
    with pytest.raises(ValueError):m.validate_request(p)


def test_real_frozen_math_and_module_array_verifier_accept_rolling_history():
    p=request();m.validate_request(p);b=m.branch_input(p)
    store=math_helpers.RAM();sink=m.suffix.Sink(store,lambda:None)
    # Runs frozen evaluate with NumPy arithmetic / hook stand-ins only.
    row=dict(index=0,branch_id=b['branch_id']);math_helpers.runner()(b,sink,row)
    m.F.check_result(b,row);m.F.saved_arrays(b,row,sink.load)
    response=dict(schema=m.NAME+'.compiled',request_sha256=m.old.packet_digest(p),row=row,
        model_hash_before=m.suffix.MODEL_SHA,model_hash_after=m.suffix.MODEL_SHA,source_verified=True,
        dispatch_raw5=row['new_suffix_commands'],predicted_successor=row['evaluations'][1]['context'],artifact_sha256={})
    response['release_sha256']=m.old.packet_digest(response)
    assert np.array_equal(m.verify_compile(p,response),row['new_suffix_commands'])
    assert row['predict_API_resources']['predict']==dict(attempted=5,completed=5)
    assert row['AD_API_resources']['grad']==dict(attempted=1,completed=1)
    bad=deepcopy(response);bad['predicted_successor']['latent_history'][-1][0]+=.5
    bad['release_sha256']=m.old.packet_digest({k:v for k,v in bad.items() if k!='release_sha256'})
    with pytest.raises(ValueError):m.verify_compile(p,bad)


def test_budget_gate_before_native_dispatch():
    store=RAM();meter=m.Meter(store);meter.counts['public_step']['attempted']=240;called=[]
    with pytest.raises(ValueError):meter.call('public_step',lambda:called.append(1))
    assert called==[]


def test_startup_no_model_without_MAIN_authorization(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no verification/loader allowed'))
    with pytest.raises(ValueError):m.run('/none','a'*64,'/none',authorization='worker')


def test_prepare_contract_fixed_before_any_runtime():
    c=m.contract();rows,slots=m.roster()
    assert len(rows)==8 and len(slots)==24 and c['feedback_boundaries']==[5,10,15]
    assert c['flops'] is c['device_time_ns'] is c['inherited_compute'] is None
    assert all(r['native_success'] is None for r in rows)
    assert c['per_completed_episode']['control_public_steps']==20
    assert not c['P1_admission'] and not c['P4_admission']


def test_production_RollingSuffix_entry_calls_frozen_math_with_RAM_interfaces(monkeypatch):
    """Exercise production method itself, not a replacement callback runner."""
    p=request();b=m.branch_input(p);holder={};device=NS(type='cuda')
    model=NS(predict=lambda *a,**kw:None,parameters=lambda:[],buffers=lambda:[])
    class Modules(m.F.Modules):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);holder['modules']=self
        def __enter__(self):return self  # fake modules are directly driven by NumPy backend
    class AD(m.F.SuffixAD):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);holder['ad']=self
        def __enter__(self):return self
        def __exit__(self,*a):return False
    class Backend(math_helpers.h.Backend):
        def __init__(self,model,device):super().__init__(b,holder['modules'])
        def step(self,context,commands):model.predict();return super().step(context,commands)
        def jacobian(self,context,operation):
            ad=holder['ad']
            def calculate():
                def forward(**kw):
                    self.nominal(context)
                    return ad.dispatch('grad',lambda **kw:self.j.copy(),(),dict(is_grads_batched=True))
                return ad.dispatch('jacobian',forward,(),dict(vectorize=True))
            return operation('jacobian',calculate)
    torch=ModuleType('torch');torch.cuda=NS(is_available=lambda:True,synchronize=lambda device:None)
    torch.is_inference_mode_enabled=lambda:False;torch.enable_grad=nullcontext;torch.autocast=lambda **kw:nullcontext()
    torch.backends=NS(cuda=NS(matmul=NS(allow_tf32=True)),cudnn=NS(allow_tf32=True))
    attention=ModuleType('torch.nn.attention');attention.SDPBackend=NS(MATH='RAM');attention.sdpa_kernel=lambda _:nullcontext()
    monkeypatch.setattr(m.F,'Modules',Modules);monkeypatch.setattr(m.F,'SuffixAD',AD)
    monkeypatch.setattr(m.F.d2p,'TorchBackend',Backend)
    monkeypatch.setattr(m.F.observed.enc,'frozen',lambda model:None)
    monkeypatch.setattr(m.F.observed.enc,'tensor_hash',lambda model:m.suffix.MODEL_SHA)
    store=math_helpers.RAM()
    with patch.dict(sys.modules,{'torch':torch,'torch.nn':ModuleType('torch.nn'),'torch.nn.attention':attention}):
        response=m.RollingSuffix().compile_suffix(p,store,model=model,device=device,check_limits=lambda:None)
    assert np.array_equal(m.verify_compile(p,response),response['dispatch_raw5'])
    assert 'receipt.json.gz' in store.files and response['row']['predict_API_resources']['predict']['completed']==5
    assert torch.backends.cuda.matmul.allow_tf32 is True and torch.backends.cudnn.allow_tf32 is True
    assert 'predict' not in model.__dict__ or model.predict is not None


def test_real_run_entry_loads_once_and_connects_concrete_Handler_factory(monkeypatch):
    p,data=h.fixture();p.update(contract=m.contract(),original_manifest='/RAM_manifest',
        modelbinding=data['tasks'][0]['modelbinding'])
    store=RAM();compiler=FakeRolling();envs=[]
    manifest=dict(sourcepack='RAM_pack',source_prepared='RAM_prepare',source_prepare_sha256='a'*64)
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'Store',lambda path:store)
    monkeypatch.setattr(m,'RollingSuffix',lambda:compiler);monkeypatch.setattr(m,'read',lambda path:manifest)
    monkeypatch.setattr(m,'sha',store.hash);monkeypatch.setattr(m.old,'sha',store.hash)
    monkeypatch.setattr(m.old.kernel,'load_source',lambda *a:deepcopy(data))
    monkeypatch.setattr(m.source,'sample_state',lambda *a:h.REQUEST.copy())
    monkeypatch.setattr(h.movingenc,'load_task',lambda path,hsh:dict(gn_task=data['tasks'][int(path[-1])]['gn_task']))
    def factory(meter):e=m.old.metered_class(h.ArrayEnv,meter)();envs.append(e);return e
    monkeypatch.setattr(m.old,'native_factory',factory)
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    with h.visual_mock(store) as (runtime,holder):
        r=m.run('/RAM_prepare','a'*64,str(m.ROOT/'results'/(m.NAME+'_RAM')),authorization=m.AUTH)
    assert r['status']=='known_moving_multiboundary_recorded',r['errors']
    assert holder.loads==1 and len(envs)==8 and len(compiler.requests)==24
    assert 'startup_roster.json' in store.files


def test_prepare_reconstructs_new_contract_without_calling_runtime(monkeypatch):
    p,data=h.fixture();p.update(modelbinding=data['tasks'][0]['modelbinding'],manifest='/RAM_manifest')
    monkeypatch.setattr(m.old,'verify',lambda path,hsh:deepcopy(p))
    original=m.sha
    def hashed(path):
        if str(path)==str(m.OLD_PREPARE):return m.OLD_SHA
        return original(path)
    monkeypatch.setattr(m,'sha',hashed)
    built=m.build()
    assert len(built['slots'])==24 and built['contract']['control_public_steps']==20
    assert built['layers']==p['layers'] and not built['scientific_pass']
