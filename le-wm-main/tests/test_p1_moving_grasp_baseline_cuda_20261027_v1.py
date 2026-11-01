"""RAM and non-Module scalar/tensor algebra only; no models/CUDA/physics."""
from copy import deepcopy
from dataclasses import asdict
import gzip
from pathlib import Path
from types import SimpleNamespace
import zlib
import numpy as np
import pytest
import torch
from experiments.spectral_reachability import p1_moving_grasp_baseline_cuda_20261027_v1 as m

SMALL=m.Config(steps=4,sync_interval=2,sync_steps=1)


def task(slot=0):
    rng=np.random.default_rng(37+slot);mean=m.base.d2p.ACTION_MEAN.astype(float).tolist();std=m.base.d2p.ACTION_STD.astype(float).tolist()
    return dict(environment='analytic_fixture_only',task_contract_sha256='1'*64,model_sha256='2'*64,
        model_config_sha256='3'*64,metric_source_sha256='4'*64,goal_source_sha256='5'*64,
        representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=mean,action_std=std,packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only',
        goal=rng.normal(size=192).astype(np.float32).astype(float).tolist(),projection=np.eye(192).tolist(),
        root_raw_action_history=m.binding.moving.fixed_raw().astype(float).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=rng.normal(size=(3,192)).astype(np.float32).astype(float).tolist(),
            action_history=m.core.normalized_tokens(dict(action_mean=mean,action_std=std),m.binding.moving.fixed_raw()).astype(float).tolist()))


class Algebra:
    """Explicit arithmetic, not nn.Module/learned parameters/neural kernels."""
    def __init__(self,t):self.task=t;self.index=0
    def __call__(self,z,h,raw,meter):
        mean=z.new_tensor(self.task['action_mean']).repeat(5);std=z.new_tensor(self.task['action_std']).repeat(5)
        token=meter.call('action_encoder',lambda:(raw.reshape(10)-mean)/std)
        i=self.index;self.index+=1;meter.emit('predict_API_attempt',index=i)
        meter.emit('native_context_attempt',index=i,context=dict(latent_history=m.grasp.host(z[0]),
            action_history=m.grasp.host(torch.cat((h,token[None,None]),1)[0])))
        # Include genuine between-state action history in this arithmetic double.
        pred=meter.call('predict',lambda:2*z[:,-1]+3*raw.sum()+.01*h[:,-1].sum())
        meter.emit('predict_API_completed',index=i)
        return torch.cat((z[:,1:],pred[:,None]),1),torch.cat((h[:,1:],token[None,None]),1)


def dynamics(t):return m.Dynamics(t,torch.device('cpu'),Algebra(t),analytic_fixture=True)


class RAMStore(m.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False;self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert append or name not in self.files
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        self.files[name]=self.files.get(name,b'')+raw


@pytest.fixture(scope='module')
def small():
    t=task();events=[];meter=m.Meter(lambda e:events.append(deepcopy(e)),cfg=SMALL);d=dynamics(t)
    r=m.optimize(t,0,d,meter,cfg=SMALL,analytic_fixture=True)
    return t,r,events


def test_exact_original_parameters_math_and_horizon():
    assert m.Config is m.grasp.Config and m.optimize is m.grasp.optimize and m.Dynamics is m.grasp.Dynamics
    assert m.choice is m.grasp.choice and m.journal_check is m.grasp.journal_check
    assert asdict(m.Config())==dict(steps=750,lr_s=.1,lr_a=.005,state_noise=.005,sync_interval=50,sync_steps=10,sync_lr=.05)
    assert m.budget()==dict(predict=1802,action_encoder=1802,backward=900)
    c=m.contract();assert c['total_budget']==dict(predict=3604,action_encoder=3604,backward=1800)
    assert c['horizon_semantics']==dict(observed_history_frames=3,imagined_future_packets=2,imagined_future_public_steps=10,released_commands=10,physical_steps_this_producer=0,third_future_packet=False)
    assert not c['official_equivalence'] and not c['cross_contract_reported_comparison'] and not c['P1_admission']
    assert c['output_cap_bytes']==64*1024**2 and c['reserve_bytes']==200*1024**2


def test_actual_nonzero_history_lifted_sync_and_tokens(small):
    t,r,es=small;m.journal_check(t,r,es,SMALL);v=m.history_check(t,es,SMALL)
    assert v['native_contexts']==14 and v['all_nonzero_history_paths_verified']
    for e in es:
        if e['kind']=='transition_attempt' and e['packet']==0:
            assert e['latent_history'][0]==t['root_context']['latent_history']
            assert e['action_history'][0]==t['root_context']['action_history']
    m.choice(t,r,0,SMALL)
    init=next(e for e in es if e['kind']=='initialized')
    assert not np.asarray(init['logits']).any()
    assert np.asarray(t['root_raw_action_history']).any()
    assert [e['after_iteration'] for e in es if e['kind']=='joint_optimizer_reset']==[2,4]


def test_state_stopgrad_and_live_action_history():
    t=task();d=dynamics(t);v=torch.ones((1,192),requires_grad=True);x=torch.zeros((2,10),requires_grad=True)
    loss,_,_=d.soft(v,x,m.Meter(lambda e:None,cfg=SMALL));dv,dx=torch.autograd.grad(loss,(v,x))
    p0=2*d.z[:,-1]+.01*d.h[:,-1].sum()
    assert torch.allclose(dv,2*(v-p0)/192)
    assert torch.isfinite(dx).all() and dx[0].abs().sum()>0 and dx[1].abs().sum()>0
    # Serial sync has state-chain derivatives unlike lifted state-input stopgrad.
    x2=torch.zeros((2,10),requires_grad=True);terminal,_=d.serial(x2,m.Meter(lambda e:None,cfg=SMALL))
    g=torch.autograd.grad(terminal.sum(),x2)[0]
    assert g[0].abs().sum()>g[1].abs().sum()


@pytest.mark.parametrize('damage',['root_token','zero','AD_latent','missing','API','order','sync'])
def test_saved_actual_context_tamper_rejected(small,damage):
    t,r,es=small;es=deepcopy(es);tr=[e for e in es if e['kind']=='native_context_attempt']
    if damage=='root_token':tr[0]['context']['action_history'][0][0]+=.1
    elif damage=='zero':tr[0]['context']['action_history'][:2]=np.zeros((2,10)).tolist()
    elif damage=='AD_latent':tr[1]['context']['latent_history'][-1][0]+=.1
    elif damage=='missing':es.remove(tr[1])
    elif damage=='API':next(e for e in es if e['kind']=='predict_API_completed')['index']=4
    elif damage=='order':
        i=next(i for i,e in enumerate(es) if e['kind']=='predict_API_attempt');es.insert(i,es.pop(i+1))
    else:next(e for e in es if e['kind']=='transition_attempt' and e['evaluation_id'].startswith('sync'))['action_history'][0][0][0]+=.1
    with pytest.raises(ValueError):m.history_check(t,es,SMALL)


@pytest.mark.parametrize('damage',['history','freeze','reset','resources','commands','schedule'])
def test_original_protocol_choice_rejects_damage(small,damage):
    t,r,es=small;r=deepcopy(r);es=deepcopy(es)
    if damage=='history':next(e for e in es if e['kind']=='transition_completed')['context']['action_history'][0][0]+=.1
    elif damage=='freeze':
        i=next(i for i,e in enumerate(es) if e['kind']=='actions_frozen_before_verification');es.insert(0,es.pop(i))
    elif damage=='reset':es.pop(next(i for i,e in enumerate(es) if e['kind']=='joint_optimizer_reset'))
    elif damage=='resources':r['resources']['backward']['completed']-=1
    elif damage=='commands':r['commands'][0][0]+=.1
    else:r['sync_after_iterations']=[]
    with pytest.raises(ValueError):
        m.choice(t,r,0,SMALL);m.journal_check(t,r,es,SMALL)


def test_api_cap_partial_and_restore():
    model=SimpleNamespace(predict=lambda:123);events=[];original=model.predict
    with m.api_trace(model,events.append) as entries:
        for _ in range(1802):assert model.predict()==123
        with pytest.raises(ValueError,match='cap'):model.predict()
    assert model.predict is original and len(entries)==1802 and all(e['completed'] for e in entries)
    def fail():raise RuntimeError('native')
    model.predict=fail;events=[]
    with pytest.raises(RuntimeError),m.api_trace(model,events.append) as partial:model.predict()
    assert partial==[dict(index=0,completed=False)] and events==[dict(kind='predict_API_attempt',index=0)]
    assert model.predict is fail


def test_no_CPU_neural_fallback():
    t=task()
    with pytest.raises(ValueError,match='CPU neural'):m.Dynamics(t,torch.device('cpu'),Algebra(t))
    with pytest.raises(ValueError,match='CUDA only'):m.grasp.gn.screen.NativeCuda(None,SimpleNamespace(type='cpu'),t)


def test_nonrefundable_attempt_on_sync_failure():
    def fail():raise RuntimeError('async mock')
    meter=m.Meter(lambda e:None,sync=fail);token=meter.begin('backward')
    with pytest.raises(RuntimeError):meter.end(token)
    assert meter.counts()['backward']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.begin('backward')


@pytest.mark.parametrize('which',['wall','memory','disk'])
def test_caps(which):
    tick=[0];l=m.Limits(free=lambda:m.FLOOR-1 if which=='disk' else m.FLOOR,
        clock=lambda:tick[0],memory=lambda:(3221225473,0) if which=='memory' else (0,0))
    if which=='wall':tick[0]=600
    with pytest.raises(ValueError):l.check()


def test_run_real_first_disk_probe(monkeypatch):
    paths=[];monkeypatch.setattr(m,'verify',lambda *a:{})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:paths.append(p) or SimpleNamespace(free=m.FLOOR-1))
    monkeypatch.setattr(m,'Store',lambda p:pytest.fail('no writes below floor'))
    with pytest.raises(ValueError,match='floor'):m.run('unused','unused','/tmp/not_written')
    assert paths==[Path('/tmp')]


def test_first_error_preserved_no_model_import(monkeypatch):
    class Broken:
        output=SimpleNamespace(iterdir=lambda:[])
        def put(self,name,value,**kw):
            if name=='binding.json':raise OSError('original failure')
            self.report=value
        def event(self,e):return None
        def close(self):raise OSError('close failure')
    store=Broken();slots=[dict(source_slot=s,blind_sha256=str(s)*64,task_id=str(s+2)*64) for s in (0,1)]
    monkeypatch.setattr(m,'verify',lambda *a:dict(slots=slots,contract=m.contract()));monkeypatch.setattr(m,'Store',lambda p:store)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='partial moving GRASP'):m.run('unused','unused','/tmp/not_written')
    r=store.report;assert r['first_error']['message']=='original failure' and r['errors'][1]['message']=='close failure'
    assert not r['commands_released'] and len(r['rows'])==2 and all(row['status']=='not_started' for row in r['rows'])


def test_actual_saved_RAM_build_and_own_loader(monkeypatch):
    monkeypatch.setattr(m.base,'load_export',lambda *a:pytest.fail('old zero loader forbidden'))
    monkeypatch.setattr(m.grasp,'build',lambda *a:pytest.fail('old zero build forbidden'))
    b=m.binding;p=m.build(task_exports=b.moving.EXPORTS,task_shas=b.moving.EXPORT_SHAS,loader_sha=b.moving.LOADER_SHA,
        encoder_report=str(b.ENCODER),encoder_sha=b.ENCODER_SHA,sourcepack=str(b.SOURCE),source_sha=b.SOURCE_SHA)
    assert p['schema']==m.NAME+'.prepare' and len(m.canonical(p))<=256*1024 and len(p['slots'])==2
    assert not any('certificate' in path or path.endswith('.npz') for path in p['bindings_sha256'])
    for s in p['slots']:
        t=m.load_task(s['task_export'],s['task_sha256']);np.testing.assert_array_equal(t['root_raw_action_history'],b.moving.fixed_raw())


def test_full_default_two_root_RAM_producer_consumer(monkeypatch):
    """750 joint / 15*10 sync per root, only analytical non-Module arithmetic."""
    store=RAMStore();tasks={};slots=[];rows=[]
    for s in (0,1):
        t=task(s);path=f'moving_algebra_{s}';tasks[path]=t
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=path,task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        rows.append(dict(source_slot=s,arm=m.ARM,task_id=m.core.task_id(t),blind_sha256=str(s+2)*64,status='not_started',commands=None,resources={},predict_API_resources={}))
    monkeypatch.setattr(m,'load_task',lambda path,h:tasks[path]);p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    def runner(t,s,sink,row):
        meter=m.Meter(sink);result=m.optimize(t,s,dynamics(t),meter,analytic_fixture=True)
        row['resources']=meter.counts();row['predict_API_resources']=dict(predict=dict(attempted=1802,completed=1802));return result
    store.put('binding.json',dict(prepared='ram_prepare',prepare_sha256='a'*64));m.plan_all(p,store,runner,rows);store.close()
    digest=m.grasp.enc.digest
    report=dict(schema=m.NAME+'.report',status='moving_grasp_model_plans_sealed_not_admitted',commands_released=True,errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.grasp.enc.MODEL_TENSOR_SHA,model_hash_after=m.grasp.enc.MODEL_TENSOR_SHA,
        prepared='ram_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,
        resources={k:dict(attempted=v,completed=v) for k,v in m.contract()['total_budget'].items()},predict_API_resources=dict(predict=dict(attempted=3604,completed=3604)),
        limits=dict(checks=123,minimum_observed_free_bytes=m.FLOOR,peak_allocated_bytes=1000,peak_reserved_bytes=1000,elapsed_seconds=123),
        P1_admission=False,training=False,official_equivalence=False,new_images=0,new_physics=0,artifact_sha256={n:digest(b) for n,b in store.files.items()})
    store.put('report.json',report,final=True);expected=digest(store.files['report.json']);prefix=Path('/RAM_moving_grasp')
    actual_sha,actual_read=m.sha,m.read;actual_iter,actual_bytes,actual_stat=Path.iterdir,Path.read_bytes,Path.stat
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m,'sha',lambda path:digest(store.files[Path(path).name]) if Path(path).parent==prefix else actual_sha(path))
    def ramread(path):
        path=Path(path)
        if path.parent!=prefix:return actual_read(path)
        raw=store.files[path.name];return m.base.decode(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'read',ramread)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(prefix/n for n in store.files) if path==prefix else actual_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==prefix else actual_bytes(path))
    monkeypatch.setattr(Path,'stat',lambda path,*a,**kw:SimpleNamespace(st_size=len(store.files[path.name])) if path.parent==prefix else actual_stat(path,*a,**kw))
    out=m.consume(prefix,expected);assert len(out['rows'])==2 and out['P1_admission'] is False
    assert all(len(plan['commands'])==10 and plan['schema']==m.NAME+'.model_plan' for plan in out['rows'])
    print('Default two-moving-root RAM all-artifact bytes',sum(map(len,store.files.values())))
    store.files[rows[0]['plan_file']]+=b' '
    with pytest.raises(ValueError,match='artifact SHA'):m.consume(prefix,expected)
