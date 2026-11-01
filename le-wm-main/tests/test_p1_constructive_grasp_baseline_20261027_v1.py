"""Non-Module analytic tensor algebra/RAM only. Never models/CUDA/physics."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
from pathlib import Path
import zlib
import numpy as np
import pytest
import torch
from experiments.spectral_reachability import p1_constructive_grasp_baseline_20261027_v1 as m

SMALL=m.Config(steps=4,sync_interval=2,sync_steps=1)


def task(slot=0):
    rng=np.random.default_rng(37+slot)
    mean=[-.0087,.0068];std=[.2019,.2002]
    return dict(environment='analytic_fixture_only',task_contract_sha256='1'*64,model_sha256='2'*64,
        model_config_sha256='3'*64,metric_source_sha256='4'*64,goal_source_sha256='5'*64,
        representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=mean,action_std=std,packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only',
        goal=rng.normal(size=192).tolist(),projection=np.eye(192).tolist(),root_raw_action_history=np.zeros((10,2)).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=rng.normal(size=(3,192)).astype(np.float32).astype(float).tolist(),
            action_history=m.core.normalized_tokens(dict(action_mean=mean,action_std=std),np.zeros((10,2))).astype(float).tolist()))


class Algebra:
    """No Module, learned parameters, checkpoints or neural operators."""
    def __init__(self,t):self.task=t;self.calls=[]
    def __call__(self,z,h,raw,meter):
        self.calls.append((z.detach().clone(),h.detach().clone(),raw.detach().clone()))
        mean=z.new_tensor(self.task['action_mean']).repeat(5);std=z.new_tensor(self.task['action_std']).repeat(5)
        token=meter.call('action_encoder',lambda:(raw.reshape(10)-mean)/std)
        pred=meter.call('predict',lambda:2*z[:,-1]+3*raw.sum())
        return torch.cat((z[:,1:],pred[:,None]),1),torch.cat((h[:,1:],token[None,None]),1)


def dynamics(t,callback=None):
    return m.Dynamics(t,torch.device('cpu'),Algebra(t) if callback is None else callback,analytic_fixture=True)


class RAMStore(m.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert append or name not in self.files
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        self.files[name]=self.files.get(name,b'')+raw


@pytest.fixture(scope='module')
def small():
    t=task();events=[];meter=m.Meter(events.append,cfg=SMALL);d=dynamics(t)
    r=m.optimize(t,0,d,meter,cfg=SMALL,analytic_fixture=True)
    return t,r,events,meter,d


def test_official_assets_and_version_git_blobs():
    pins=m.official_bindings()
    assert len(pins)==8 and m.REVISIONS['lewm__']=='94be3c42aa2f0867c6cb1e38bda498b293b6921e'
    text=(m.OFFICIAL/'lewm__grasp_solver.py.source.txt').read_text()
    for literal in ('s_t = s_full[:, t].detach()','loss = loss_dyn + cur_goal_weight * loss_goal',
        's_virtual.data += cur_noise * torch.randn_like(s_virtual)',
        '(k + 1) % self.gd_interval == 0 and k > 0','sync_optimizer = torch.optim.Adam'):
        assert literal in text
    assert (m.OFFICIAL/'LICENSE.source.txt').read_text().startswith('MIT License')


def test_official_pin_drift_fails(monkeypatch):
    monkeypatch.setattr(m,'PINS',{**m.PINS,'lewm__grasp_solver.py.source.txt':'0'*64})
    with pytest.raises(ValueError,match='official asset SHA'):m.official_bindings()


def test_budget_and_honest_variant_contract():
    assert m.budget()==dict(predict=1802,action_encoder=1802,backward=900)
    c=m.contract();assert c['total_budget']==dict(predict=3604,action_encoder=3604,backward=1800)
    assert m.schedule(m.Config())==list(range(50,751,50))
    assert c['official_equivalence'] is False and c['full_pipeline_equal_compute'] is False
    assert c['forward_count_matches_D2P'] is False and c['P1_admission'] is False
    assert c['config']==asdict(m.Config())


def test_h3_soft_detach_target_and_action_gradients_not_plain_GD():
    t=task();t['root_context']['latent_history']=np.ones((3,192)).tolist();t['goal']=np.full(192,4.).tolist()
    d=dynamics(t);v=torch.full((1,192),3.,requires_grad=True);x=torch.zeros((2,10),requires_grad=True)
    loss,dyn,goal=d.soft(v,x,m.Meter(lambda e:None))
    assert [loss.item(),dyn.item(),goal.item()]==pytest.approx([13.,5.,8.])
    dv,dx=torch.autograd.grad(loss,(v,x))
    assert torch.allclose(dv,torch.full_like(v,2/192))
    assert dx[0].tolist()==pytest.approx([-18.]*10) and dx[1].tolist()==pytest.approx([24.]*10)
    # Input virtual value affects edge1 forward but has no edge1 state-input gradient.
    z,h,raw=d.step.calls[1]
    assert z.shape==(1,3,192) and torch.equal(z[:,-1],v.detach())
    assert not z.requires_grad


def test_serial_sync_full_chain_gradient():
    t=task();t['root_context']['latent_history']=np.ones((3,192)).tolist();d=dynamics(t)
    x=torch.zeros((2,10),requires_grad=True);terminal,path=d.serial(x,m.Meter(lambda e:None))
    assert terminal[0,0].item()==4. and len(path)==2
    dx=torch.autograd.grad(terminal.sum(),x)[0]
    assert dx[0].tolist()==pytest.approx([6*192.]*10)
    assert dx[1].tolist()==pytest.approx([3*192.]*10)


def test_original_history_and_lifted_between_actions_preserved(small):
    t,r,events,meter,d=small
    requests=[e for e in events if e['kind']=='transition_attempt']
    assert requests[0]['latent_history'][0]==t['root_context']['latent_history']
    assert requests[0]['action_history'][0]==t['root_context']['action_history']
    virtual=next(e for e in events if e['kind']=='initialized')['virtual'][0]
    assert requests[1]['latent_history'][0]==t['root_context']['latent_history'][1:]+[virtual]
    assert requests[1]['action_history'][0][0]==t['root_context']['action_history'][1]
    assert requests[1]['action_history'][0][1]==m.core.normalized_tokens(t,np.zeros((5,2)))[0].astype(float).tolist()


def test_complete_schedule_counts_state_noise_and_joint_reset(small):
    t,r,events,meter,d=small
    assert meter.counts()=={k:dict(attempted=v,completed=v) for k,v in m.budget(SMALL).items()}
    assert len(d.step.calls)==14 and r['sync_after_iterations']==[2,4]
    assert [e['after_iteration'] for e in events if e['kind']=='joint_optimizer_reset']==[2,4]
    for start,reset in zip([e for e in events if e['kind']=='sync_started'],[e for e in events if e['kind']=='joint_optimizer_reset']):
        assert start['virtual_unchanged']==reset['virtual_unchanged']
    init=next(e for e in events if e['kind']=='initialized')
    assert np.count_nonzero(init['logits'])==0
    assert np.array_equal(init['virtual'],(np.asarray(t['root_context']['latent_history'][-1],np.float32)[None]+np.asarray(t['goal'],np.float32)[None])/2)
    assert all(np.asarray(e['noise']).any() for e in events if e['kind']=='joint_completed')
    m.choice(t,r,0,SMALL);m.journal_check(t,r,events,SMALL)


def test_exact_normalizer_raw_tanh_and_projection(small):
    t,r,events,_,_=small
    assert np.array_equal(r['commands'],np.tanh(np.asarray(r['logits'],np.float32)).reshape(10,2))
    for e in events:
        if e['kind'] in ('joint_completed','sync_completed'):
            assert (np.linalg.norm(e['logits'],axis=1)<=t['packet_logit_cap']).all()
    x=torch.full((2,10),100.,requires_grad=True);before=m.projected(x,t['packet_logit_cap'])
    assert before[0][0]==100. and (np.linalg.norm(m.host(x),axis=1)<=t['packet_logit_cap']).all()


def test_exact_numpy_tanh_and_single_analytic_chain():
    x=torch.linspace(-1.5,1.5,20).reshape(2,10).requires_grad_();raw=m.exact_raw(x)
    assert np.array_equal(raw.detach().numpy(),np.tanh(x.detach().numpy()))
    gradient=torch.autograd.grad(raw.sum(),x)[0]
    assert torch.equal(gradient,1-raw.detach().square())


def test_local_noise_seed_reproducible_without_global_rng_change(small):
    t,r,events,_,_=small;state=torch.random.get_rng_state().clone();other=[]
    again=m.optimize(t,0,dynamics(t),m.Meter(other.append,cfg=SMALL),cfg=SMALL,analytic_fixture=True)
    assert torch.equal(state,torch.random.get_rng_state())
    assert again==r and other==events


def test_production_rejects_cpu_and_fixture_flag_drift():
    t=task()
    with pytest.raises(ValueError,match='CPU neural'):m.Dynamics(t,torch.device('cpu'),Algebra(t))
    with pytest.raises(ValueError,match='fixture boundary'):m.optimize(t,0,dynamics(t),m.Meter(lambda e:None),cfg=SMALL)


@pytest.mark.parametrize('kw',[dict(steps=0),dict(steps=True),dict(sync_interval=0),dict(sync_steps=-1),dict(lr_s=0),dict(lr_a=float('nan')),dict(state_noise=-1)])
def test_bad_config(kw):
    with pytest.raises(ValueError):m.Config(**kw)


@pytest.mark.parametrize('slot',[True,-1,2])
def test_no_extra_root(slot):
    with pytest.raises(ValueError):m.seed(slot)


@pytest.mark.parametrize('bad',['commands','logits','task','history','terminal','schedule','resources','config'])
def test_result_tamper_rejected(small,bad):
    t,r,_,_,_=small;r=deepcopy(r)
    if bad=='commands':r['commands'][0][0]+=.1
    elif bad=='logits':r['logits'][0][0]=100.
    elif bad=='task':r['task_id']='0'*64
    elif bad=='history':r['verification'][1]['action_history'][0][0]+=.1
    elif bad=='terminal':r['terminal_cost']+=.1
    elif bad=='schedule':r['sync_after_iterations']=[]
    elif bad=='resources':r['resources']['backward']['completed']-=1
    else:r['config']['lr_s']=.2
    with pytest.raises(ValueError):m.choice(t,r,0,SMALL)


@pytest.mark.parametrize('bad',['attempt','history','freeze','missing_packet','missing_reset','swap_sync'])
def test_saved_protocol_tamper_rejected(small,bad):
    t,r,events,_,_=small;events=deepcopy(events)
    if bad=='attempt':next(e for e in events if e['kind']=='operation_completed')['event']['index']=100
    elif bad=='history':next(e for e in events if e['kind']=='transition_completed')['context']['action_history'][0][0]+=.1
    elif bad=='freeze':
        i=next(i for i,e in enumerate(events) if e['kind']=='actions_frozen_before_verification');events.insert(0,events.pop(i))
    elif bad=='missing_packet':events.pop(next(i for i,e in enumerate(events) if e['kind']=='transition_completed'))
    elif bad=='missing_reset':events.pop(next(i for i,e in enumerate(events) if e['kind']=='joint_optimizer_reset'))
    else:next(e for e in events if e['kind']=='transition_attempt')['evaluation_id']='sync_2_0'
    with pytest.raises(ValueError):m.journal_check(t,r,events,SMALL)


def test_failure_keeps_attempt_before_call_and_no_retry():
    t=task();events=[];meter=m.Meter(events.append,cfg=SMALL)
    class Broken(Algebra):
        def __call__(self,z,h,raw,meter):
            meter.begin('action_encoder');raise RuntimeError('analytic injected module failure')
    with pytest.raises(RuntimeError,match='injected'):m.optimize(t,0,dynamics(t,Broken(t)),meter,cfg=SMALL,analytic_fixture=True)
    assert any(e['kind']=='transition_attempt' for e in events) and not any(e['kind']=='transition_completed' for e in events)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.begin('predict')


def test_no_66_forward_match_or_infinite_retry():
    meter=m.Meter(lambda e:None,cfg=SMALL);calls=[]
    for _ in range(m.budget(SMALL)['backward']):meter.call('backward',lambda:calls.append(1))
    with pytest.raises(ValueError):meter.call('backward',lambda:calls.append(1))
    assert len(calls)==6
    for key in ('jacobian','svd','vjp_directions'):
        with pytest.raises(ValueError):m.Meter(lambda e:None).begin(key)


def test_only_approved_exports_no_material_reader(monkeypatch):
    def blocked(*a,**kw):raise AssertionError('no sourcepack/outcome material read')
    monkeypatch.setattr(m.gn,'material',blocked)
    for path,h in zip(m.EXPORTS,m.EXPORT_SHAS,strict=True):
        t=m.base.load_export(path,h)['gn_task']
        assert t['projection']==np.eye(192).tolist() and t['horizon']==2


def test_RAM_gzip_transport_and_full_saved_result(small):
    t,r,events,_,_=small;store=RAMStore()
    for e in events:store.event(e)
    store.put('result.json.gz',r);store.close()
    plain,eof=m.base.inflate(store.files['events.jsonl.gz'])
    assert eof and [m.base.decode(line) for line in plain.splitlines()]==events
    assert m.base.decode(gzip.decompress(store.files['result.json.gz']))==r
    assert sum(map(len,store.files.values()))<1024**2


def test_default_two_root_producer_to_consume_in_RAM(monkeypatch):
    """Real default optimizer schedule, analytical dynamics; fake CUDA attestation.

    This validates interface wiring and resource arithmetic, NOT a CUDA run.
    """
    store=RAMStore();tasks={};slots=[];rows=[]
    for s in (0,1):
        t=task(s);path=f'algebra_task_{s}';tasks[path]=t
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=path,task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        rows.append(dict(source_slot=s,arm=m.ARM,blind_sha256=str(s+2)*64,task_id=m.core.task_id(t),status='not_started',commands=None,resources={}))
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    monkeypatch.setattr(m.base,'load_export',lambda path,h:dict(gn_task=tasks[path]))
    def runner(t,s,sink,row):
        meter=m.Meter(sink)
        try:return m.optimize(t,s,dynamics(t),meter,analytic_fixture=True)
        finally:row['resources']=meter.counts()
    store.put('binding.json',dict(prepared='ram_prepare',prepare_sha256='a'*64))
    m.plan_all(p,store,runner,rows);store.close()
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    report=dict(schema=m.NAME+'.report',status='grasp_model_plans_sealed_not_admitted',commands_released=True,
        errors=[],first_error=None,source_verified=True,model_unchanged=True,P1_admission=False,
        model_hash_before=m.enc.MODEL_TENSOR_SHA,model_hash_after=m.enc.MODEL_TENSOR_SHA,
        prepared='ram_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,
        resources={k:dict(attempted=v,completed=v) for k,v in m.contract()['total_budget'].items()},
        artifact_sha256={n:digest(v) for n,v in store.files.items()})
    store.put('report.json',report,final=True);expected=digest(store.files['report.json'])
    prefix=Path('/RAM_grasp_analytic');actual_sha=m.sha;actual_read=m.read;actual_iter=Path.iterdir;actual_bytes=Path.read_bytes
    monkeypatch.setattr(m,'verify',lambda path,h:p)
    monkeypatch.setattr(m,'sha',lambda path:digest(store.files[Path(path).name]) if Path(path).parent==prefix else actual_sha(path))
    def ramread(path):
        path=Path(path)
        if path.parent!=prefix:return actual_read(path)
        raw=store.files[path.name];return m.base.decode(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'read',ramread)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(prefix/n for n in store.files) if path==prefix else actual_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==prefix else actual_bytes(path))
    consumed=m.consume(prefix,expected)
    assert len(consumed['rows'])==2 and consumed['P1_admission'] is False
    assert [r['source_slot'] for r in consumed['rows']]==[0,1]
    assert all(r['arm']==m.ARM and len(r['commands'])==10 for r in consumed['rows'])
    print('Default analytical two-root RAM full-artifact bytes:',sum(map(len,store.files.values())))
    store.files[rows[0]['plan_file']]+=b' '
    with pytest.raises(ValueError,match='artifact SHA'):m.consume(prefix,expected)
