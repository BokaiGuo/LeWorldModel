"""Only NumPy recurrence/RAM journal mocks; no neural, physics, tmp or outcomes."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_gradient_mpc_cuda_20261027_v1 as m

F=np.sin(np.arange(1920).reshape(192,10)/37)/20
G=np.cos(np.arange(1920).reshape(192,10)/41)/25


def task(seed):
    rng=np.random.default_rng(seed)
    t=dict(environment='swm/PushT-v1',task_contract_sha256='a'*64,
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=rng.normal(0,.2,(3,192)).astype(np.float32).astype(float).tolist(),action_history=np.zeros((2,10)).tolist()),
        root_raw_action_history=np.zeros((10,2)).tolist(),goal=rng.normal(0,.2,192).astype(np.float32).astype(float).tolist(),
        projection=np.eye(192).tolist(),model_sha256='b'*64,model_config_sha256='c'*64,metric_source_sha256='d'*64,
        goal_source_sha256='e'*64,representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=[-.0087,.0068],action_std=[.2019,.2002],packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only')
    t['root_context']['action_history']=m.core.normalized_tokens(t,np.zeros((10,2))).astype(float).tolist()
    m.core.validate_task(t);return t


def path(t,raw):
    previous=deepcopy(t['root_context']);out=[]
    for k in (0,1):
        token=m.core.normalized_tokens(t,raw[5*k:5*k+5])[0].astype(float)
        z=(.6*np.asarray(previous['latent_history'][-1])+F@token+.1*G@np.asarray(previous['action_history'][-1])).astype(np.float32)
        c=dict(previous,latent_history=deepcopy(previous['latent_history'][1:])+[z.astype(float).tolist()],
               action_history=[deepcopy(previous['action_history'][-1]),token.tolist()])
        out.append(c);previous=c
    return out


class Backend:
    def __init__(self):self.active=False;self.calls=[];self.points=[]
    @contextmanager
    def guard(self,meter):
        assert not self.active
        self.active=True
        try:yield
        finally:self.active=False
    def forwards(self,t,raw,meter,kind,mode=None):
        contexts=path(t,raw)
        for k,c in enumerate(contexts):
            for r in ('action_encoder','predict'):
                token=meter.begin(r);meter.end(token)
            if kind=='native_packet':meter.emit(kind,packet=k,mode=mode,context=c)
            else:meter.emit(kind,packet=k,context=c)
        return contexts
    def rollout(self,t,raw,meter,mode):
        assert self.active and raw.shape==(10,2) and raw.dtype==np.float32
        self.calls.append((raw.copy(),mode))
        return self.forwards(t,raw,meter,'native_packet',mode)
    def gradient(self,t,incumbent,meter):
        assert self.active
        self.points.append(deepcopy(incumbent['logits']))
        contexts=self.forwards(t,np.asarray(incumbent['commands'],np.float32),meter,'gradient_forward_packet')
        assert contexts==incumbent['contexts']
        z=np.asarray(contexts[-1]['latent_history'][-1]);residual=z-np.asarray(t['goal'])
        before=dict(task_id=m.core.task_id(t),at_logits=deepcopy(incumbent['logits']),contexts=contexts,at_terminal=z.tolist(),
            native_float32_loss=float(np.mean(residual.astype(np.float32)**2)),saved_float64_cost=m.core.model_cost(t,z))
        meter.emit('gradient_forward_complete',forward=before)
        def backward():
            e=dict(index=len(self.points)-1,resource='grad',evaluation_id=meter.group,completed=False,
                   scalar_outputs=1,input_shape=[10,2],retain_graph=False,create_graph=False)
            meter.emit('AD_API_attempt',entry=e)
            std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
            j=np.column_stack(((.6*F+.1*G)/std,F/std))
            raw=(2*j.T@residual/192).reshape(10,2)
            meter.emit('AD_API_completed',entry=dict(e,completed=True));return raw
        raw=meter.call('backward',backward);chain=1-np.asarray(incumbent['commands'])**2
        d=dict(**before,raw_control_gradient=raw.tolist(),objective_gradient=(raw*chain).reshape(2,10).tolist(),
               tanh_chain=chain.tolist(),scalar_reverse_sweeps=1)
        meter.emit('gradient_complete',gradient=d);return d


class RAMStore(m.base.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw


@pytest.fixture(scope='module')
def fullmock():
    runs=[];store=RAMStore();wrapped=[]
    for s in (0,1):
        t=task(20261027+s);events=[]
        def sink(e,s=s,t=t):
            events.append(deepcopy(e));w=dict(source_slot=s,arm=m.ARM,task_id=m.core.task_id(t),event=deepcopy(e))
            wrapped.append(w);store.event(w)
        meter=m.Meter(sink);backend=Backend()
        with backend.guard(meter):result=m.optimize(t,backend,meter)
        store.put(f'root_{s}_{m.ARM}.result.json.gz',result)
        runs.append((t,result,events,meter,backend))
    store.close();return runs,store,wrapped


def test_presealed_config_and_heterogeneous_budget():
    c=m.contract();b=m.budget()
    assert m.UPDATES==32 and m.STEP_SIZE==1.0 and not np.asarray(c['initial_logits']).any()
    assert b==dict(predict=136,action_encoder=136,backward=32,proposal=32,selection=1)
    assert c['total_budget']==dict(predict=272,action_encoder=272,backward=64,proposal=64,selection=2)
    assert c['gradient_forward_predict_per_root']==64 and c['non_AD_predict_per_root']==72
    assert c['AD_API_per_root']==dict(grad=32) and c['scalar_reverse_sweeps_per_root']==32
    assert not c['equal_total_resource_claim'] and not c['closed_loop'] and not c['P1_admission']


def test_two_complete_fixed_runs_real_H2_no_J(fullmock):
    for t,r,events,meter,b in fullmock[0]:
        assert len(b.calls)==36 and len(b.points)==32 and not b.active
        assert [p[1] for p in b.calls[-3:]]==['nominal','whole_replay','segmented_replay']
        assert r['resources']=={k:dict(attempted=v,completed=v) for k,v in m.budget().items()}
        assert len(r['candidates'])==33 and len(r['gradients'])==32
        assert m.choice(t,r)['commands']==r['final_prediction']['commands']
        m.journal_check(t,r,events)
        assert sum(e['kind']=='gradient_forward_packet' for e in events)==64
        assert sum(e['kind']=='native_packet' for e in events)==72
        assert not any('jacobian' in e['kind'] or 'svd' in e['kind'] for e in events)


def test_fullsize_RAM_capacity_syncflush_readback(fullmock):
    runs,store,events=fullmock
    raw,eof=m.base.inflate(store.files['events.jsonl.gz']);assert eof
    assert [m.base.decode(x) for x in raw.splitlines()]==events
    for s,(_,r,_,_,_) in enumerate(runs):
        assert m.base.decode(gzip.decompress(store.files[f'root_{s}_{m.ARM}.result.json.gz']))==r
    assert sum(map(len,store.files.values()))+m.FINAL<m.CAP
    print('full two-root RAM journal+result bytes',sum(map(len,store.files.values())))


def test_projection_is_per_packet_and_gradient_sign():
    x=np.zeros((2,10));g=np.vstack((np.ones(10)*10,np.zeros(10)))
    y=m.proposal(x,g,float(.75*np.sqrt(5)))
    assert (y[0]<0).all() and not y[1].any()
    assert np.linalg.norm(y[0])<=float(.75*np.sqrt(5)) and np.array_equal(y,y.astype(np.float32))


def test_whole_gradient_includes_prefix_state_and_action_history(fullmock):
    t,r,_,_,_=fullmock[0][0];d=r['gradients'][0]
    residual=np.asarray(d['at_terminal'])-t['goal'];std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
    expected=np.column_stack(((.6*F+.1*G)/std,F/std)).T@residual*2/192
    assert np.array_equal(d['objective_gradient'],expected.reshape(2,10))
    wrong=np.column_stack((np.zeros((192,10)),F/std)).T@residual*2/192
    assert np.linalg.norm(expected[:10]-wrong[:10])>0
    assert np.linalg.norm(expected[:10]-(.6*F/std).T@residual*2/192)>0


def test_nonzero_logit_gradient_matches_array_central_FD(fullmock):
    t,r,_,_,_=fullmock[0][0];d=r['gradients'][8];x=np.asarray(d['at_logits']);measured=[]
    for i in range(20):
        v=np.zeros(20);v[i]=1;v=v.reshape(2,10)
        costs=[]
        for sign in (-1,1):
            raw=np.tanh((x+sign*.001*v).astype(np.float32)).reshape(10,2)
            costs.append(m.core.model_cost(t,path(t,raw)[-1]['latent_history'][-1]))
        measured.append((costs[1]-costs[0])/.002)
    assert np.allclose(measured,np.asarray(d['objective_gradient']).reshape(20),atol=2e-5,rtol=.02) # array mock only


@pytest.mark.parametrize('damage',('gradient','chain','factor192','point','cost','first','iteration','winner','truncate'))
def test_choice_rejects_changed_recurrence_or_ownplan(fullmock,damage):
    t,r,_,_,_=fullmock[0][0];r=deepcopy(r)
    if damage=='gradient':r['gradients'][0]['objective_gradient'][0][0]+=.01
    elif damage=='chain':r['gradients'][0]['tanh_chain'][0][0]=.5
    elif damage=='factor192':r['gradients'][0]['raw_control_gradient']=(np.asarray(r['gradients'][0]['raw_control_gradient'])*192).tolist()
    elif damage=='point':r['gradients'][0]['at_logits'][0][0]+=.1
    elif damage=='cost':r['candidates'][0]['cost']+=.1
    elif damage=='first':r['candidates'][0]['logits'][0][0]=.1
    elif damage=='iteration':r['updates']=31
    elif damage=='winner':r['winner_iteration']=(r['winner_iteration']+1)%33
    else:r['gradients'].pop()
    with pytest.raises(ValueError):m.choice(t,r)


@pytest.mark.parametrize('damage',('history','ADhistory','ADterminal','earlywinner','proposal','dispatch','attempt','api','api_shape','workscope','omit','duplicate'))
def test_release_journal_rejects_causal_and_budget_tamper(fullmock,damage):
    t,r,events,_,_=fullmock[0][0];events=deepcopy(events)
    if damage=='history':next(e for e in events if e['kind']=='native_packet')['context']['action_history'][0][0]+=.1
    elif damage=='ADhistory':next(e for e in events if e['kind']=='gradient_forward_packet')['context']['action_history'][0][0]+=.1
    elif damage=='ADterminal':next(e for e in events if e['kind']=='gradient_forward_complete')['forward']['at_terminal'][0]+=.1
    elif damage=='earlywinner':
        k=next(k for k,e in enumerate(events) if e['kind']=='winner_frozen_before_replay');events.insert(0,events.pop(k))
    elif damage=='proposal':next(e for e in events if e['kind']=='proposal_frozen')['logits'][0][0]+=.1
    elif damage=='dispatch':next(e for e in events if e['kind']=='evaluation_request' and e['evaluation_id']=='iterate_1')['commands'][0][0]+=.1
    elif damage=='attempt':next(e for e in events if e['kind']=='operation_completed')['event']['index']=99
    elif damage=='api':next(e for e in events if e['kind']=='AD_API_completed')['entry']['evaluation_id']='wrong'
    elif damage=='api_shape':next(e for e in events if e['kind']=='AD_API_attempt')['entry']['input_shape']=[20]
    elif damage=='workscope':
        for e in events:
            if e['kind'] in ('operation_attempted','operation_completed') and e['event']['resource']=='predict' and e['event']['evaluation_id']=='gradient_1':
                e['event']['evaluation_id']='gradient_2';e['evaluation_id']='gradient_2'
    elif damage=='omit':events.pop(next(k for k,e in enumerate(events) if e['kind']=='AD_API_attempt'))
    else:events.append(deepcopy(next(e for e in events if e['kind']=='operation_attempted')))
    with pytest.raises(ValueError):m.journal_check(t,r,events)


def test_fixed_descent_not_silently_early_stopped_on_zero_gradient():
    class Zero(Backend):
        def gradient(self,t,inc,meter):
            d=super().gradient(t,inc,meter)
            d['raw_control_gradient']=np.zeros((10,2)).tolist();d['objective_gradient']=np.zeros((2,10)).tolist()
            return d
    t=task(1);meter=m.Meter(lambda e:None);b=Zero()
    with b.guard(meter):r=m.optimize(t,b,meter)
    assert len(r['gradients'])==32 and r['winner_iteration']==0
    assert len(b.calls)==36 and all(c['logits']==np.zeros((2,10)).tolist() for c in r['candidates'])


def test_nonrefundable_sync_or_journal_failure():
    def sync():raise RuntimeError('async CUDA error mock')
    meter=m.Meter(lambda e:None,sync);e=meter.begin('backward')
    with pytest.raises(RuntimeError):meter.end(e)
    assert meter.counts()['backward']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.begin('backward')
    def sink(e):raise OSError('fsync mock')
    meter=m.Meter(sink)
    with pytest.raises(OSError):meter.begin('predict')
    assert meter.counts()['predict']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.begin('predict')


def fake_torch():
    def unused(*a,**kw):return ('gradient',)
    f=SimpleNamespace(**{k:unused for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')})
    func=SimpleNamespace(**{k:unused for k in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value')})
    return SimpleNamespace(autograd=SimpleNamespace(grad=unused,backward=unused,functional=f),
                           Tensor=SimpleNamespace(backward=unused),func=func)


def test_grad_API_gate_only_inside_scalar_attempt_and_restore():
    torch=fake_torch();original=torch.autograd.grad;events=[];meter=m.Meter(events.append);gate=m.GradGate(torch,meter)
    loss=SimpleNamespace(ndim=0);raw=SimpleNamespace(shape=(10,2),is_leaf=True,requires_grad=True,grad=None)
    with gate:
        e=meter.begin('backward')
        assert torch.autograd.grad(loss,raw,retain_graph=False,create_graph=False)==('gradient',)
        meter.end(e)
    assert torch.autograd.grad is original and gate.counts()==dict(grad=dict(attempted=1,completed=1))
    with m.GradGate(torch,m.Meter(lambda e:None)):
        with pytest.raises(ValueError,match='declared scalar'):torch.autograd.grad(loss,raw,retain_graph=False,create_graph=False)


@pytest.mark.parametrize('which',('backward','jacobian','func'))
def test_unbudgeted_AD_is_rejected(which):
    torch=fake_torch();gate=m.GradGate(torch,m.Meter(lambda e:None))
    with gate:
        fn=torch.autograd.backward if which=='backward' else torch.autograd.functional.jacobian if which=='jacobian' else torch.func.grad
        with pytest.raises(ValueError,match='unbudgeted'):fn()
    assert gate.failed


def test_first_execution_error_survives_cleanup_RAM(monkeypatch):
    mem={}
    class RAMPath:
        def iterdir(self):return []
    class BrokenStore:
        def __init__(self,output):self.output=RAMPath()
        def put(self,name,v,final=False):mem[name]=deepcopy(v)
        def event(self,e):mem.setdefault('events',[]).append(e)
        def close(self):raise OSError('journal finalization mock')
    slots=[dict(source_slot=s,blind_sha256=str(s)*64,task_id=str(s+2)*64) for s in (0,1)]
    p=dict(slots=slots,contract=m.contract(),global_freshness='mock')
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m.base,'Store',BrokenStore)
    monkeypatch.setitem(sys.modules,'torch',None)
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    with pytest.raises(ValueError,match='partial gradient MPC saved'):m.run('unused','unused','unused')
    r=mem['report.json'];assert r['first_error']['stage']=='execution' and r['errors'][-1]['stage']=='journal_finalize'
    assert r['status']=='partial_or_failed' and not r['commands_released'] and r['model_unchanged'] is None


def test_own_plans_and_complete_consume_in_RAM(fullmock,monkeypatch):
    store=RAMStore();slots=[];rows=[];tasks={}
    for s,(t,_,_,_,_) in enumerate(fullmock[0]):
        path=f'task_{s}';tasks[path]=t
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=path,task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        rows.append(dict(source_slot=s,arm=m.ARM,task_id=m.core.task_id(t),blind_sha256=str(s+2)*64,status='not_started',resources={},AD_API_resources={}))
    monkeypatch.setattr(m.base,'load_export',lambda path,h:dict(gn_task=tasks[path]))
    def runner(t,sink,row):
        k=next(k for k,x in enumerate(fullmock[0]) if x[0] is t)
        for e in fullmock[0][k][2]:sink(deepcopy(e))
        r=deepcopy(fullmock[0][k][1]);row['resources']=r['resources'];row['AD_API_resources']=dict(grad=dict(attempted=32,completed=32));return r
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    store.put('binding.json',dict(prepared='ram_prepare',prepare_sha256='a'*64))
    m.plan_all(p,store,runner,rows);store.close()
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    report=dict(schema=m.NAME+'.report',status='gradient_mpc_model_plans_sealed_not_admitted',commands_released=True,errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.enc.MODEL_TENSOR_SHA,model_hash_after=m.enc.MODEL_TENSOR_SHA,
        prepared='ram_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,
        resources={k:dict(attempted=2*v,completed=2*v) for k,v in m.budget().items()},AD_API_resources=dict(grad=dict(attempted=64,completed=64)),
        P1_admission=False,new_Q=False,training=False,equal_total_resource_claim=False,new_images=0,new_physics=0,
        artifact_sha256={n:digest(v) for n,v in store.files.items()})
    store.put('report.json',report,final=True);expected=digest(store.files['report.json'])
    prefix=Path('/RAM_gradient_MPC');original_sha=m.sha;original_read=m.read
    original_iter=Path.iterdir;original_bytes=Path.read_bytes;original_stat=Path.stat
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m,'sha',lambda path:digest(store.files[Path(path).name]) if Path(path).parent==prefix else original_sha(path))
    def read(path):
        path=Path(path)
        if path.parent!=prefix:return original_read(path)
        raw=store.files[path.name];return m.base.decode(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([prefix/n for n in store.files]) if path==prefix else original_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==prefix else original_bytes(path))
    monkeypatch.setattr(Path,'stat',lambda path,*a,**kw:SimpleNamespace(st_size=len(store.files[path.name])) if path.parent==prefix else original_stat(path,*a,**kw))
    result=m.consume(prefix,expected)
    assert len(result['rows'])==2 and result['new_physics']==0 and not result['P1_admission']
    assert all(x['arm']==m.ARM and len(x['commands'])==10 and x['with_target'] is False for x in result['rows'])
    print('full two-root RAM journal+results+plans+report bytes',sum(map(len,store.files.values())))
    store.files[rows[0]['plan_file']]+=b' '
    with pytest.raises(ValueError,match='artifact hash'):m.consume(prefix,expected)


def test_no_outcome_or_privileged_actual_inputs_RAM_build():
    if not Path(m.EXPORTS[0]).is_file():pytest.skip('explicit approved exports absent')
    p=m.build()
    assert len(p['slots'])==2 and len(m.canonical(p))<256*1024 and p['contract']==m.contract()
    paths=p['bindings_sha256']
    assert not any(Path(x).name=='sourcepack.json' or 'certificate' in x or Path(x).suffix=='.npz' for x in paths)
    assert all('results' not in Path(x).parts or x in m.EXPORTS for x in paths)
    assert list(s['task_export'] for s in p['slots'])==m.EXPORTS
