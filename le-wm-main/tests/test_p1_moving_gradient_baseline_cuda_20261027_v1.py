"""RAM-only two-root integration, projected math and command-release contracts."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
from pathlib import Path
from types import SimpleNamespace
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_gradient_baseline_cuda_20261027_v1 as m

F=np.sin(np.arange(1920).reshape(192,10)/37)/20
G=np.cos(np.arange(1920).reshape(192,10)/41)/25


def task(seed):
    rng=np.random.default_rng(seed)
    t=dict(environment='swm/PushT-v1',task_contract_sha256='a'*64,
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=rng.normal(0,.2,(3,192)).astype(np.float32).astype(float).tolist(),action_history=[]),
        root_raw_action_history=m.moving.fixed_raw().astype(float).tolist(),goal=rng.normal(0,.2,192).astype(np.float32).astype(float).tolist(),
        projection=np.eye(192).tolist(),model_sha256='b'*64,model_config_sha256='c'*64,metric_source_sha256='d'*64,
        goal_source_sha256='e'*64,representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=m.base.d2p.ACTION_MEAN.astype(float).tolist(),action_std=m.base.d2p.ACTION_STD.astype(float).tolist(),
        packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only')
    t['root_context']['action_history']=m.core.normalized_tokens(t,m.moving.fixed_raw()).astype(float).tolist()
    m.core.validate_task(t);return t


class Backend:
    def __init__(self):self.calls=[];self.points=[];self.index=0;self.inputs=[]
    @contextmanager
    def guard(self,meter):yield
    def forwards(self,t,raw,meter,kind,mode=None):
        previous=deepcopy(t['root_context']);contexts=[]
        for k in range(2):
            packet=raw[5*k:5*k+5];token=m.core.normalized_tokens(t,packet)[0].astype(float)
            meter.call('action_encoder',lambda:None)
            i=self.index;self.index+=1;meter.emit('predict_API_attempt',index=i)
            incoming=m.moving.incoming(t,previous,packet);self.inputs.append(deepcopy(incoming))
            meter.emit('native_context_attempt',index=i,context=incoming)
            z=meter.call('predict',lambda:(.6*np.asarray(previous['latent_history'][-1])+F@token+.1*G@np.asarray(previous['action_history'][-1])).astype(np.float32))
            meter.emit('predict_API_completed',index=i)
            c=dict(previous,latent_history=deepcopy(previous['latent_history'][1:])+[z.astype(float).tolist()],
                action_history=[deepcopy(previous['action_history'][-1]),token.tolist()])
            contexts.append(c);previous=c
            meter.emit(kind,packet=k,context=c,**(dict(mode=mode) if mode is not None else {}))
        return contexts
    def rollout(self,t,raw,meter,mode):
        assert raw.dtype==np.float32 and raw.shape==(10,2)
        self.calls.append((raw.copy(),mode));return self.forwards(t,raw,meter,'native_packet',mode)
    def gradient(self,t,inc,meter):
        self.points.append(deepcopy(inc['logits']))
        contexts=self.forwards(t,np.asarray(inc['commands'],np.float32),meter,'gradient_forward_packet')
        assert contexts==inc['contexts']
        z=np.asarray(contexts[-1]['latent_history'][-1]);res=z-t['goal']
        before=dict(task_id=m.core.task_id(t),at_logits=deepcopy(inc['logits']),contexts=contexts,at_terminal=z.tolist(),
            native_float32_loss=float(np.mean(res.astype(np.float32)**2)),saved_float64_cost=m.core.model_cost(t,z))
        meter.emit('gradient_forward_complete',forward=before)
        def backward():
            e=dict(index=len(self.points)-1,resource='grad',evaluation_id=meter.group,completed=False,
                scalar_outputs=1,input_shape=[10,2],retain_graph=False,create_graph=False)
            meter.emit('AD_API_attempt',entry=e)
            std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
            j=np.column_stack(((.6*F+.1*G)/std,F/std));raw=(2*j.T@res/192).reshape(10,2)
            meter.emit('AD_API_completed',entry=dict(e,completed=True));return raw
        raw=meter.call('backward',backward);chain=1-np.asarray(inc['commands'])**2
        d=dict(**before,raw_control_gradient=raw.tolist(),objective_gradient=(raw*chain).reshape(2,10).tolist(),tanh_chain=chain.tolist(),scalar_reverse_sweeps=1)
        meter.emit('gradient_complete',gradient=d);return d


class RAMStore(m.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False;self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw


@pytest.fixture(scope='module')
def runs():
    values=[]
    for s in (0,1):
        t=task(20261027+s);es=[];meter=m.gradient.Meter(lambda e:es.append(deepcopy(e)));b=Backend()
        r=m.optimize(t,b,meter);values.append((t,r,es,b))
    return values


def test_frozen_math_and_counts():
    assert m.UPDATES==32 and m.STEP_SIZE==1
    assert m.optimize is m.gradient.optimize and m.choice is m.gradient.choice and m.proposal is m.gradient.proposal
    assert m.NativeGradient.gradient is m.gradient.NativeGradient.gradient
    c=m.contract();assert c['total_budget']==dict(predict=272,action_encoder=272,backward=64,proposal=64,selection=2)
    assert c['AD_API_per_root']==dict(grad=32) and c['scalar_reverse_sweeps_per_root']==32
    assert not c['closed_loop'] and not c['feedback_MPC'] and not c['P1_admission'] and not c['official_reproduction']
    assert c['output_cap_bytes']==8*1024**2 and c['reserve_bytes']==200*1024**2


def test_two_root_end_to_end_nonzero_every_forward(runs):
    for t,r,es,b in runs:
        assert len(b.inputs)==136 and len(b.calls)==36 and len(b.points)==32
        assert len(r['candidates'])==33 and len(r['gradients'])==32
        assert [mode for _,mode in b.calls[-3:]]==['nominal','whole_replay','segmented_replay']
        m.journal_check(t,r,es)
        assert m.history_check(t,es)==dict(native_contexts=136,predict_API=136,all_nonzero_history_paths_verified=True)
        for i in range(0,136,2):
            assert b.inputs[i]['latent_history']==t['root_context']['latent_history']
            assert b.inputs[i]['action_history'][:2]==t['root_context']['action_history']
            assert b.inputs[i+1]['action_history'][0]==t['root_context']['action_history'][-1]
        assert m.choice(t,r)['commands']==r['final_prediction']['commands']


def test_native_start_exact_moving_no_zero_substitution():
    t=task(3);b=object.__new__(m.NativeGradient);b.bound_id=m.core.task_id(t);b._tensor=lambda a:np.asarray(a,np.float32)
    z,h=b._start(t)
    np.testing.assert_array_equal(z[0],np.asarray(t['root_context']['latent_history'],np.float32))
    np.testing.assert_array_equal(h[0],m.core.normalized_tokens(t,m.moving.fixed_raw()))
    for key in ('root_raw_action_history','tokens'):
        bad=deepcopy(t)
        if key=='tokens':bad['root_context']['action_history']=np.zeros((2,10)).tolist()
        else:bad[key]=np.zeros((10,2)).tolist()
        with pytest.raises(ValueError):b._start(bad)


@pytest.mark.parametrize('damage',['latent','token','zero','API','order','missing','AD','replay'])
def test_native_input_release_rejects_tamper(runs,damage):
    t,r,es,_=runs[0];es=deepcopy(es)
    traces=[e for e in es if e['kind']=='native_context_attempt']
    if damage=='latent':traces[0]['context']['latent_history'][0][0]+=.1
    elif damage=='token':traces[0]['context']['action_history'][0][0]+=.1
    elif damage=='zero':traces[0]['context']['action_history'][:2]=np.zeros((2,10)).tolist()
    elif damage=='API':next(e for e in es if e['kind']=='predict_API_completed')['index']=1
    elif damage=='order':
        i=next(i for i,e in enumerate(es) if e['kind']=='predict_API_attempt');es.insert(i,es.pop(i+1))
    elif damage=='missing':es.remove(traces[10])
    elif damage=='AD':next(e for e in es if e['kind']=='gradient_forward_packet')['context']['action_history'][0][0]+=.1
    else:traces[-1]['context']['latent_history'][0][0]+=.1
    with pytest.raises(ValueError):m.history_check(t,es)


def test_joint_gradient_mock_includes_both_paths(runs):
    t,r,_,_=runs[0];d=r['gradients'][0];res=np.asarray(d['at_terminal'])-t['goal'];std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
    want=(2*np.column_stack(((.6*F+.1*G)/std,F/std)).T@res/192).reshape(2,10)
    np.testing.assert_array_equal(d['objective_gradient'],want)
    assert np.linalg.norm(want[0])>0
    assert (m.proposal(np.zeros((2,10)),np.ones((2,10))*10,t['packet_logit_cap'])<0).all()


@pytest.mark.parametrize('damage',['winner','gradient','chain','truncate'])
def test_inherited_selection_rejects_modified_results(runs,damage):
    t,r,_,_=runs[0];r=deepcopy(r)
    if damage=='winner':r['winner_iteration']=(r['winner_iteration']+1)%33
    elif damage=='gradient':r['gradients'][0]['objective_gradient'][0][0]+=.1
    elif damage=='chain':r['gradients'][0]['tanh_chain'][0][0]=.5
    else:r['candidates'].pop()
    with pytest.raises(ValueError):m.choice(t,r)


def test_api_trace_attempt_on_failure_and_restoration():
    es=[]
    def bad(*a):raise RuntimeError('native')
    model=SimpleNamespace(predict=bad)
    with pytest.raises(RuntimeError),m.api_trace(model,es.append):model.predict()
    assert es==[dict(kind='predict_API_attempt',index=0)] and model.predict is bad


def test_api_exact_cap_and_partial_counters():
    model=SimpleNamespace(predict=lambda:123);events=[]
    with m.api_trace(model,events.append) as entries:
        for _ in range(136):assert model.predict()==123
        with pytest.raises(ValueError,match='cap'):model.predict()
    assert len(entries)==136 and all(e['completed'] for e in entries) and len(events)==272


def test_native_tensor_backend_rejects_CPU_before_import():
    with pytest.raises(ValueError,match='CUDA only'):m.NativeGradient(None,SimpleNamespace(type='cpu'),task(1))


def test_GradGate_single_scalar_sweep_not_192_VJP():
    meter=m.gradient.Meter(lambda e:None)
    torch=SimpleNamespace()
    gate=m.gradient.GradGate(torch,meter)
    scalar=SimpleNamespace(ndim=0);leaf=SimpleNamespace(shape=(10,2),is_leaf=True,requires_grad=True,grad=None)
    with pytest.raises(ValueError,match='declared scalar'):gate.dispatch(lambda *a,**kw:None,(scalar,leaf),dict(retain_graph=False,create_graph=False))
    for _ in range(32):
        token=meter.begin('backward')
        assert gate.dispatch(lambda *a,**kw:123,(scalar,leaf),dict(retain_graph=False,create_graph=False))==123
        meter.end(token)
    assert gate.counts()==dict(grad=dict(attempted=32,completed=32))
    assert meter.counts()['backward']==dict(attempted=32,completed=32)
    with pytest.raises(ValueError):gate.dispatch(lambda:None,(scalar,leaf),dict(retain_graph=False,create_graph=False))


@pytest.mark.parametrize('which',['encoder','source'])
def test_wrong_external_source_SHA_fails_before_task_loading(which,monkeypatch):
    monkeypatch.setattr(m.moving,'build',lambda *a:pytest.fail('must reject external SHA before task loading'))
    with pytest.raises(ValueError,match='SHA'):
        m.build(m.moving.EXPORTS,m.moving.EXPORT_SHAS,m.moving.LOADER_SHA,str(m.ENCODER),
            '0'*64 if which=='encoder' else m.ENCODER_SHA,str(m.SOURCE),'0'*64 if which=='source' else m.SOURCE_SHA)


@pytest.mark.parametrize('damage',['wall','memory','disk'])
def test_real_resource_caps(damage):
    tick=[0];l=m.Limits(free=lambda:m.FLOOR-1 if damage=='disk' else m.FLOOR+12345,
        clock=lambda:tick[0],memory=lambda:(3221225473,0) if damage=='memory' else (0,0))
    if damage=='wall':tick[0]=600
    with pytest.raises(ValueError):l.check()


def test_real_run_initial_disk_no_fixture_constant(monkeypatch):
    paths=[];monkeypatch.setattr(m,'verify',lambda *a:{})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:paths.append(p) or SimpleNamespace(free=m.FLOOR-1))
    monkeypatch.setattr(m,'Store',lambda p:pytest.fail('no output below floor'))
    with pytest.raises(ValueError,match='floor'):m.run('unused','unused','/tmp/not_written')
    assert paths==[Path('/tmp')]


def test_first_error_and_partial_fixed_rows_no_neural(monkeypatch):
    class Broken:
        output=SimpleNamespace(iterdir=lambda:[])
        def put(self,name,value,**kw):
            if name=='binding.json':raise OSError('first write failure')
            self.report=value
        def event(self,e):return None
        def close(self):raise OSError('close failure')
    store=Broken();slots=[dict(source_slot=s,blind_sha256=str(s)*64,task_id=str(s+2)*64) for s in (0,1)]
    monkeypatch.setattr(m,'verify',lambda *a:dict(slots=slots,contract=m.contract()))
    monkeypatch.setattr(m,'Store',lambda p:store)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='partial moving gradient saved'):m.run('unused','unused','/tmp/not_written')
    r=store.report;assert r['status']=='partial_or_failed' and not r['commands_released']
    assert r['first_error']['message']=='first write failure' and r['errors'][1]['message']=='close failure'
    assert len(r['rows'])==2 and all(row['status']=='not_started' for row in r['rows'])


def test_own_consumer_roundtrip_in_RAM(runs,monkeypatch):
    store=RAMStore();slots=[];rows=[];tasks={}
    for s,(t,_,_,_) in enumerate(runs):
        path=f'moving_task_{s}';tasks[path]=t
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=path,task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        rows.append(dict(source_slot=s,arm=m.ARM,task_id=m.core.task_id(t),blind_sha256=str(s+2)*64,status='not_started',resources={},AD_API_resources={}))
    monkeypatch.setattr(m,'load_task',lambda path,h:tasks[path])
    def runner(t,sink,row):
        k=next(k for k,v in enumerate(runs) if v[0] is t)
        for e in runs[k][2]:sink(deepcopy(e))
        r=deepcopy(runs[k][1]);row['resources']=r['resources'];row['AD_API_resources']=dict(grad=dict(attempted=32,completed=32))
        row['predict_API_resources']=dict(predict=dict(attempted=136,completed=136));return r
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    store.put('binding.json',dict(prepared='ram_prepare',prepare_sha256='a'*64));m.plan_all(p,store,runner,rows);store.close()
    r=dict(schema=m.NAME+'.report',status='moving_gradient_model_plans_sealed_not_admitted',commands_released=True,errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.gradient.enc.MODEL_TENSOR_SHA,model_hash_after=m.gradient.enc.MODEL_TENSOR_SHA,
        prepared='ram_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,
        resources={k:dict(attempted=2*v,completed=2*v) for k,v in m.budget().items()},AD_API_resources=dict(grad=dict(attempted=64,completed=64)),
        predict_API_resources=dict(predict=dict(attempted=272,completed=272)),
        limits=dict(checks=123,minimum_observed_free_bytes=m.FLOOR,peak_allocated_bytes=100000,peak_reserved_bytes=100000,elapsed_seconds=123),
        P1_admission=False,new_Q=False,training=False,equal_total_resource_claim=False,new_images=0,new_physics=0,
        artifact_sha256={n:m.gradient.enc.digest(b) for n,b in store.files.items()})
    store.put('report.json',r,final=True);expected=m.gradient.enc.digest(store.files['report.json']);prefix=Path('/RAM_moving_gradient')
    original_sha,original_read=m.sha,m.read;orig_iter,orig_bytes,orig_stat=Path.iterdir,Path.read_bytes,Path.stat
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m,'sha',lambda path:m.gradient.enc.digest(store.files[Path(path).name]) if Path(path).parent==prefix else original_sha(path))
    def read(path):
        path=Path(path)
        if path.parent!=prefix:return original_read(path)
        raw=store.files[path.name];return m.base.decode(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(prefix/n for n in store.files) if path==prefix else orig_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==prefix else orig_bytes(path))
    monkeypatch.setattr(Path,'stat',lambda path,*a,**kw:SimpleNamespace(st_size=len(store.files[path.name])) if path.parent==prefix else orig_stat(path,*a,**kw))
    result=m.consume(prefix,expected)
    assert len(result['rows'])==2 and result['P1_admission'] is False
    assert all(p['schema']==m.NAME+'.model_plan' and p['arm']==m.ARM for p in result['rows'])
    print('RAM complete two-root bytes',sum(map(len,store.files.values())))
    store.files[rows[0]['plan_file']]+=b' '
    with pytest.raises(ValueError,match='artifact SHA'):m.consume(prefix,expected)


def test_loader_not_legacy_zero(monkeypatch):
    monkeypatch.setattr(m.base,'load_export',lambda *a:pytest.fail('old zero loader forbidden'))
    monkeypatch.setattr(m.moving.moving.low,'load_task',lambda *a:pytest.fail('old zero task loader forbidden'))
    for path,h in zip(m.moving.EXPORTS,m.moving.EXPORT_SHAS,strict=True):
        t=m.load_task(path,h)
        np.testing.assert_array_equal(t['root_raw_action_history'],m.moving.fixed_raw())


def test_saved_RAM_build_no_outcome_or_witness():
    p=m.build(m.moving.EXPORTS,m.moving.EXPORT_SHAS,m.moving.LOADER_SHA,str(m.ENCODER),m.ENCODER_SHA,str(m.SOURCE),m.SOURCE_SHA)
    assert p['schema']==m.NAME+'.prepare' and len(p['slots'])==2 and len(m.canonical(p))<=256*1024
    assert not any('certificate' in path or path.endswith('.npz') for path in p['bindings_sha256'])
    assert 'torch' not in sys.modules
