"""RAM/NumPy execution contracts only; no torch/model/simulator/temp files."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as m


def helper(stem):
    spec=importlib.util.spec_from_file_location(stem,Path(__file__).with_name(stem+'.py'))
    x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);return x


h=helper('test_p1_constructive_blind_planners_cuda_20261027_v1')
enc=helper('test_p1_constructive_moving_history_encode_cuda_20261027_v1')


def inputs(seed=17):
    i=h.inputs(seed);t=i['gn_task'];t['root_raw_action_history']=m.fixed_raw().astype(float).tolist()
    t['root_context']['action_history']=m.core.normalized_tokens(t,m.fixed_raw()).tolist()
    i.update(d2p_packet=dict(root_context=deepcopy(t['root_context']),goal=t['goal']),task_id=m.core.task_id(t))
    return m.interface(i)


def test_nonzero_interface_never_calls_old_gate(monkeypatch):
    i=inputs()
    monkeypatch.setattr(m.blind,'interface',lambda *_:pytest.fail('zero-only old gate'))
    assert m.interface(i)==i and i['gn_task']['root_context']['action_history'][0]!=i['gn_task']['root_context']['action_history'][1]
    assert m.contract()['per_root_budget'][m.ARMS[0]]['predict']==78
    assert m.contract()['total_budget'][m.ARMS[0]]['predict']==156
    assert m.contract()['mathematical_vjp_per_root'][m.ARMS[0]]==768
    assert m.contract()['per_root_budget'][m.ARMS[1]]==dict(predict=66,action_encoder=66,jacobian=11,svd=12)


@pytest.mark.parametrize('bad',('zero','reverse','divide100','double','metadata','normalizer','Q','id','model','extra'))
def test_interface_rejects_changed_history_or_task(bad):
    i=inputs();t=i['gn_task']
    if bad in ('zero','reverse','divide100','double'):
        raw=m.fixed_raw().copy()
        if bad=='zero':raw[:]=0
        elif bad=='reverse':raw=raw[::-1]
        elif bad=='divide100':raw/=100
        else:raw*=2
        t['root_raw_action_history']=raw.tolist();t['root_context']['action_history']=m.core.normalized_tokens(t,raw).tolist()
    elif bad=='metadata':t['root_context']['context_length']=2
    elif bad=='normalizer':t['action_mean']=[0.,0.]
    elif bad=='Q':i['q']=np.zeros((192,192)).tolist()
    elif bad=='id':i['task_id']='a'*64
    elif bad=='model':t['model_sha256']='b'*64
    else:i['witness']='forbidden'
    with pytest.raises(ValueError):m.interface(i)


def test_load_task_own_schema_and_real_nonzero_export_in_RAM(monkeypatch):
    e=enc.exported();called=[]
    monkeypatch.setattr(m,'sha',lambda _:'a'*64);monkeypatch.setattr(m,'read',lambda *_:deepcopy(e))
    def load(path,h):called.append((str(path),h));return m.moving.task_interfaces(e)
    monkeypatch.setattr(m.moving,'load_task',load)
    monkeypatch.setattr(m.blind,'load_export',lambda *_:pytest.fail('old loader'))
    i=m.load_export('/RAM/task_00.json.gz','a'*64)
    assert called==[('/RAM/task_00.json.gz','a'*64)] and np.array_equal(i['gn_task']['root_raw_action_history'],m.fixed_raw())
    with pytest.raises(ValueError):m.load_export('/RAM/sourcepack.json','a'*64)
    with pytest.raises(ValueError):m.load_export('/RAM/task_00.json.gz','b'*64)


def mock_runner(arm,i,sink,row):
    count=0
    def trace(before,raw):
        nonlocal count
        sink(dict(kind='native_context_attempt',index=count,context=m.incoming(i['gn_task'],before,raw)));count+=1
    if arm==m.ARMS[0]:
        class GN(h.g.Backend):
            def forward(self,t,raw,meter,kind,mode=None):
                path=self.path(t,raw);before=t['root_context']
                for k,c in enumerate(path):
                    trace(before,np.asarray(raw)[5*k:5*k+5]);before=c
                    for resource in ('action_encoder','predict'):meter.end(meter.begin(resource))
                    meter.emit(kind,packet=k,context=c,**({} if mode is None else dict(mode=mode)))
                return path
        ledger=m.blind.gn.Meter(sink);r=m.blind.gn.optimize(i['gn_task'],GN(),ledger)
    else:
        ledger=m.blind.D2PLedger(sink)
        class D2P(h.DBackend):
            def step(self,c,a):trace(c,a);return super().step(c,a)
        r=m.blind.d2p.compile_arm(i['d2p_packet'],i['q'],D2P(ledger),ledger.journal)
    row['resources']=ledger.counts()
    # Explicit synthetic AD API bookkeeping: no real autograd/model is called.
    for index,kind in enumerate([k for k,n in m.contract()['AD_API_per_root'][arm].items() for _ in range(n)]):
        entry=dict(index=index,resource=kind,completed=False,is_grads_batched=arm==m.ARMS[1])
        sink(dict(kind='AD_API_attempt',entry=deepcopy(entry)))
        sink(dict(kind='AD_API_completed',entry=dict(entry,completed=True)))
    row['AD_API_resources']={k:dict(attempted=v,completed=v) for k,v in m.contract()['AD_API_per_root'][arm].items()}
    return r


@pytest.fixture(scope='module')
def full():
    i=inputs();out={};events={};rows={}
    for a in m.ARMS:
        events[a]=[];rows[a]={};out[a]=mock_runner(a,i,events[a].append,rows[a])
    return i,out,events,rows


@pytest.mark.parametrize('arm',m.ARMS)
def test_all_original_kernels_nonzero_full_history_and_ledger(full,arm):
    i,out,events,rows=full
    m.blind.exact_counts(rows[arm]['resources'],m.blind.budgets()[arm]);m.ledger_check(arm,events[arm])
    receipt=m.history_check(arm,i,out[arm],events[arm])
    assert receipt['native_contexts']==(78 if arm==m.ARMS[0] else 66)
    assert receipt['all_actual_nonzero_history_paths_verified']
    assert len(m.blind.selected(arm,i,out[arm])['commands'])==10


@pytest.mark.parametrize('arm',m.ARMS)
@pytest.mark.parametrize('bad',('rawroot','tracezero','tracelatent','missing','reorder','return','selectedcommands'))
def test_fullpath_tamper_rejected(full,arm,bad):
    i,out,events,rows=deepcopy(full);r=out[arm];es=events[arm]
    trace=next(e for e in es if e['kind']=='native_context_attempt')
    if bad=='rawroot':i['gn_task']['root_context']['action_history']=np.zeros((2,10)).tolist()
    elif bad=='tracezero':trace['context']['action_history'][0]=[0.]*10
    elif bad=='tracelatent':trace['context']['latent_history'][0][0]+=1
    elif bad=='missing':es.remove(trace)
    elif bad=='reorder':trace['index']=1
    elif bad=='return':es[:]=[e for e in es if e['kind']!='AD_API_completed']
    elif arm==m.ARMS[0]:r['final_prediction']['commands'][0][0]+=.001
    else:r['winner']['commands'][0][0]+=.001
    with pytest.raises(ValueError):
        m.blind.selected(arm,i,r);m.ledger_check(arm,es);m.history_check(arm,i,r,es)


def plan_fixture(monkeypatch):
    ii=[inputs(17+s) for s in (0,1)]
    slots=[dict(order=s,source_slot=s,task_id=ii[s]['task_id'],task_export=f'/RAM/task_{s:02d}.json.gz',task_sha256=chr(97+s)*64,blind_sha256=chr(99+s)*64) for s in (0,1)]
    monkeypatch.setattr(m,'load_export',lambda p,h:deepcopy(ii[int(str(p)[-10:-8])]))
    rows=[dict(order=s['order'],source_slot=s['source_slot'],task_id=s['task_id'],arm=a,status='not_started',resources={},AD_API_resources={}) for s in slots for a in m.ARMS]
    return dict(slots=slots),rows


def test_all_four_own_plans_lossless_RAM_and_8MiBcap(monkeypatch):
    p,rows=plan_fixture(monkeypatch);store=h.RAMStore()
    m.plan_all(p,store,mock_runner,rows);store.close()
    assert len(rows)==4 and all(r['status']=='model_plan_sealed' for r in rows)
    assert sum(map(len,store.files.values()))<m.CAP
    for row in rows:
        plan=m.blind.decode(store.files[row['plan_file']]);ident=plan.pop('selection_sha256')
        assert plan['schema']==m.NAME+'.model_plan' and m.core.digest(plan)==ident
        assert plan['source_slot']==row['order'] and not plan['physical_execution']
    plain,eof=m.blind.inflate(store.files['events.jsonl.gz']);assert eof and plain.endswith(b'\n')


def test_failure_preserves_four_census_and_no_retry(monkeypatch):
    p,rows=plan_fixture(monkeypatch);store=h.RAMStore();calls=[]
    def fail(arm,i,sink,row):
        calls.append(arm);sink(dict(kind='partial_native_input'));row['resources']=dict(predict=dict(attempted=1,completed=0));raise RuntimeError('first failure')
    with pytest.raises(RuntimeError,match='first failure'):m.plan_all(p,store,fail,rows)
    assert calls==[m.ARMS[0]] and rows[0]['resources']['predict']==dict(attempted=1,completed=0)
    assert [r['status'] for r in rows]==['attempted']+['not_started']*3


def test_prepare_refuses_missing_paths_or_external_hash_before_loading(monkeypatch):
    with pytest.raises(ValueError):m.build([],[],'not-a-sha')
    monkeypatch.setattr(m,'sha',lambda path:m.BLIND_SHA if str(path)==str(m.blind.__file__) else m.LOADER_SHA)
    with pytest.raises(ValueError,match='exactly two'):m.build([],[],m.LOADER_SHA)
    with pytest.raises(ValueError,match='fixed two export'):m.build(m.EXPORTS,['a'*64,'b'*64],m.LOADER_SHA)
    with pytest.raises(ValueError,match='frozen primitive'):m.build(m.EXPORTS,m.EXPORT_SHAS,'f'*64)


def test_run_invalid_prepare_never_creates_output_or_imports_model(monkeypatch):
    def fail(*a):raise ValueError('external prepare failed')
    monkeypatch.setattr(m,'verify',fail);monkeypatch.setattr(m,'Store',lambda *_:pytest.fail('store before verify'))
    with pytest.raises(ValueError,match='external prepare'):m.run('RAM','x','RAM')


def test_actual_pre_hooks_read_only_and_cleanup_no_torch():
    class Module:
        def __init__(self):self.hooks=[]
        def register_forward_pre_hook(self,fn):
            self.hooks.append(fn)
            return NS(remove=lambda:self.hooks.remove(fn))
        def fire(self,x):
            for fn in self.hooks:fn(self,(x,))
    class Array:
        device=NS(type='cuda')
        def __init__(self,a):self.a=a;self.shape=a.shape
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.a
    model=NS(action_encoder=Module(),predictor=Module());es=[];i=inputs();t=i['gn_task']
    want=m.incoming(t,t['root_context'],np.zeros((5,2)))
    with m.context_trace(model,es.append):
        model.action_encoder.fire(Array(np.asarray([want['action_history']],np.float32)))
        model.predictor.fire(Array(np.asarray([want['latent_history']],np.float32)))
    assert es==[dict(kind='native_context_attempt',index=0,context=want)]
    assert model.action_encoder.hooks==model.predictor.hooks==[]


@pytest.fixture
def saved_RAM(monkeypatch):
    p,rows=plan_fixture(monkeypatch);store=h.RAMStore();ph='f'*64
    store.put('binding.json',dict(prepared='/RAM/prepare',prepare_sha256=ph))
    m.plan_all(p,store,mock_runner,rows);store.close()
    p.update(contract=m.contract(),expected_model_hash='e'*64,bindings_sha256={})
    report=dict(schema=m.NAME+'.report',status='moving_history_plans_sealed_not_admitted',
        commands_released=True,errors=[],first_error=None,source_verified=True,model_unchanged=True,
        model_hash_before='e'*64,model_hash_after='e'*64,contract=m.contract(),prepared='/RAM/prepare',prepare_sha256=ph,
        rows=rows,artifact_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in store.files.items()})
    files={'/RAM/run/'+n:raw for n,raw in store.files.items()}
    def publish():
        files['/RAM/run/report.json']=m.canonical(report)
        return hashlib.sha256(files['/RAM/run/report.json']).hexdigest()
    class RAMPath:
        def __init__(self,p):self.p=str(p)
        def __str__(self):return self.p
        def __truediv__(self,n):return RAMPath(self.p+'/'+n)
        def absolute(self):return self
        @property
        def name(self):return self.p.rsplit('/',1)[-1]
        def read_bytes(self):return files[self.p]
        def iterdir(self):return [RAMPath(k) for k in files if k.startswith(self.p+'/')]
    def read(path,*a):
        raw=files[str(path)]
        if str(path).endswith('.gz'):raw=gzip.decompress(raw)
        return m.blind.decode(raw)
    monkeypatch.setattr(m,'Path',RAMPath);monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m,'sha',lambda p:hashlib.sha256(files[str(p)]).hexdigest())
    monkeypatch.setattr(m,'verify',lambda path,h:deepcopy(p))
    return files,report,publish


def test_own_consume_all_four_nonzero_plans_in_RAM(saved_RAM):
    files,report,publish=saved_RAM;r=m.consume('/RAM/run',publish())
    assert r['schema']==m.NAME+'.consume' and len(r['plans'])==4 and r['new_physics']==0 and not r['P1_admission']
    assert [(x['source_slot'],x['arm']) for x in r['plans']]==[(s,a) for s in (0,1) for a in m.ARMS]
    assert sum(map(len,files.values()))<m.CAP
    print('RAM full four-arm artifacts bytes:',sum(map(len,files.values())))


@pytest.mark.parametrize('bad',('oldschema','partial','census','resources','roworder','AD','artifact'))
def test_own_consume_rejects_partial_or_relabelled_output(saved_RAM,bad):
    files,r,publish=saved_RAM
    if bad=='oldschema':r['schema']=m.blind.NAME+'.report'
    elif bad=='partial':r['commands_released']=False
    elif bad=='census':files['/RAM/run/extra.json']=b'{}'
    elif bad=='resources':r['rows'][0]['resources']['predict']['completed']-=1
    elif bad=='roworder':r['rows'].reverse()
    elif bad=='AD':r['rows'][0]['AD_API_resources']['grad']['completed']-=1
    else:files['/RAM/run/'+r['rows'][0]['plan_file']]=b'{}'
    with pytest.raises(ValueError):m.consume('/RAM/run',publish())
