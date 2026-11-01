"""Array-only arithmetic/full-size RAM mock; no model, physics or temp fixture."""
from copy import deepcopy
import gzip
import importlib.util
from pathlib import Path
import sys
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_matched_ridge_gn_cuda_20261027_v1 as m


spec=importlib.util.spec_from_file_location('matched_gn_mock',Path(__file__).with_name('test_p1_line_search_gn_cuda_20261027_v1.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def solve_once():
    rng=np.random.default_rng(20261027);a=rng.normal(size=(192,20));r=rng.normal(size=192)
    return a,r,m.algebra(a,r,lambda_mean=m.LAMBDA_MEAN)


def test_exact_scale_identity_and_unnormalized_normal_equations():
    a,r,delta=solve_once()
    assert 192*m.LAMBDA_MEAN==.002
    expected=-np.linalg.solve(a.T@a+.002*np.eye(20),a.T@r)
    np.testing.assert_allclose(delta,expected,rtol=1e-12,atol=1e-13)
    # D2P bounded_ridge starts with this solution when the cap is inactive.
    np.testing.assert_allclose((a.T@a+.002*np.eye(20))@delta,-a.T@r,rtol=1e-12,atol=1e-12)


def test_old_point002_mean_is_not_same_effective_coefficient():
    a,r,delta=solve_once();old=m.algebra(a,r,lambda_mean=.002)
    expected=-np.linalg.solve(a.T@a+.384*np.eye(20),a.T@r)
    np.testing.assert_allclose(old,expected,rtol=1e-12,atol=1e-13)
    assert not np.allclose(old,delta,rtol=1e-5,atol=1e-7)


@pytest.mark.parametrize('lam',[0.,-1.,float('nan'),float('inf')])
def test_bad_lambda(lam):
    with pytest.raises(ValueError):m.algebra(np.ones((192,20)),np.ones(192),lam)


class RAMStore(m.base.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.base.CAP-(0 if final else m.base.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw


@pytest.fixture(scope='module')
def mockrun():
    store=RAMStore();results=[];events=[];backends=[]
    before=(m.gn.DAMPING,m.gn.SCALES,m.gn.optimize)
    for slot in range(2):
        task=fixture.task(20261027+slot);backend=fixture.Backend(7+slot);backends.append(backend)
        ident=m.core.task_id(task)
        def sink(e,slot=slot):
            event=dict(source_slot=slot,arm=m.ARM,task_id=ident,event=deepcopy(e))
            events.append(event);store.event(event)
        meter=m.gn.Meter(sink)
        result=m.optimize(task,backend,meter,lambda_mean=m.LAMBDA_MEAN,updates=4,scales=m.SCALES)
        results.append((task,result,meter.counts()))
        store.put(f'root_{slot}.result.json.gz',result)
    store.close()
    assert (m.gn.DAMPING,m.gn.SCALES,m.gn.optimize)==before
    return store,results,events,backends


def test_complete_two_root_mock_budget_and_capacity(mockrun):
    store,results,events,backends=mockrun
    assert len(results)==2 and sum(map(len,store.files.values()))<m.base.CAP
    for i,(task,result,counts) in enumerate(results):
        assert counts=={k:dict(attempted=v,completed=v) for k,v in m.gn.budget().items()}
        assert counts['predict']['completed']==78 and counts['vjp_directions']['completed']==768
        assert len(backends[i].calls)==35 and len(backends[i].jac_points)==4
        assert not np.asarray(backends[i].jac_points[0]['logits']).any()
        assert result['lambda_mean']==.002/192
        assert result['scales']==list(m.SCALES)
        assert m.choice(task,result)['commands']==result['final_prediction']['commands']
        assert m.base.decode(gzip.decompress(store.files[f'root_{i}.result.json.gz']))==result
    plain,eof=m.base.inflate(store.files['events.jsonl.gz']);assert eof
    saved=[m.base.decode(line) for line in plain.splitlines()]
    assert saved==events
    assert sum(e['event']['kind']=='jacobian_row' for e in saved)==1536
    assert sum(e['event']['kind']=='native_packet' for e in saved)==140


def test_saved_direction_uses_new_explicit_lambda(mockrun):
    _,results,events,_=mockrun
    for slot,(task,result,_) in enumerate(results):
        for iteration in range(1,5):
            group=[e['event'] for e in events if e['source_slot']==slot and e['event'].get('evaluation_id')==f'iter_{iteration}_AD']
            d=next(e['derivative'] for e in group if e['kind']=='derivative_complete')
            p=next(e for e in group if e['kind']=='line_proposals_frozen')
            a=np.asarray(task['projection'])@np.asarray(d['raw_terminal_jacobian'])
            r=np.asarray(task['projection'])@(np.asarray(d['at_terminal'])-task['goal'])
            np.testing.assert_allclose(p['delta'],m.algebra(a,r,m.LAMBDA_MEAN),rtol=0,atol=0)
            assert p['lambda_mean']==m.LAMBDA_MEAN and p['lambda_unnormalized']==.002


@pytest.mark.parametrize('bad',['lambda','updates','scales'])
def test_no_runtime_parameter_retuning(bad):
    arguments=dict(lambda_mean=m.LAMBDA_MEAN,updates=4,scales=m.SCALES)
    if bad=='lambda':arguments['lambda_mean']=.002
    if bad=='updates':arguments['updates']=8
    if bad=='scales':arguments['scales']=(1.,.5)
    meter=m.gn.Meter(lambda e:None)
    with pytest.raises(ValueError):m.optimize(fixture.task(),fixture.Backend(),meter,**arguments)
    assert all(v['attempted']==0 for v in meter.counts().values())


@pytest.mark.parametrize('bad',['lambda','cost','selection','prediction','resources'])
def test_choice_rejects_wrong_result(mockrun,bad):
    _,results,_,_=mockrun;task,result,_=deepcopy(results[0])
    if bad=='lambda':result['lambda_mean']=.002
    if bad=='cost':result['steps'][0]['selected_cost']+=1
    if bad=='selection':result['steps'][0]['selected_index']=99
    if bad=='prediction':result['final_prediction']['commands'][0][0]+=.01
    if bad=='resources':result['resources']['predict']['completed']-=1
    with pytest.raises(ValueError):m.choice(task,result)


def plan_inputs():
    tasks=[fixture.task(20261027+i) for i in range(2)]
    slots=[dict(order=i,source_slot=i,blind_sha256=str(i+1)*64,task_export=f'/mock/task_{i:02d}.json.gz',
                task_sha256='a'*64,task_id=m.core.task_id(t)) for i,t in enumerate(tasks)]
    rows=[dict(source_slot=s['source_slot'],arm=m.ARM,blind_sha256=s['blind_sha256'],task_id=s['task_id'],status='not_started',commands=None,
               resources={},AD_API_resources={}) for s in slots]
    return tasks,dict(slots=slots),rows


def test_new_ownplans_and_physical_report_fields(monkeypatch,mockrun):
    tasks,p,rows=plan_inputs();store=RAMStore()
    monkeypatch.setattr(m.base,'load_export',lambda path,h:dict(gn_task=tasks[0 if '00.' in path else 1]))
    def runner(task,sink,row):
        index=0 if task==tasks[0] else 1
        result=deepcopy(mockrun[1][index][1]);row['resources']=result['resources'];row['AD_API_resources']=dict(grad=dict(attempted=768,completed=768),jacobian=dict(attempted=0,completed=0))
        return result
    m.plan_all(p,store,runner,rows);store.close()
    for i,row in enumerate(rows):
        assert row['source_slot']==i and row['arm']=='matched_ridge_GN' and row['blind_sha256']==str(i+1)*64
        assert row['status']=='model_plan_sealed' and np.asarray(row['commands']).shape==(10,2)
        plan=m.base.decode(store.files[row['plan_file']]);ident=plan.pop('selection_sha256')
        assert m.core.digest(plan)==ident==row['selection_sha256']
        assert plan['commands']==row['commands'] and plan['lambda_mean']==m.LAMBDA_MEAN


def test_partial_stops_before_second_root_and_keeps_journal(monkeypatch):
    tasks,p,rows=plan_inputs();store=RAMStore();calls=[]
    monkeypatch.setattr(m.base,'load_export',lambda *args:dict(gn_task=tasks[0]))
    def failed(task,sink,row):
        calls.append(task);sink(dict(kind='mock_attempt',completed=False));raise RuntimeError('first failure')
    with pytest.raises(RuntimeError,match='first failure'):m.plan_all(p,store,failed,rows)
    assert len(calls)==1 and rows[0]['status']=='attempted' and rows[1]['status']=='not_started'
    assert rows[0]['commands'] is rows[1]['commands'] is None
    raw,eof=m.base.inflate(store.files['events.jsonl.gz']);assert not eof
    assert b'first failure' not in raw and b'mock_attempt' in raw
    assert not any(name.endswith('.plan.json') for name in store.files)


def test_unauthorized_export_rejected_without_native_execution():
    with pytest.raises(ValueError,match='approved ordered'):m.build(['/mock/task.json'],['0'*64])
    assert not {'torch','stable_worldmodel','gymnasium','pymunk'} & set(sys.modules)


def test_protocol_keeps_unequal_cost_scope():
    c=m.contract()
    assert c['total_budget']['predict']==156 and c['total_budget']['jacobian']==8
    assert c['ADforward_in_predict_per_root']==8 and c['mathematical_vjp_per_root']==768
    assert c['equal_budget_claim'] is False and c['P1_admission'] is False


@pytest.mark.parametrize('damage',['none','missing','duplicate','wrong_lambda'])
def test_raw_journal_release_counts(mockrun,damage):
    events=[deepcopy(e['event']) for e in mockrun[2] if e['source_slot']==0]
    if damage=='missing':events.remove(next(e for e in events if e['kind']=='operation_completed'))
    if damage=='duplicate':events.append(next(e for e in events if e['kind']=='operation_completed'))
    if damage=='wrong_lambda':events.append(dict(kind='forbidden_AD_API'))
    if damage=='none':m.journal_counts(events,'operation_attempted','operation_completed','event',m.gn.budget())
    else:
        with pytest.raises(ValueError):m.journal_counts(events,'operation_attempted','operation_completed','event',m.gn.budget())


@pytest.mark.parametrize('damage',['none','plan_commands','partial','journal_return'])
def test_full_own_consume_RAM(mockrun,monkeypatch,damage):
    tasks,p,rows=plan_inputs();store=RAMStore();output=Path('/mock/matched_output')
    p.update(contract=m.contract(),bindings_sha256={})
    monkeypatch.setattr(m.base,'load_export',lambda path,h:dict(gn_task=tasks[0 if '00.' in path else 1]))
    def runner(task,sink,row):
        i=0 if task==tasks[0] else 1
        for e in mockrun[2]:
            if e['source_slot']==i:sink(deepcopy(e['event']))
        # Explicit stand-in for API instrumentation, not a real backward call.
        for k in range(768):
            v=dict(index=k,resource='grad',completed=False,is_grads_batched=False)
            sink(dict(kind='AD_API_attempt',entry=v));sink(dict(kind='AD_API_completed',entry=dict(v,completed=True)))
        result=deepcopy(mockrun[1][i][1]);row['resources']=result['resources'];row['AD_API_resources']=dict(grad=dict(attempted=768,completed=768),jacobian=dict(attempted=0,completed=0))
        return result
    store.put('binding.json',dict(prepared='/mock/prepare.json',prepare_sha256='a'*64))
    m.plan_all(p,store,runner,rows);store.close()
    if damage=='plan_commands':
        name=rows[0]['plan_file'];v=m.base.decode(store.files[name]);v['commands'][0][0]+=.1;store.files[name]=m.canonical(v)
    if damage=='journal_return':
        raw=gzip.decompress(store.files['events.jsonl.gz']);events=[m.base.decode(line) for line in raw.splitlines()]
        events.pop(next(k for k,e in enumerate(events) if e['event']['kind']=='operation_completed'))
        store.files['events.jsonl.gz']=gzip.compress(b''.join(m.canonical(e) for e in events))
    hashes={name:m.enc.digest(raw) for name,raw in store.files.items()}
    r=dict(schema=m.NAME+'.report',status='matched_ridge_model_plans_sealed_not_admitted',commands_released=True,
           errors=[],first_error=None,source_verified=True,model_unchanged=True,model_hash_before=m.enc.MODEL_TENSOR_SHA,
           model_hash_after=m.enc.MODEL_TENSOR_SHA,prepared='/mock/prepare.json',prepare_sha256='a'*64,contract=p['contract'],
           rows=rows,artifact_sha256=hashes,resources={k:dict(attempted=2*v,completed=2*v) for k,v in m.gn.budget().items()})
    if damage=='partial':r['status']='partial_or_failed'
    store.files['report.json']=m.canonical(r);reportsha=m.enc.digest(store.files['report.json'])
    monkeypatch.setattr(m,'verify',lambda *args:p)
    monkeypatch.setattr(m,'read',lambda path:m.base.decode(gzip.decompress(store.files[Path(path).name]) if str(path).endswith('.gz') else store.files[Path(path).name]))
    monkeypatch.setattr(m,'sha',lambda path:m.enc.digest(store.files[Path(path).name]))
    old_iterdir=Path.iterdir;old_read_bytes=Path.read_bytes
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(output/name for name in store.files) if path==output else old_iterdir(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==output else old_read_bytes(path))
    if damage=='none':
        result=m.consume(output,reportsha)
        assert result['status']=='own_model_commands_verified_not_admitted' and len(result['rows'])==2
        assert result['rows'][0]['arm']==m.ARM and result['new_physics']==0
    else:
        with pytest.raises(ValueError):m.consume(output,reportsha)
