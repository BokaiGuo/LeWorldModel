"""Kernel-faithfulness/nominal-dispatch NumPy mocks. No CPU neural or physics."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
from pathlib import Path
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_mppi_cuda_20261027_v2 as m

spec=importlib.util.spec_from_file_location('mppi_v2_array_mock',Path(__file__).with_name('test_p1_line_search_gn_cuda_20261027_v1.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class RAMStore(m.base.Store):
    def __init__(self):
        self.files={};self.events=[];self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.base.CAP-(0 if final else m.base.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw
    def event(self,e):self.events.append(deepcopy(e));super().event(e)


@pytest.fixture(scope='module')
def fullmock():
    store=RAMStore();slots=[];tasks=[];results=[];rows=[];backends=[]
    for s in (0,1):
        t=fixture.task(20261027+s);tasks.append(t)
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=f'task_{s}',task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        rows.append(dict(source_slot=s,arm=m.ARM,task_id=m.core.task_id(t),blind_sha256=str(s+2)*64,status='not_started',resources={}))
    original=m.base.load_export;m.base.load_export=lambda path,h:dict(gn_task=tasks[int(path[-1])])
    def runner(t,s,sink,row):
        b=fixture.Backend(7+s);backends.append(b);meter=m.Meter(sink)
        try:
            r=m.optimize(t,s,b,meter);results.append(r);return r
        finally:row['resources']=meter.counts()
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    try:m.plan_all(p,store,runner,rows)
    finally:m.base.load_export=original
    store.close();return store,p,tasks,results,rows,backends


def test_exact72_includes_initial_updated_nominals_and_fresh_dispatch(fullmock):
    store,p,tasks,results,rows,backends=fullmock
    for t,r,row,b in zip(tasks,results,rows,backends,strict=True):
        assert len(b.calls)==36 and not b.jac_points
        assert b.calls[-1]==('final_nominal_independent_H2','whole_replay')
        assert len(r['evaluations'])==35 and [len(x['parameters']) for x in r['batches']]==[1,16,1,16,1]
        assert [r['evaluations'][i]['stage'] for i in (0,17,34)]==['initial','updated_nominal','updated_nominal']
        assert row['resources']=={k:dict(attempted=72,completed=72) for k in m.budget()}
        assert m.choice(t,r,row['source_slot'])['commands']==r['evaluations'][34]['commands']
    assert m.contract()['total_budget']==dict(predict=144,action_encoder=144)
    assert m.contract()['kernel_H2_evaluations']+m.contract()['fresh_nominal_H2_replays']==36
    assert sum(map(len,store.files.values()))<m.base.CAP


def test_complete_saved_paths_ledger_and_RAM_gzip(fullmock):
    store,p,tasks,results,_,_=fullmock
    raw,eof=m.base.inflate(store.files['events.jsonl.gz']);assert eof
    assert [m.base.decode(x) for x in raw.splitlines()]==store.events
    for s in (0,1):m.journal_check(tasks[s],results[s],p['slots'][s],[e['event'] for e in store.events if e['source_slot']==s])
    assert sum(e['event']['kind']=='native_packet' for e in store.events)==144
    print('v2 two-root RAM journal/results/plans bytes:',sum(map(len,store.files.values())))


def test_actual_unmodified_kernel_receipt_and_every_parameter(fullmock):
    for s,r in enumerate(fullmock[3]):
        seen=[];offset=0
        def cost(batch):
            nonlocal offset
            seen.append(batch.copy());n=len(batch)
            assert not batch.flags.writeable
            vals=r['evaluations'][offset:offset+n]
            assert np.array_equal(batch,[v['parameters'] for v in vals])
            offset+=n;return np.array([v['cost'] for v in vals])
        best,receipt=m.kernels.mppi(cost,initial=np.zeros((2,10)),lower=-float(.75*np.sqrt(5)),upper=float(.75*np.sqrt(5)),
            samples=16,iterations=2,temperature=1.,noise_scale=.5,seed=20261029+101*s)
        assert receipt==r['kernel_receipt'] and best.tolist()==r['kernel_best_parameters'] and offset==35
        assert [x.shape[0] for x in seen]==[1,16,1,16,1]


def test_weighted_nominal_equations_in_parameter_not_projected_coordinates(fullmock):
    for r in fullmock[3]:
        for it,start,nom in ((0,1,17),(1,18,34)):
            chunk=r['evaluations'][start:start+16];costs=np.array([c['cost'] for c in chunk])
            weights=np.exp(-(costs-costs.min())/1.);weights/=weights.sum()
            params=np.array([c['parameters'] for c in chunk])
            expected=np.clip(np.tensordot(weights,params,axes=(0,0)),-float(.75*np.sqrt(5)),float(.75*np.sqrt(5)))
            assert np.array_equal(weights,r['updates'][it]['weights'])
            assert expected.tolist()==r['updates'][it]['nominal_parameters']==r['evaluations'][nom]['parameters']
            mapped_average=np.tensordot(weights,np.array([c['logits'] for c in chunk]),axes=(0,0))
            assert not np.array_equal(mapped_average,expected)


def test_dispatch_is_final_nominal_even_when_best_ever_is_strictly_better():
    t=fixture.task();t['goal']=deepcopy(t['root_context']['latent_history'][-1])
    r=m.optimize(t,0,fixture.Backend(),m.Meter(lambda e:None))
    assert r['diagnostic_best_index']==0 and r['evaluations'][0]['cost']==0
    assert r['selected_index']==34 and r['final_prediction']['cost']>0
    assert r['final_prediction']['commands']==r['evaluations'][34]['commands']
    assert r['kernel_best_parameters']!=r['evaluations'][34]['parameters']
    selected=m.choice(t,r,0);assert selected['commands']!=r['evaluations'][0]['commands']


def test_projection_changes_distribution_without_changing_legal_actions(fullmock):
    changed=0
    for r in fullmock[3]:
        for c in r['evaluations']:
            parameters=np.asarray(c['parameters']);x=np.asarray(c['logits'])
            assert (np.abs(parameters)<=float(.75*np.sqrt(5))).all()
            assert (np.linalg.norm(x,axis=1)<=float(.75*np.sqrt(5))).all()
            assert np.array_equal(x,x.astype(np.float32))
            assert np.array_equal(c['commands'],np.tanh(x.astype(np.float32)).reshape(10,2))
            changed+=int(not np.array_equal(parameters,x))
    assert changed>0
    assert m.contract()['projection_density_correction'] is False and m.contract()['official_MPPI'] is False


def test_stable_weights_equal_costs_offsets_and_extremes():
    assert np.array_equal(m.kernels.softmax_weights(np.ones(16),temperature=1.),np.full(16,1/16))
    a=np.arange(16,dtype=float)
    assert np.array_equal(m.kernels.softmax_weights(a,temperature=1.),m.kernels.softmax_weights(a+1000,temperature=1.))
    assert np.array_equal(m.kernels.softmax_weights([0.,1e308],temperature=1.),[1.,0.])


@pytest.mark.parametrize('bad',['parameters','logits','commands','cost','terminal','stage','seed','weights','nominal','receipt','bestdiagnostic','selection','bestsubstitution','replay','budget'])
def test_result_tamper_rejected(fullmock,bad):
    t=fullmock[2][0];r=deepcopy(fullmock[3][0])
    if bad=='parameters':r['evaluations'][1]['parameters'][0][0]+=.01
    elif bad=='logits':r['evaluations'][1]['logits'][0][0]+=.01
    elif bad=='commands':r['evaluations'][1]['commands'][0][0]+=.01
    elif bad=='cost':r['evaluations'][1]['cost']+=.01
    elif bad=='terminal':r['evaluations'][1]['terminal'][0]+=.01
    elif bad=='stage':r['evaluations'][34]['stage']='proposals'
    elif bad=='seed':r['seed']+=1
    elif bad=='weights':r['updates'][0]['weights'][0]+=.01
    elif bad=='nominal':r['updates'][0]['nominal_parameters'][0][0]+=.01
    elif bad=='receipt':r['kernel_receipt']['cost_evaluations']=32
    elif bad=='bestdiagnostic':r['diagnostic_best_index']=(r['diagnostic_best_index']+1)%35
    elif bad=='selection':r['selection']='best_sample'
    elif bad=='bestsubstitution':r['selected_index']=r['diagnostic_best_index'] if r['diagnostic_best_index']!=34 else 0
    elif bad=='replay':r['final_prediction']['contexts'][-1]['action_history'][-1][0]+=.01
    else:r['resources']['predict']['completed']=66
    with pytest.raises(ValueError):m.choice(t,r,0)


@pytest.mark.parametrize('bad',['early_dispatch','late_bank','update_weights','extraAD','history','missing_return','bestdispatch'])
def test_saved_chronology_and_real_call_arrays_rejected(fullmock,bad):
    store,p,tasks,results,_,_=fullmock;events=deepcopy([e['event'] for e in store.events if e['source_slot']==0])
    if bad=='early_dispatch':
        i=next(i for i,e in enumerate(events) if e['kind']=='nominal_dispatch_frozen');events.insert(1,events.pop(i))
    elif bad=='late_bank':
        i=next(i for i,e in enumerate(events) if e['kind']=='kernel_batch_frozen');events.append(events.pop(i))
    elif bad=='update_weights':next(e for e in events if e['kind']=='kernel_update')['weights'][0]+=.01
    elif bad=='extraAD':events.insert(1,dict(kind='AD_API_attempt'))
    elif bad=='history':next(e for e in events if e['kind']=='native_packet')['context']['action_history'][0][0]+=.01
    elif bad=='missing_return':events.pop(next(i for i,e in enumerate(events) if e['kind']=='operation_completed'))
    else:next(e for e in events if e['kind']=='nominal_dispatch_frozen')['index']=results[0]['diagnostic_best_index'] if results[0]['diagnostic_best_index']!=34 else 0
    with pytest.raises(ValueError):m.journal_check(tasks[0],results[0],p['slots'][0],events)


def test_failure_preserves_actual_partial_and_kernel_attempted_batch_scope():
    events=[];meter=m.Meter(events.append)
    class Broken(fixture.Backend):
        def rollout(self,t,raw,meter,mode):
            if len(self.calls)==2:meter.begin('predict');raise RuntimeError('actual model failure')
            return super().rollout(t,raw,meter,mode)
    with pytest.raises(m.kernels.CostEvaluationError,match='actual model failure') as caught:m.optimize(fixture.task(),0,Broken(),meter)
    assert meter.counts()['predict']==dict(attempted=5,completed=4)
    assert len([e for e in events if e['kind']=='kernel_evaluation_saved'])==2
    assert caught.value.receipt['cost_evaluations_attempted']==17  # declared whole batch, not actual forwards
    assert next(e for e in events if e['kind']=='kernel_failure')['kernel_receipt']==caught.value.receipt
    assert not any(e['kind']=='nominal_dispatch_frozen' for e in events)


def test_replay_failure_or_mismatch_cannot_release_nominal():
    class Wrong(fixture.Backend):
        def rollout(self,t,raw,meter,mode):
            path=super().rollout(t,raw,meter,mode)
            if meter.group=='final_nominal_independent_H2':path[-1]['latent_history'][-1][0]+=1.
            return path
    with pytest.raises(ValueError,match='nominal search/fresh replay'):m.optimize(fixture.task(),0,Wrong(),m.Meter(lambda e:None))


def test_predispatch_cap_and_frozen_helpers():
    assert m.sha(m.kernels.__file__)==m.KERNEL_SHA and m.sha(m.sampling.__file__)==m.SAMPLING_SHA
    meter=m.Meter(lambda e:None)
    for _ in range(72):meter.call('predict',lambda:None)
    with pytest.raises(ValueError):meter.begin('predict')
    with pytest.raises(ValueError):m.Meter(lambda e:None).begin('jacobian')
    assert m.TEMPERATURE==1. and m.NOISE_SCALE==.5


def test_complete_RAM_nominal_consumer(fullmock,monkeypatch):
    original,p,tasks,results,rows,_=fullmock;store=RAMStore();store.files=deepcopy(original.files)
    store.put('binding.json',dict(prepared='RAM_prepare',prepare_sha256='a'*64));digest=lambda raw:hashlib.sha256(raw).hexdigest()
    report=dict(schema=m.NAME+'.report',status='kernel_nominal_plans_sealed_not_admitted',commands_released=True,errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.enc.MODEL_TENSOR_SHA,model_hash_after=m.enc.MODEL_TENSOR_SHA,
        prepared='RAM_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,resources={k:dict(attempted=144,completed=144) for k in m.budget()},
        artifact_sha256={n:digest(v) for n,v in store.files.items()})
    store.put('report.json',report,final=True);expected=digest(store.files['report.json'])
    prefix=Path('/RAM_kernel_nominal_v2');actual_sha=m.sha;actual_read=m.read;actual_iter=Path.iterdir;actual_bytes=Path.read_bytes
    monkeypatch.setattr(m,'verify',lambda path,h:p)
    monkeypatch.setattr(m.base,'load_export',lambda path,h:dict(gn_task=tasks[int(path[-1])]))
    monkeypatch.setattr(m,'sha',lambda path:digest(store.files[Path(path).name]) if Path(path).parent==prefix else actual_sha(path))
    def read(path):
        path=Path(path)
        if path.parent!=prefix:return actual_read(path)
        raw=store.files[path.name];return m.base.decode(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([prefix/n for n in store.files]) if path==prefix else actual_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==prefix else actual_bytes(path))
    consumed=m.consume(prefix,expected);assert len(consumed['rows'])==2 and not consumed['paper_strong_MPPI_requirement_closed']
    assert all(p['selected_index']==34 and p['selection']=='last_updated_nominal_not_best' for p in consumed['rows'])
    store.files[rows[0]['plan_file']]+=b' '
    with pytest.raises(ValueError,match='artifact SHA'):m.consume(prefix,expected)
