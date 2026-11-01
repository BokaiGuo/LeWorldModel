"""NumPy/RAM engineering fixtures only, never a CPU neural experiment."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
from pathlib import Path
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_mppi_cuda_20261027_v1 as m

spec=importlib.util.spec_from_file_location('mppi_array_mock',Path(__file__).with_name('test_p1_line_search_gn_cuda_20261027_v1.py'))
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
    # No disk fixture and no producer output/physical reads.
    original=m.base.load_export
    m.base.load_export=lambda path,h:dict(gn_task=tasks[int(path[-1])])
    def runner(t,s,sink,row):
        b=fixture.Backend(7+s);backends.append(b);meter=m.Meter(sink)
        try:
            r=m.optimize(t,s,b,meter);results.append(r);return r
        finally:row['resources']=meter.counts()
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    try:m.plan_all(p,store,runner,rows)
    finally:m.base.load_export=original
    store.close();return store,p,tasks,results,rows,backends


def test_both_roots_exact66_and_no_hidden_nominal_cost(fullmock):
    store,p,tasks,results,rows,backends=fullmock
    for t,r,row,b in zip(tasks,results,rows,backends,strict=True):
        assert len(r['candidates'])==32 and len(r['updates'])==2
        assert len(b.calls)==33 and not b.jac_points
        assert b.calls[-1]==('winner_independent_H2','whole_replay')
        assert row['resources']=={k:dict(attempted=66,completed=66) for k in m.budget()}
        assert m.choice(t,r,row['source_slot'])['commands']==r['final_prediction']['commands']
        assert not any(k in r['final_prediction'] for k in ('whole_replay','segmented_replay'))
    assert sum(map(len,store.files.values()))<m.base.CAP
    assert m.contract()['total_budget']==dict(predict=132,action_encoder=132)
    assert m.contract()['official_MPPI'] is False


def test_saved_grammar_and_RAM_lossless(fullmock):
    store,p,tasks,results,_,_=fullmock
    plain,eof=m.base.inflate(store.files['events.jsonl.gz']);assert eof
    assert [m.base.decode(x) for x in plain.splitlines()]==store.events
    for s in (0,1):
        events=[e['event'] for e in store.events if e['source_slot']==s]
        m.journal_check(tasks[s],results[s],p['slots'][s],events)
    assert sum(e['event']['kind']=='native_packet' for e in store.events)==132
    print('MPPI two-root RAM bytes:',sum(map(len,store.files.values())))


def test_seeds_and_fixed_algorithm_constants():
    assert [m.seed(s) for s in (0,1)]==[20261029,20261130]
    assert m.TEMPERATURE==1. and m.NOISE_SCALE==.5 and m.ITERATIONS==2 and m.SAMPLES==16
    assert m.contract()['candidate_predict']+m.contract()['winner_predict']==66
    with pytest.raises(ValueError):m.seed(True)
    with pytest.raises(ValueError):m.seed(2)


def weight_bank(costs):
    xs=np.random.default_rng(47).normal(size=(16,2,10))
    return [dict(cost=float(c),logits=m.gn.screen.projected_logits(x,float(.75*np.sqrt(5))).tolist()) for c,x in zip(costs,xs,strict=True)]


def test_weights_equal_costs_and_convex_projected_update():
    bank=weight_bank(np.ones(16));w,x=m.weight_update(bank,float(.75*np.sqrt(5)))
    assert np.array_equal(w,np.full(16,1/16))
    want=m.gn.screen.projected_logits(np.asarray([c['logits'] for c in bank]).mean(axis=0),float(.75*np.sqrt(5)))
    np.testing.assert_allclose(x,want,rtol=0,atol=1e-7)


def test_weights_shift_invariance_and_extreme_underflow():
    a=weight_bank(np.arange(16));b=deepcopy(a)
    for c in b:c['cost']+=1000
    wa,xa=m.weight_update(a,float(.75*np.sqrt(5)));wb,xb=m.weight_update(b,float(.75*np.sqrt(5)))
    assert np.array_equal(wa,wb) and np.array_equal(xa,xb)
    a=weight_bank([0.]+[1e308]*15);w,x=m.weight_update(a,float(.75*np.sqrt(5)))
    assert np.array_equal(w,[1.]+[0.]*15) and np.isfinite(x).all()


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_invalid_weights_input_rejected(bad):
    a=weight_bank(np.ones(16));a[0]['cost']=bad
    with pytest.raises(ValueError):m.weight_update(a,float(.75*np.sqrt(5)))


def test_proposals_and_updates_reconstruct_without_outcome(fullmock):
    _,_,tasks,results,_,_=fullmock
    for s,r in enumerate(results):
        rng=np.random.Generator(np.random.PCG64(20261029+101*s));nominal=np.zeros((2,10))
        for it in (0,1):
            xs=[m.gn.screen.projected_logits(x,float(.75*np.sqrt(5))) for x in nominal+.5*rng.standard_normal((16,2,10))]
            chunk=r['candidates'][it*16:(it+1)*16]
            assert [c['logits'] for c in chunk]==[x.tolist() for x in xs]
            for c,x in zip(chunk,xs,strict=True):
                assert np.array_equal(c['commands'],np.tanh(x.astype(np.float32)).reshape(10,2))
                assert c['cost']==m.core.model_cost(tasks[s],c['terminal'])
            c=np.asarray([v['cost'] for v in chunk]);w=np.exp(-(c-c.min()));w/=w.sum()
            nominal=m.gn.screen.projected_logits(np.einsum('i,ijk->jk',w,np.asarray(xs)),float(.75*np.sqrt(5)))
            np.testing.assert_allclose(r['updates'][it]['nominal'],nominal,rtol=0,atol=1e-7)
            # Producer rounding controls the next proposal's exact float32 identity.
            nominal=np.asarray(r['updates'][it]['nominal'])


@pytest.mark.parametrize('bad',['seed','candidate','commands','cost','terminal','weights','temperature','nominal','winner','budget','replay','tripleclaim'])
def test_choice_tamper_rejected(fullmock,bad):
    t=fullmock[2][0];r=deepcopy(fullmock[3][0])
    if bad=='seed':r['seed']+=1
    elif bad=='candidate':r['candidates'][0]['logits'][0][0]+=.01
    elif bad=='commands':r['candidates'][0]['commands'][0][0]+=.01
    elif bad=='cost':r['candidates'][0]['cost']+=.01
    elif bad=='terminal':r['candidates'][0]['terminal'][0]+=.01
    elif bad=='weights':r['updates'][0]['weights'][0]+=.01
    elif bad=='temperature':r['updates'][0]['temperature']=.5
    elif bad=='nominal':r['updates'][0]['nominal'][0][0]+=.01
    elif bad=='winner':r['winner_index']=(r['winner_index']+1)%32
    elif bad=='budget':r['resources']['predict']['completed']=64
    elif bad=='replay':r['final_prediction']['contexts'][-1]['action_history'][-1][0]+=.01
    else:r['final_prediction']['whole_replay']=r['final_prediction']['contexts'][-1]
    with pytest.raises(ValueError):m.choice(t,r,0)


@pytest.mark.parametrize('bad',['earlywinner','latebank','extraAD','history','missingreturn','weights'])
def test_journal_scope_and_chronology_rejected(fullmock,bad):
    store,p,tasks,results,_,_=fullmock;events=deepcopy([e['event'] for e in store.events if e['source_slot']==0])
    if bad=='earlywinner':
        i=next(i for i,e in enumerate(events) if e['kind']=='winner_frozen_before_replay');events.insert(1,events.pop(i))
    elif bad=='latebank':
        i=next(i for i,e in enumerate(events) if e['kind']=='MPPI_bank_frozen');events.append(events.pop(i))
    elif bad=='extraAD':events.insert(1,dict(kind='AD_API_attempt'))
    elif bad=='history':next(e for e in events if e['kind']=='native_packet')['context']['action_history'][0][0]+=.01
    elif bad=='missingreturn':events.pop(next(i for i,e in enumerate(events) if e['kind']=='operation_completed'))
    else:next(e for e in events if e['kind']=='MPPI_update')['weights'][0]+=.01
    with pytest.raises(ValueError):m.journal_check(tasks[0],results[0],p['slots'][0],events)


def test_failure_preserves_partial_and_unreturned_attempt():
    t=fixture.task();events=[];meter=m.Meter(events.append)
    class Broken(fixture.Backend):
        def rollout(self,t,raw,meter,mode):
            if len(self.calls)==2:meter.begin('predict');raise RuntimeError('predict failure')
            return super().rollout(t,raw,meter,mode)
    with pytest.raises(RuntimeError,match='predict failure'):m.optimize(t,0,Broken(),meter)
    assert len([e for e in events if e['kind']=='candidate_saved'])==2
    assert meter.counts()['predict']==dict(attempted=5,completed=4)
    assert not any(e['kind']=='winner_frozen_before_replay' for e in events)


def test_single_independent_replay_mismatch_rejected():
    class Wrong(fixture.Backend):
        def rollout(self,t,raw,meter,mode):
            p=super().rollout(t,raw,meter,mode)
            if meter.group=='winner_independent_H2':p[-1]['latent_history'][-1][0]+=1.
            return p
    with pytest.raises(ValueError,match='search/independent replay'):m.optimize(fixture.task(),0,Wrong(),m.Meter(lambda e:None))


def test_predispatch66_cap_and_no_AD_resources():
    meter=m.Meter(lambda e:None)
    for _ in range(66):meter.call('predict',lambda:None)
    with pytest.raises(ValueError):meter.begin('predict')
    for resource in ('jacobian','vjp_directions','svd','linear_solve'):
        with pytest.raises(ValueError):m.Meter(lambda e:None).begin(resource)


def test_sealed_reference_source_and_approved_tasks_only():
    assert m.sha(m.sampling.__file__)==m.SAMPLING_SHA
    for path,h in zip(m.sampling.EXPORTS,m.sampling.EXPORT_SHAS,strict=True):
        t=m.base.load_export(path,h)['gn_task']
        assert (t['horizon'],t['latent_dim'],t['control_dim'])==(2,192,2)


def test_complete_RAM_consume(fullmock,monkeypatch):
    original,p,tasks,results,rows,_=fullmock;store=RAMStore()
    store.files=deepcopy(original.files);store.put('binding.json',dict(prepared='RAM_prepare',prepare_sha256='a'*64))
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    report=dict(schema=m.NAME+'.report',status='mppi_model_plans_sealed_not_admitted',commands_released=True,errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.enc.MODEL_TENSOR_SHA,model_hash_after=m.enc.MODEL_TENSOR_SHA,
        prepared='RAM_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,resources={k:dict(attempted=132,completed=132) for k in m.budget()},
        artifact_sha256={n:digest(v) for n,v in store.files.items()})
    store.put('report.json',report,final=True);expected=digest(store.files['report.json'])
    prefix=Path('/RAM_constructive_MPPI');actual_sha=m.sha;actual_read=m.read;actual_iter=Path.iterdir;actual_bytes=Path.read_bytes
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
    result=m.consume(prefix,expected);assert len(result['rows'])==2 and not result['official_MPPI'] and not result['P1_admission']
    store.files[rows[0]['plan_file']]+=b' '
    with pytest.raises(ValueError,match='artifact hash'):m.consume(prefix,expected)
