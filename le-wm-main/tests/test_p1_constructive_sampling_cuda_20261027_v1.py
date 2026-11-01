"""Full-size NumPy/RAM mocks only; no neural, physics, network or tmp files."""
from copy import deepcopy
import gzip
import importlib.util
from pathlib import Path
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_sampling_cuda_20261027_v1 as m

spec=importlib.util.spec_from_file_location('sampling_array_mock',Path(__file__).with_name('test_p1_line_search_gn_cuda_20261027_v1.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class RAMStore(m.base.Store):
    def __init__(self):
        self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.base.CAP-(0 if final else m.base.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw


@pytest.fixture(scope='module')
def fullmock():
    store=RAMStore();rows=[];events=[];results=[]
    tasks=[fixture.task(20261027+i) for i in range(2)]
    for slot,t in enumerate(tasks):
        for arm in m.ARMS:
            backend=fixture.Backend(7+slot)
            def sink(e,slot=slot,arm=arm,t=t):
                v=dict(source_slot=slot,arm=arm,task_id=m.core.task_id(t),event=deepcopy(e));events.append(v);store.event(v)
            meter=m.Meter(sink);r=m.optimize(t,slot,arm,backend,meter)
            store.put(f'root_{slot}_{arm}.result.json.gz',r)
            rows.append(meter.counts());results.append((t,slot,arm,r,backend))
    store.close();return store,rows,events,results


def test_all_four_fullH2_exact134_and_resource_vector(fullmock):
    store,rows,events,results=fullmock
    assert len(rows)==4 and sum(map(len,store.files.values()))<m.base.CAP
    for counts in rows:assert counts=={k:dict(attempted=134,completed=134) for k in m.budget()}
    for t,s,a,r,b in results:
        assert len(b.calls)==67 and len(b.jac_points)==0
        assert [x[1] for x in b.calls[-3:]]==['nominal','whole_replay','segmented_replay']
        assert len(r['candidates'])==64
        assert m.choice(t,r,s,a)['commands']==r['final_prediction']['commands']
    assert m.contract()['total_budget']==dict(predict=536,action_encoder=536)
    assert m.contract()['AD_API']==m.contract()['mathematical_VJP']==0


def test_RAM_lossless_and_small_artifacts(fullmock):
    store,_,events,results=fullmock
    plain,eof=m.base.inflate(store.files['events.jsonl.gz']);assert eof
    assert [m.base.decode(x) for x in plain.splitlines()]==events
    for _,s,a,r,_ in results:assert m.base.decode(gzip.decompress(store.files[f'root_{s}_{a}.result.json.gz']))==r
    assert sum(e['event']['kind']=='native_packet' for e in events)==536
    assert sum(e['event']['kind']=='candidate_saved' for e in events)==256


def test_all_saved_paths_and_preselection_rebuilt(fullmock):
    for t,s,a,r,_ in fullmock[3]:
        group=[e['event'] for e in fullmock[2] if e['source_slot']==s and e['arm']==a]
        m.journal_check(t,s,a,r,group)


@pytest.mark.parametrize('bad',['history','terminal','early_winner','late_bank','attempt','cost','bank'])
def test_journal_arrays_and_chronology_tamper(fullmock,bad):
    t,s,a,r,_=fullmock[3][-1]
    group=deepcopy([e['event'] for e in fullmock[2] if e['source_slot']==s and e['arm']==a])
    if bad in ('history','terminal'):
        e=next(e for e in group if e['kind']=='native_packet')
        if bad=='history':e['context']['action_history'][0][0]+=.1
        else:e['context']['latent_history'][-1][0]+=.1
    elif bad=='early_winner':
        i=next(i for i,e in enumerate(group) if e['kind']=='winner_frozen_before_replay');group.insert(0,group.pop(i))
    elif bad=='late_bank':
        i=next(i for i,e in enumerate(group) if e['kind']=='bank_frozen');group.append(group.pop(i))
    elif bad=='attempt':next(e for e in group if e['kind']=='operation_completed')['event']['index']=9
    elif bad=='cost':next(e for e in group if e['kind']=='evaluation_complete')['prediction']['cost']+=.1
    else:next(e for e in group if e['kind']=='bank_frozen')['logits'][0][0][0]+=.1
    with pytest.raises(ValueError):m.journal_check(t,s,a,r,group)


def test_fixed_four_direct_rows_and_plan_generation(fullmock,monkeypatch):
    store=RAMStore();slots=[];tasks={};rows=[]
    results={(s,a):r for _,s,a,r,_ in fullmock[3]}
    for s in (0,1):
        t=fullmock[3][2*s][0];path=f'task_{s}';tasks[path]=t
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=path,task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        for a in m.ARMS:rows.append(dict(source_slot=s,arm=a,status='not_started',resources={}))
    monkeypatch.setattr(m.base,'load_export',lambda path,h:dict(gn_task=tasks[path]))
    def runner(t,s,a,sink,row):
        r=deepcopy(results[s,a]);row['resources']=r['resources'];return r
    m.plan_all(dict(slots=slots),store,runner,rows);store.close()
    assert [(r['source_slot'],r['arm']) for r in rows]==[(s,a) for s in (0,1) for a in m.ARMS]
    for row in rows:
        plan=m.base.decode(store.files[row['plan_file']]);h=plan.pop('selection_sha256')
        assert m.core.digest(plan)==h==row['selection_sha256']
        assert plan['commands']==row['commands'] and not plan['selection_uses_physics']


def test_complete_saved_consumer_in_RAM(fullmock,monkeypatch):
    store=RAMStore();slots=[];tasks={};rows=[];results={(s,a):r for _,s,a,r,_ in fullmock[3]}
    for s in (0,1):
        task=fullmock[3][2*s][0];path=f'task_{s}';tasks[path]=task
        slots.append(dict(source_slot=s,task_id=m.core.task_id(task),task_export=path,task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        for a in m.ARMS:rows.append(dict(source_slot=s,arm=a,blind_sha256=str(s+2)*64,task_id=m.core.task_id(task),status='not_started',resources={}))
    monkeypatch.setattr(m.base,'load_export',lambda path,h:dict(gn_task=tasks[path]))
    def runner(t,s,a,sink,row):
        for e in fullmock[2]:
            if e['source_slot']==s and e['arm']==a:sink(e['event'])
        r=deepcopy(results[s,a]);row['resources']=r['resources'];return r
    store.put('binding.json',dict(prepared='ram_prepare',prepare_sha256='a'*64))
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    m.plan_all(p,store,runner,rows);store.close()
    import hashlib
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    report=dict(schema=m.NAME+'.report',status='sampling_model_plans_sealed_not_admitted',commands_released=True,errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.enc.MODEL_TENSOR_SHA,model_hash_after=m.enc.MODEL_TENSOR_SHA,
        prepared='ram_prepare',prepare_sha256='a'*64,contract=m.contract(),rows=rows,resources={k:dict(attempted=536,completed=536) for k in m.budget()},
        artifact_sha256={n:digest(v) for n,v in store.files.items()})
    store.put('report.json',report,final=True);expected=digest(store.files['report.json'])
    actual_sha=m.sha;actual_read=m.read;actual_iter=Path.iterdir;actual_bytes=Path.read_bytes
    prefix=Path('/RAM_constructive_sampling')
    monkeypatch.setattr(m,'verify',lambda path,h:p)
    monkeypatch.setattr(m,'sha',lambda path:digest(store.files[Path(path).name]) if Path(path).parent==prefix else actual_sha(path))
    def read(path):
        path=Path(path)
        if path.parent!=prefix:return actual_read(path)
        raw=store.files[path.name]
        return m.base.decode(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([prefix/n for n in store.files]) if path==prefix else actual_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==prefix else actual_bytes(path))
    consumed=m.consume(prefix,expected)
    assert len(consumed['rows'])==4 and not consumed['P1_admission']
    assert sum(map(len,store.files.values()))<m.base.CAP
    print('RAM full four-branch journal/results/plans/report bytes:',sum(map(len,store.files.values())))
    # Even with a newly supplied report hash, stale artifact bindings reject tamper.
    name=rows[0]['plan_file'];store.files[name]+=b' '
    with pytest.raises(ValueError,match='artifact hash'):m.consume(prefix,expected)


@pytest.mark.parametrize('slot,arm,expected',[(0,'random_shooting',20261027),(0,'CEM',20261028),(1,'random_shooting',20261128),(1,'CEM',20261129)])
def test_fixed_seeds(slot,arm,expected):assert m.seed(slot,arm)==expected


@pytest.mark.parametrize('slot,arm',[(True,'CEM'),(2,'CEM'),(0,'unknown')])
def test_no_extra_root_or_arm(slot,arm):
    with pytest.raises(ValueError):m.seed(slot,arm)


def test_packetball_float32_tanh_and_no_zero_injection(fullmock):
    for t,_,_,r,_ in fullmock[3]:
        for c in r['candidates']:
            x=np.asarray(c['logits'])
            assert np.array_equal(x,x.astype(np.float32))
            assert np.all(np.linalg.norm(x,axis=1)<=t['packet_logit_cap'])
            assert not np.all(x==0)
            assert c['commands']==np.tanh(x.astype(np.float32)).reshape(10,2).astype(float).tolist()
            assert c['cost']==m.core.model_cost(t,c['terminal'])


def test_elite_tie_order_and_std_clip():
    bank=[dict(index=i,logits=np.zeros((2,10)).tolist(),cost=1.) for i in range(16)]
    mean,std,elite=m.elite_update(bank)
    assert elite==[0,1,2,3] and np.all(mean==0) and np.all(std==.05)
    for i,c in enumerate(bank):c['logits']=(np.ones((2,10))*(100 if i%2 else -100)).tolist()
    _,std,_=m.elite_update(bank);assert np.all(std==1.)


def test_CEM_four_causal_updates_and_global_winner(fullmock):
    for t,s,a,r,_ in fullmock[3]:
        if a!='CEM':assert r['updates']==[];continue
        assert len(r['updates'])==4
        for i,u in enumerate(r['updates']):
            chunk=r['candidates'][16*i:16*i+16];mean,std,elite=m.elite_update(chunk)
            assert u==dict(iteration=i,elite_indices=[chunk[j]['index'] for j in elite],mean=mean.tolist(),std=std.tolist())
        assert r['winner_index']==min(range(64),key=lambda i:(r['candidates'][i]['cost'],i))


@pytest.mark.parametrize('bad',['cost','logits','commands','terminal','index','iteration','seed','slot','arm','winner','updates','budget','replay'])
def test_saved_tamper_rejected(fullmock,bad):
    t,s,a,original,_=fullmock[3][-1];r=deepcopy(original)
    if bad=='cost':r['candidates'][0]['cost']+=.01
    elif bad=='logits':r['candidates'][0]['logits'][0][0]+=.01
    elif bad=='commands':r['candidates'][0]['commands'][0][0]+=.01
    elif bad=='terminal':r['candidates'][0]['terminal'][0]+=.01
    elif bad=='index':r['candidates'][0]['index']=1
    elif bad=='iteration':r['candidates'][0]['iteration']=1
    elif bad=='seed':r['seed']+=1
    elif bad=='slot':r['source_slot']=0
    elif bad=='arm':r['arm']='random_shooting'
    elif bad=='winner':r['winner_index']=(r['winner_index']+1)%64
    elif bad=='updates':r['updates'][0]['std'][0][0]=.99
    elif bad=='budget':r['resources']['predict']['completed']=133
    else:r['final_prediction']['commands'][0][0]+=.01
    with pytest.raises(ValueError):m.choice(t,r,s,a)


def test_model_call_failure_keeps_attempt_and_completed_candidates():
    task=fixture.task();events=[];meter=m.Meter(events.append)
    class Broken(fixture.Backend):
        def rollout(self,t,raw,meter,mode):
            if len(self.calls)==2:
                meter.begin('predict');raise RuntimeError('injected predict failure')
            return super().rollout(t,raw,meter,mode)
    with pytest.raises(RuntimeError,match='injected'):m.optimize(task,0,'CEM',Broken(),meter)
    assert len([e for e in events if e['kind']=='candidate_saved'])==2
    assert meter.counts()['predict']==dict(attempted=5,completed=4)
    assert not any(e['kind']=='winner_frozen_before_replay' for e in events)
    with pytest.raises(ValueError):meter.begin('action_encoder')


def test_replay_mismatch_rejects_release():
    task=fixture.task();meter=m.Meter(lambda e:None)
    class Wrong(fixture.Backend):
        def rollout(self,t,raw,meter,mode):
            p=super().rollout(t,raw,meter,mode)
            if mode=='segmented_replay':p[-1]['latent_history'][-1][0]+=1.
            return p
    with pytest.raises(ValueError,match='triple replay'):m.optimize(task,0,'CEM',Wrong(),meter)


def test_predispatch_caps_and_no_AD():
    meter=m.Meter(lambda e:None)
    for _ in range(134):meter.call('predict',lambda:None)
    with pytest.raises(ValueError):meter.begin('predict')
    for r in ('jacobian','vjp_directions','svd','linear_solve'):
        with pytest.raises(ValueError):m.Meter(lambda e:None).begin(r)


def test_sealed_helpers_not_changed():
    assert m.sha(m.base.__file__)==m.BASE_SHA and m.sha(m.gn.__file__)==m.GN_SHA
    assert m.gn.budget()['predict']==78


def test_only_approved_exports_and_no_privileged_load(monkeypatch):
    def blocked(*a,**k):raise AssertionError('no material reader')
    monkeypatch.setattr(m.gn,'material',blocked)
    # Local explicit export reads only; no model/physics and no source NPZ.
    for path,h in zip(m.EXPORTS,m.EXPORT_SHAS,strict=True):
        t=m.base.load_export(path,h)['gn_task']
        assert t['latent_dim']==192 and t['horizon']==2 and t['packet_steps']==5
        assert t['projection']==np.eye(192).tolist()
