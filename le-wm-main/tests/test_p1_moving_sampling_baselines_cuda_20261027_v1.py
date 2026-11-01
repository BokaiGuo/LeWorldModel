"""Pure NumPy, RAM gzip transport; no actual inputs/model/physics/tmpdir."""
from copy import deepcopy
from contextlib import nullcontext
import gzip
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_sampling_baselines_cuda_20261027_v1 as m


def exported(slot):
    moving=m.moving
    packet=dict(rgb_history=np.stack([np.full((224,224,3),i+slot,np.uint8) for i in range(3)]),
        raw_action_history=moving.fixed_raw(),goal_rgb=np.full((224,224,3),9+slot,np.uint8))
    s=dict(source_slot=slot,blind_sha256=str(slot+2)*64,
        pixel_sha256=[moving.digest(x.tobytes()) for x in [*packet['rgb_history'],packet['goal_rgb']]],
        raw_action_sha256=moving.digest(packet['raw_action_history'].tobytes()))
    p=dict(normalization=moving.normalization(),model_bindings=moving.MODELS,
        arguments=dict(manifest_sha='a'*64))
    z=np.random.default_rng(17+slot).normal(size=(4,192)).astype(np.float32)
    return moving.task_export(p,'b'*64,s,packet,list(z))


def task(slot):return m.interface(m.moving.task_interfaces(exported(slot)))['gn_task']


class Backend:
    """Analytic array stand-in depends on both incoming history tokens."""
    def __init__(self,seed=11,fail_after=None):
        self.a=np.random.default_rng(seed).normal(scale=.1,size=(192,10))
        self.calls=[];self.inputs=[];self.fail_after=fail_after;self.history_scale=.0001
    def rollout(self,t,raw,meter,mode):
        self.calls.append((meter.group,mode));before=deepcopy(t['root_context']);path=[]
        tokens=m.core.normalized_tokens(t,raw)
        for k in (0,1):
            self.inputs.append(deepcopy(before))
            for resource in ('action_encoder','predict'):
                token=meter.begin(resource)
                if resource=='predict' and self.fail_after==len(self.inputs):raise RuntimeError('injected array forward failure')
                meter.end(token)
            h=np.asarray(before['action_history']);z=np.asarray(before['latent_history'])
            last=(z[-1]+self.a@raw.reshape(2,10)[k]+self.history_scale*(h[0].sum()+2*h[1].sum())).astype(np.float32)
            before=dict(before,latent_history=before['latent_history'][1:]+[last.astype(float).tolist()],
                action_history=[before['action_history'][-1],tokens[k].astype(float).tolist()])
            path.append(before);meter.emit('native_packet',mode=mode,packet=k,context=before)
        return path


class RAMStore(m.Store):
    def __init__(self,output='/RAM_moving_sampling'):
        self.output=Path(output);self.files={};self.events=[]
        self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+raw
    def event(self,e):self.events.append(deepcopy(e));super().event(e)


@pytest.fixture(scope='module')
def fullmock():
    store=RAMStore();tasks={s:task(s) for s in (0,1)};slots=[];rows=[];results={};backends={}
    for s,t in tasks.items():
        slots.append(dict(source_slot=s,task_id=m.core.task_id(t),task_export=f'task_{s}',task_sha256=str(s)*64,blind_sha256=str(s+2)*64))
        for arm in m.ARMS:rows.append(dict(source_slot=s,arm=arm,blind_sha256=str(s+2)*64,task_id=m.core.task_id(t),status='not_started',resources={}))
    def runner(t,s,arm,sink,row):
        b=Backend(11+s);backends[s,arm]=b
        meter=(m.cem.Meter if arm=='CEM' else m.nominal.Meter)(sink)
        try:
            r=m.optimize(t,s,arm,b,meter);results[s,arm]=r;return r
        finally:row['resources']=meter.counts()
    p=dict(slots=slots,contract=m.contract(),bindings_sha256={})
    original=m.load_export
    try:
        m.load_export=lambda path,h:dict(gn_task=tasks[int(path[-1])])
        store.put('binding.json',dict(prepared='ram_prepare',prepare_sha256='a'*64))
        m.plan_all(p,store,runner,rows)
    finally:m.load_export=original
    store.close();return store,p,tasks,rows,results,backends


def test_complete_budgets_contexts_and_frozen_schedules(fullmock):
    store,p,tasks,rows,results,backends=fullmock
    assert m.contract()['total_budget']==dict(predict=412,action_encoder=412)
    for row in rows:
        s,a=row['source_slot'],row['arm'];r=results[s,a];b=backends[s,a];t=tasks[s]
        n=134 if a=='CEM' else 72
        assert row['resources']=={k:dict(attempted=n,completed=n) for k in m.budget(a)}
        assert len(b.calls)==(67 if a=='CEM' else 36)
        assert len(b.inputs)==n
        assert b.inputs[0]==t['root_context']
        assert b.inputs[1]['action_history'][0]==t['root_context']['action_history'][-1]
        assert not np.array_equal(t['root_context']['action_history'][0],t['root_context']['action_history'][1])
        assert m.choice(t,r,s,a)['commands']==r['final_prediction']['commands']
        if a=='CEM':assert len(r['candidates'])==64 and len(r['updates'])==4
        else:assert len(r['evaluations'])==35 and r['selected_index']==34 and len(r['updates'])==2
    assert sum(map(len,store.files.values()))<m.CAP


def test_own_saved_journal_closure_both_arms(fullmock):
    store,p,tasks,rows,results,_=fullmock
    plain,eof=m.base.inflate(store.files['events.jsonl.gz']);assert eof
    assert [m.base.decode(x) for x in plain.splitlines()]==store.events
    for row in rows:
        s,a=row['source_slot'],row['arm']
        events=[e['event'] for e in store.events if (e['source_slot'],e['arm'])==(s,a)]
        m.journal_check(tasks[s],p['slots'][s],a,results[s,a],events)
    assert sum(e['event']['kind']=='native_packet' for e in store.events)==412


def test_original_seeds_and_single_arm_contracts():
    assert [m.seed(s,'CEM') for s in (0,1)]==[20261028,20261129]
    assert [m.seed(s,m.ARMS[1]) for s in (0,1)]==[20261029,20261130]
    assert m.contract(('CEM',))['total_budget']==dict(predict=268,action_encoder=268)
    assert m.contract((m.ARMS[1],))['total_budget']==dict(predict=144,action_encoder=144)
    for c in [m.contract(),m.contract(('CEM',)),m.contract((m.ARMS[1],))]:
        assert c['inherited_encoding_once']==dict(encoder=dict(attempted=8,completed=8),projector=dict(attempted=8,completed=8))
        assert c['AD']==c['new_encoder']==c['new_projector']==c['new_physics']==0
        assert not c['P1_admission'] and not c['official_MPPI'] and not c['equal_total_compute_claim']


@pytest.mark.parametrize('arms',[[],['CEM','CEM'],['random_shooting'],list(reversed(m.ARMS))])
def test_invalid_arm_sets(arms):
    with pytest.raises(ValueError):m.contract(arms)


@pytest.mark.parametrize('change',['zero','divide100','swaphistory','Q','cap','oldschema'])
def test_zero_or_changed_moving_task_rejected(change):
    e=exported(0)
    if change=='oldschema':
        e['schema']=m.enc.NAME+'.task_export'
        with pytest.raises(ValueError):m.moving.task_interfaces(e)
        return
    v=m.moving.task_interfaces(e);t=v['gn_task']
    if change=='zero':t['root_raw_action_history']=np.zeros((10,2)).tolist()
    elif change=='divide100':t['root_context']['action_history']=(m.core.normalized_tokens(t,m.moving.fixed_raw()/100)).astype(float).tolist()
    elif change=='swaphistory':t['root_context']['action_history'].reverse()
    elif change=='Q':t['projection'][0][0]=2
    elif change=='cap':t['packet_logit_cap']*=2
    with pytest.raises(ValueError):m.interface(v)


def test_final_nominal_not_best_sample():
    t=task(0);t['goal']=deepcopy(t['root_context']['latent_history'][-1])
    # Zero incoming-history effect in this fixture makes initial zero exactly best.
    b=Backend();b.history_scale=0
    for token in t['root_context']['action_history']:
        token[:]=[0.]*10
    t['root_raw_action_history']=np.tile(np.asarray(t['action_mean'],np.float32),(10,1)).astype(float).tolist()
    # This synthetic task exercises the mathematical kernel only, not moving loader.
    t['root_context']['action_history']=m.core.normalized_tokens(t,t['root_raw_action_history']).astype(float).tolist()
    r=m.nominal.optimize(t,0,b,m.nominal.Meter(lambda e:None))
    assert r['selected_index']==34
    assert r['final_prediction']['commands']==r['evaluations'][34]['commands']
    assert r['selection']=='last_updated_nominal_not_best'
    assert r['diagnostic_best_index']==0 and r['evaluations'][0]['cost']==0
    assert r['final_prediction']['cost']>0


@pytest.mark.parametrize('arm',m.ARMS)
def test_forward_failure_attempts_and_durable_partial(arm):
    events=[];meter=(m.cem.Meter if arm=='CEM' else m.nominal.Meter)(events.append)
    with pytest.raises(Exception):m.optimize(task(0),0,arm,Backend(fail_after=2),meter)
    assert meter.counts()['predict']==dict(attempted=2,completed=1)
    assert meter.counts()['action_encoder']==dict(attempted=2,completed=2)
    assert sum(e['kind']=='native_packet' for e in events)==1
    assert not any(e['kind'] in ('model_selection_sealed','winner_frozen_before_replay','nominal_dispatch_frozen') for e in events)
    with pytest.raises(ValueError):meter.begin('predict')


@pytest.mark.parametrize('arm',m.ARMS)
def test_synchronization_failure_never_completed(arm):
    def fail():raise RuntimeError('sync failure')
    meter=(m.cem.Meter if arm=='CEM' else m.nominal.Meter)(lambda e:None,fail)
    token=meter.begin('action_encoder')
    with pytest.raises(RuntimeError):meter.end(token)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=0)


@pytest.mark.parametrize('field',['commands','cost','history','seed','resources'])
def test_saved_result_tamper_rejected(fullmock,field):
    for a in m.ARMS:
        r=deepcopy(fullmock[4][0,a]);t=fullmock[2][0]
        if field=='commands':r['final_prediction']['commands'][0][0]+=.01
        elif field=='cost':r['final_prediction']['cost']+=.01
        elif field=='history':r['final_prediction']['contexts'][0]['action_history'][0][0]+=.01
        elif field=='seed':r['seed']+=1
        else:r['resources']['predict']['completed']-=1
        with pytest.raises(ValueError):m.choice(t,r,0,a)


def ram_consumer(monkeypatch,fullmock,arms=None):
    store,p,tasks,rows,results,_=fullmock;arms=m.checked_arms(arms)
    files=deepcopy(store.files);events=[e for e in store.events if e['arm'] in arms]
    if arms!=m.ARMS:
        files={n:v for n,v in files.items() if not n.startswith('root_') or any('_'+a+'.' in n for a in arms)}
        files['events.jsonl.gz']=gzip.compress(b''.join(m.canonical(e) for e in events),mtime=0)
    p=deepcopy(p);p['contract']=m.contract(arms);rows=[deepcopy(r) for r in rows if r['arm'] in arms]
    digest=lambda v:hashlib.sha256(v).hexdigest()
    r=dict(schema=m.NAME+'.report',status='moving_sampling_plans_sealed_not_admitted',commands_released=True,
        errors=[],first_error=None,source_verified=True,model_unchanged=True,
        model_hash_before=m.moving.low.MODEL_TENSOR_SHA,model_hash_after=m.moving.low.MODEL_TENSOR_SHA,
        prepared='ram_prepare',prepare_sha256='a'*64,contract=p['contract'],rows=rows,
        resources={k:dict(attempted=n,completed=n) for k,n in p['contract']['total_budget'].items()},
        inherited_encoding_once=p['contract']['inherited_encoding_once'],artifact_sha256={n:digest(v) for n,v in files.items()})
    files['report.json']=m.canonical(r);expected=digest(files['report.json']);prefix=Path('/RAM_moving_sampling')
    real_sha,real_read,real_iter,real_bytes=m.sha,m.read,Path.iterdir,Path.read_bytes
    monkeypatch.setattr(m,'verify',lambda path,h:p)
    monkeypatch.setattr(m,'load_export',lambda path,h:dict(gn_task=tasks[int(path[-1])]))
    monkeypatch.setattr(m,'sha',lambda path:digest(files[Path(path).name]) if Path(path).parent==prefix else real_sha(path))
    def read(path):
        path=Path(path)
        if path.parent!=prefix:return real_read(path)
        raw=files[path.name];return m.base.decode(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([prefix/n for n in files]) if path==prefix else real_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:files[path.name] if path.parent==prefix else real_bytes(path))
    return files,prefix,expected


@pytest.mark.parametrize('arms',[None,('CEM',),(m.ARMS[1],)])
def test_own_consume_full_and_single_arm_RAM_only(monkeypatch,fullmock,arms):
    files,path,h=ram_consumer(monkeypatch,fullmock,arms)
    result=m.consume(path,h);assert len(result['rows'])==2*len(m.checked_arms(arms))
    assert not result['P1_admission'] and not result['official_MPPI']
    assert sum(map(len,files.values()))<m.CAP
    name=result['rows'][0]['arm'];files[f'root_0_{name}.plan.json']+=b' '
    with pytest.raises(ValueError,match='artifact hash'):m.consume(path,h)


def test_startup_failure_durable_report_no_model_import(monkeypatch):
    store=RAMStore();p=dict(slots=[dict(source_slot=s,task_id='a'*64,blind_sha256='b'*64) for s in (0,1)],contract=m.contract())
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'Store',lambda output:store)
    fake_torch=NS()
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setitem(sys.modules,'torch.nn.attention',NS(SDPBackend=NS(MATH=1),sdpa_kernel=lambda *a:nullcontext()))
    monkeypatch.setitem(sys.modules,'experiments.checkpoint_compat',NS(load_compatible_pretrained=lambda *a,**kw:pytest.fail('model loading forbidden')))
    def fail(*a):raise RuntimeError('mock allocation denied')
    monkeypatch.setitem(sys.modules,'experiments.gpu_preflight',NS(require_cuda_allocation=fail))
    monkeypatch.setattr(Path,'iterdir',lambda self:iter([]))
    with pytest.raises(ValueError,match='partial preserved'):m.run('ram_prepare','a'*64,'/RAM_moving_sampling')
    r=m.base.decode(store.files['report.json'])
    assert r['status']=='partial_or_failed' and not r['commands_released']
    assert r['first_error']['message']=='mock allocation denied'
    assert all(row['status']=='not_started' for row in r['rows'])


def test_build_RAM_bindings_never_calls_zero_or_old_build(monkeypatch):
    es=[exported(s) for s in (0,1)];artifacts={};rows=[]
    for s,e in enumerate(es):
        name=f'task_{s:02d}.json.gz';artifacts[name]=str(s)*64;images=[]
        for j in range(4):
            n=f'image_{4*s+j:02d}.native192.npy';artifacts[n]='c'*64
            images.append(dict(path=str(m.REPORT.parent/n),sha256='c'*64,dtype='float32',shape=[192]))
        rows.append(dict(source_slot=s,order=s,status='task_sealed',images=images,
            task_path=str(m.REPORT.parent/name),task_file_sha256=artifacts[name],task_id=e['task_id'],blind_source_sha256=e['blind_sha256']))
    artifacts.update({'binding.json':'d'*64,'image.events.jsonl':'e'*64})
    r=dict(schema=m.moving.NAME+'.report',status='moving_blind_tasks_encoded_and_sealed',errors=[],first_error=None,
        source_verified=True,model_unchanged=True,model_hash_before=m.moving.low.MODEL_TENSOR_SHA,
        model_hash_after=m.moving.low.MODEL_TENSOR_SHA,with_target=False,new_predict=0,new_actionencoder=0,new_AD=0,new_physics=0,
        contract=m.moving.contract(),rows=rows,resources=m.contract()['inherited_encoding_once'],artifact_sha256=artifacts,
        prepare_sha256='b'*64,arguments=dict(manifest_sha='a'*64))
    values={str(m.REPORT):r,**{row['task_path']:e for row,e in zip(rows,es)}}
    hashes={**m.PRIMITIVES,**m.moving.MODELS,str(m.REPORT):m.REPORT_SHA,
        **{str(m.REPORT.parent/n):h for n,h in artifacts.items()}}
    monkeypatch.setattr(m,'read',lambda p:deepcopy(values[str(p)]))
    monkeypatch.setattr(m,'sha',lambda p:hashes.get(str(p),'f'*64))
    monkeypatch.setattr(m,'load_export',lambda p,h:m.interface(m.moving.task_interfaces(values[str(p)])))
    monkeypatch.setattr(m.base,'source_files',lambda:[]);monkeypatch.setattr(m.moving,'source_files',lambda:[])
    def forbidden(*a,**kw):pytest.fail('old/zero/build/sourcepack access forbidden')
    for module in (m.cem,m.nominal,m.base,m.moving):monkeypatch.setattr(module,'build',forbidden)
    p=m.build(('CEM',))
    assert p['schema']==m.NAME+'.prepare' and p['contract']['arms']==['CEM']
    assert p['contract']['total_budget']==dict(predict=268,action_encoder=268)
    assert len(m.canonical(p))<256*1024
    assert [s['task_id'] for s in p['slots']]==[e['task_id'] for e in es]


def test_z_no_actual_neural_or_physics_imports():
    assert not any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel'))
