"""RAM arithmetic + actual temporary writer/consumer, never torch/model/physics."""
from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_same_action_actual_history_cuda_20261029_v1 as m
import test_agent_p3_same_action_history_preconditions_20261029_v1 as pre


def materials():
    f=pre.fixture();p,r,rc,pp,pr,pc,arrays,streams=f
    cp=m.resolver.resolve(p,r,rc,pp,pr,pc,lambda d:arrays[d['id']],
        lambda n,s:(streams[n['episode_id']],dict(path='RAM')))
    cp.update(schema=m.compat.NAME+'.prepare',contract=m.resolver.contract(),
        status='all8_saved_preconditions_ready_NOT_execution',ready=8,blocked=0,
        compatibility=dict(unchanged_resolver_sha256=m.compat.PIN,planner_report_cap=256*1024**2,
            other_input_caps_unchanged=True,R_IO_math_resource_checks_unchanged=True,
            original_shared_panel_report_reserve_bytes=256*1024**2),
        P1_admission=False,P3_admission=False,C3=False,physical_release=False)
    for slot,task in enumerate(pp['tasks']):
        task['root_context']=deepcopy(cp['jobs'][4*slot]['R_context'])
        task['projection']=np.eye(192).tolist()
        task['binding']={k:'a'*64 for k in m.core.BINDINGS}
        task['binding'].update(goal_sha256=m.digest(task['goal']),projection_sha256=m.digest(task['projection']),
            context_sha256=m.digest(task['root_context']),normalizer_sha256=m.digest([task['action_mean'],task['action_std']]),
            action_domain_sha256=m.digest(m.core.action_domain()))
        for j in cp['jobs'][4*slot:4*slot+4]:j['task_binding']=deepcopy(task['binding'])
    return cp,pp


class Module:
    def __init__(self,fn):self.forward=fn
    def __call__(self,*args):return self.forward(*args)


class FakeModel:
    """Arithmetic callbacks only, exercises real instance observe/call wrappers."""
    def __init__(self):
        self.action_encoder=Module(lambda x:np.tile(x.mean(axis=-1,keepdims=True),(1,1,192)).astype(np.float32))
        self.predictor=Module(lambda s,a:(s+np.float32(.125)*a).astype(np.float32))
        self.pred_proj=Module(lambda x:x.copy())
        self.encoder=Module(lambda x:x)
        self.projector=Module(lambda x:x)
    def predict(self,s,a):return self.pred_proj(self.predictor(s,a).reshape(3,192)).reshape(1,3,192)


class CallbackBackend(m.signal.NativeSignal):
    def __init__(self,model,job,store):
        self.model=model;self.task=job['task'];self.store=store;self.ledger=None;self.phase='whole';self.step=0
    def frozen(self):return None
    def arrays(self,kind,operation_index=None,**values):
        self.store.event(dict(kind=kind,scope=deepcopy(self.ledger.scope),phase=self.phase,step=0,
            operation_index=operation_index,arrays={k:m.signal.blob(v) for k,v in values.items()}))
    def replay(self,parent,actions,segmented,ledger):
        assert segmented is False;self.ledger=ledger
        s,h=m.resolver.context(parent);raw=np.asarray(actions,np.float32)
        mean=np.asarray(self.task['action_mean'],np.float32);std=np.asarray(self.task['action_std'],np.float32)
        token=((raw-mean)/std).reshape(10);outgoing=np.concatenate((h,token[None]),axis=0)[None]
        self.arrays('program_packet',latent=s[None],history=h[None],raw=raw,mean=mean,std=std,token=token,outgoing=outgoing)
        with self.observe():
            encoded=self.model.action_encoder(outgoing)
            full=self.call('predict',lambda:self.model.predict(s[None],encoded),
                dict(latent=s[None],act_emb=encoded),shape=[1,3,192])
        z=np.concatenate((s[1:],full[0,-1:]),axis=0);hist=outgoing[0,1:]
        self.arrays('program_boundary',latent=z[None],history=hist[None])
        return [dict(parent,latent_history=z.tolist(),action_history=hist.tolist())]


def execute_fixture(tmp_path):
    cp,pp=materials();p=dict(jobs=m.material(cp,pp));store=m.Store(tmp_path/(m.NAME+'_fixture'),time.monotonic())
    ledger=m.core.Ledger(m.caps(),sink=store.event);rows=[dict(index=i,status='not_started',after=None,error=None) for i in range(8)]
    models=[]
    def factory(job):
        model=FakeModel();models.append((model,model.action_encoder.forward));return CallbackBackend(model,job,store)
    m.execute(p,store,ledger,factory,rows,store.check)
    assert all(model.action_encoder.forward is fn for model,fn in models)
    seal=dict(rows=deepcopy(rows),resource_receipt=ledger.receipt(),forbidden=[])
    h=store.put('prediction_seal.json',seal);store.event(dict(kind='predictions_sealed',sha256=h))
    store.event(dict(kind='labels_unlocked',preconditions_sha256=m.PRECONDITIONS_SHA))
    store.event(dict(kind='run_finished',status='all8_A_recorded_NOT_admitted'))
    events=[m.saved.decode(l)['payload'] for l in (store.path/'events.jsonl').read_bytes().splitlines()]
    r=dict(rows=deepcopy(rows),resource_receipt=ledger.receipt(),forbidden=[],prediction_seal_sha256=h)
    return p,r,events,seal,store


def test_all8_actual_real_wrappers_and_complete_lifecycle(tmp_path):
    p,r,events,seal,store=execute_fixture(tmp_path)
    m.validate_events(p,r,events,seal)
    assert r['resource_receipt']['completed']==m.caps()
    assert len(events)==163
    assert all('label192' not in j and 'R_proof' not in j for j in p['jobs'])
    assert store.used<64*1024**2


@pytest.mark.parametrize('change',['unknown','splice','A_latent','action_token','output','shape','completed','labels_first','ID','return'])
def test_saved_lifecycle_mutations_rejected(tmp_path,change):
    p,r,es,seal,_=execute_fixture(tmp_path)
    native=[i for i,e in enumerate(es) if 'arrays' in e]
    if change=='unknown':es.insert(0,dict(kind='unknown'))
    elif change=='splice':es[native[11]],es[native[12]]=es[native[12]],es[native[11]]
    elif change=='A_latent':es[native[3]]['arrays']['predict:latent']=pre.blob(np.zeros((1,3,192),np.float32))
    elif change=='action_token':es[native[0]]['arrays']['token']=pre.blob(np.zeros(10,np.float32))
    elif change=='output':es[native[8]]['arrays']['predict:output']=pre.blob(np.zeros((1,3,192),np.float32))
    elif change=='shape':es[native[8]]['arrays']['predict:output']=pre.blob(np.zeros((3,192),np.float32))
    elif change=='completed':seal['resource_receipt']['events'][0]['status']='failed'
    elif change=='labels_first':es[-2],es[-3]=es[-3],es[-2]
    elif change=='ID':es[native[8]]['operation_index']=200
    elif change=='return':r['rows'][0]['after']['latent_history'][-1][0]+=1
    with pytest.raises((ValueError,KeyError,IndexError)):m.validate_events(p,r,es,seal)


@pytest.mark.parametrize('change',['schema','compat','Rverified','Rbudget','index','goal','mean','raw','bool','context','actrecipe','subset'])
def test_material_failclosed(change):
    cp,pp=materials()
    if change=='schema':cp['schema']=m.resolver.NAME+'.prepare'
    elif change=='compat':cp['compatibility']['planner_report_cap']=128*1024**2
    elif change=='Rverified':cp['jobs'][0]['R_reuse_verified']=False
    elif change=='Rbudget':cp['jobs'][0]['new_R_calls']=1
    elif change=='index':cp['jobs'][0]['index']=True
    elif change=='goal':cp['jobs'][0]['goal192'][0]=10.
    elif change=='mean':cp['jobs'][0]['normalizer']['mean']=[1.,1.]
    elif change=='raw':cp['jobs'][0]['future_raw5'][0][0]=2.
    elif change=='bool':cp['jobs'][0]['A_context']['latent_history'][0][0]=True
    elif change=='context':cp['jobs'][0]['A_context_sha256']='0'*64
    elif change=='actrecipe':cp['jobs'][0]['A_predict_input']['outgoing_actions'][0][0][0]+=1
    elif change=='subset':cp['jobs'].pop()
    with pytest.raises(ValueError):m.material(cp,pp)


def test_store_timeout_finalization_and_chain(tmp_path):
    now=[0.];store=m.Store(tmp_path/(m.NAME+'_timeout'),0.,clock=lambda:now[0])
    store.event(dict(kind='a'));store.put('binding.json',dict(x=1))
    now[0]=601.
    with pytest.raises(ValueError):store.event(dict(kind='not_allowed'))
    store.finalizing=True;store.event(dict(kind='run_finished',status='partial_or_failed'))
    h=m.file_sha(store.path/'events.jsonl')
    es,ref=m.resolver.read_selected_journal(store.path/'events.jsonl',h,store.tail,None)
    assert ref['records']==2 and len(es)==2
    assert b'\n\n' not in (store.path/'events.jsonl').read_bytes()
    store.put('report.json',dict(status='partial_or_failed'),final=True)
    store.census()
    with pytest.raises(ValueError):store.put('binding.json',{})


def fake_torch():
    def fn(*a,**k):return 'original'
    ns=lambda names:SimpleNamespace(**{n:fn for n in names})
    return SimpleNamespace(autograd=SimpleNamespace(grad=fn,backward=fn,functional=ns(['jacobian','hessian'])),
        Tensor=type('Tensor',(),dict(backward=fn)),func=ns(['jacrev','jacfwd','grad','hessian']),linalg=ns(['svd','solve']))


def test_noAD_forbid_and_restore():
    t=fake_torch();original=t.autograd.grad;events=[]
    with pytest.raises(ValueError):
        with m.NoAD(t,events.append):t.autograd.grad(None)
    assert t.autograd.grad is original and len(events)==1


def test_noAD_enter_failure_restores_partial():
    t=fake_torch();original=t.autograd.grad;del t.func.jacrev
    with pytest.raises(AttributeError):
        with m.NoAD(t,lambda e:None):assert False
    assert t.autograd.grad is original


def test_visual_api_forbidden_and_wrappers_restored(tmp_path):
    cp,pp=materials();job=m.material(cp,pp)[0];store=m.Store(tmp_path/(m.NAME+'_visual'),time.monotonic())
    model=FakeModel();fn=model.encoder.forward;b=CallbackBackend(model,job,store)
    b.ledger=m.core.Ledger(m.caps(),sink=store.event);b.ledger.scope=dict(job=0,phase='A')
    with pytest.raises((ValueError,m.core.BudgetExceeded)):
        with b.observe():model.encoder(np.zeros((1,3,192),np.float32))
    assert model.encoder.forward is fn


def bind_fixture(tmp_path,monkeypatch):
    args=pre.save_inputs(tmp_path)
    cp=m.compat.build(**args)
    validated_cp,validated_pp=materials()
    cp['jobs']=validated_cp['jobs']
    # Complete fixture model identity source closure; fake bytes are never loaded.
    pp=m.read(args['planner_prepared'],args['planner_prepared_sha'])
    pp['tasks']=validated_pp['tasks']
    policy=tmp_path/'policy';policy.mkdir();(policy/'weights.pt').write_bytes(b'fixture no model')
    (policy/'config.json').write_bytes(b'{}')
    pp.update(policy=str(policy),sources={str(f):m.file_sha(f) for f in policy.iterdir()})
    file=Path(args['planner_prepared']);file.write_bytes(m.canonical(pp));args['planner_prepared_sha']=m.file_sha(file)
    pr=m.saved.decode(Path(args['planner_report']).read_bytes());pr['prepare_sha256']=args['planner_prepared_sha']
    Path(args['planner_report']).write_bytes(m.canonical(pr));args['planner_report_sha']=m.file_sha(Path(args['planner_report']))
    pc=m.saved.decode(Path(args['planner_receipt']).read_bytes());pc['prepare_sha256']=args['planner_prepared_sha']
    pc['report_sha256']=args['planner_report_sha'];Path(args['planner_receipt']).write_bytes(m.canonical(pc))
    args['planner_receipt_sha']=m.file_sha(Path(args['planner_receipt']))
    ep=m.saved.decode(Path(args['encoder_prepared']).read_bytes())
    ep['bindings'].update({args[k]:args[k+'_sha'] for k in ('planner_prepared','planner_report','planner_receipt')})
    Path(args['encoder_prepared']).write_bytes(m.canonical(ep));args['encoder_prepared_sha']=m.file_sha(Path(args['encoder_prepared']))
    er=m.saved.decode(Path(args['encoder_report']).read_bytes());er['prepare_sha256']=args['encoder_prepared_sha']
    Path(args['encoder_report']).write_bytes(m.canonical(er));args['encoder_report_sha']=m.file_sha(Path(args['encoder_report']))
    ec=m.saved.decode(Path(args['encoder_receipt']).read_bytes());ec['report_sha256']=args['encoder_report_sha']
    Path(args['encoder_receipt']).write_bytes(m.canonical(ec));args['encoder_receipt_sha']=m.file_sha(Path(args['encoder_receipt']))
    cp['arguments']=deepcopy(args)
    cp['bindings'].update({args[k]:args[k+'_sha'] for k in m.resolver.KEYS})
    path=tmp_path/'compat.json';path.write_bytes(m.canonical(cp))
    h=m.file_sha(path);monkeypatch.setattr(m,'PRECONDITIONS_SHA',h)
    return path,h


def test_full_saved_prepare_verify_and_bound_tamper(tmp_path,monkeypatch):
    path,h=bind_fixture(tmp_path,monkeypatch);p=m.build(str(path),h)
    dest=tmp_path/(m.NAME+'_prepare_fixture.json');d=m.write_metadata(dest,p)
    assert m.verify(dest,d['sha256'])==p
    assert p['inherited_authority']['full_upstream_strict_rebuilt_here'] is False
    (tmp_path/'policy/weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify(dest,d['sha256'])


def test_wrong_preconditions_pin_before_read(tmp_path):
    with pytest.raises(ValueError):m.build(str(tmp_path/'absent'),'0'*64)


def test_local_binding_authenticates_actual_context_not_original_root():
    cp,pp=materials();jobs=m.material(cp,pp)
    for job in jobs:
        m.core.validate_task(job['task'])
        assert job['task']['binding']['context_sha256']==m.digest(job['task']['root_context'])
        assert job['original_task_binding']==pp['tasks'][job['root_slot']]['binding']
        assert job['task']['binding']['task_contract_sha256']!=job['original_task_binding']['task_contract_sha256']


def test_unauthorized_run_before_output(tmp_path):
    with pytest.raises(ValueError):m.run('absent','0'*64,tmp_path/(m.NAME+'_bad'),None)
    assert list(tmp_path.iterdir())==[]


def test_partial_failure_preserves_all8_no_refund(tmp_path):
    cp,pp=materials();p=dict(jobs=m.material(cp,pp));store=m.Store(tmp_path/(m.NAME+'_partial'),time.monotonic())
    rows=[dict(index=i,status='not_started',after=None,error=None) for i in range(8)]
    ledger=m.core.Ledger(m.caps(),sink=store.event)
    def factory(job):
        if job['index']==1:raise ValueError('fixture failure')
        return CallbackBackend(FakeModel(),job,store)
    with pytest.raises(ValueError):m.execute(p,store,ledger,factory,rows,store.check)
    assert len(rows)==8 and rows[0]['status']=='returned' and rows[1]['status']=='failed'
    assert ledger.attempted['predict_API']==ledger.completed['predict_API']==1
    assert all(r['status']=='not_started' for r in rows[2:])


def full_saved_bundle(tmp_path,monkeypatch):
    path,h=bind_fixture(tmp_path,monkeypatch);p=m.build(str(path),h)
    prepared=tmp_path/(m.NAME+'_prepare.json');binding=m.write_metadata(prepared,p)
    store=m.Store(tmp_path/(m.NAME+'_saved'),time.monotonic())
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=binding['sha256']))
    store.put('initial_jobs.json',p['jobs'])
    ledger=m.core.Ledger(m.caps(),sink=store.event)
    rows=[dict(index=i,status='not_started',after=None,error=None) for i in range(8)]
    m.execute(p,store,ledger,lambda j:CallbackBackend(FakeModel(),j,store),rows,store.check)
    state=dict(original_state_dict=p['expected_model_hash'],all_buffers='b'*64)
    source=str(Path(__file__).resolve());inv=[dict(type='RAM.callback.fixture',path=source,sha256=p['bindings'][source])]
    seal=dict(rows=rows,resource_receipt=ledger.receipt(),model_before=state,model_after=state,
        inventory_before=inv,inventory_after=inv,precision_before=m.precision_contract(),
        precision_after=m.precision_contract(),forbidden=[])
    h=store.put('prediction_seal.json',seal);store.event(dict(kind='predictions_sealed',sha256=h))
    store.event(dict(kind='labels_unlocked',preconditions_sha256=m.PRECONDITIONS_SHA))
    scores=m.score(p,rows);sh=store.put('scores.json',scores)
    store.event(dict(kind='run_finished',status='all8_A_recorded_NOT_admitted'))
    r=dict(schema=m.NAME+'.report',status='all8_A_recorded_NOT_admitted',contract=m.contract(),
        prepared=str(prepared),prepare_sha256=binding['sha256'],rows=rows,errors=[],
        resource_receipt=ledger.receipt(),forbidden=[],model_before=state,model_after=state,
        inventory_before=inv,inventory_after=inv,precision_before=m.precision_contract(),precision_after=m.precision_contract(),
        CUDA_peak_bytes=[0,0],elapsed_seconds=.1,prediction_seal_sha256=h,scores_sha256=sh,
        event_count=store.index,journal_tail=store.tail,artifact_sha256=store.census(),
        P1_admission=False,P3_admission=False,C3=False,physical_release=False)
    store.put('report.json',r,final=True)
    return store.path,m.file_sha(store.path/'report.json')


def test_full_own_saved_consumer_and_pair_arithmetic(tmp_path,monkeypatch):
    output,h=full_saved_bundle(tmp_path,monkeypatch);receipt=m.consume(output,h)
    assert receipt['fixed_denominator']==len(receipt['rows'])==8 and receipt['R_new_calls']==0
    assert receipt['full_upstream_strict_rebuilt_here'] is False


@pytest.mark.parametrize('change',['score','unknown','model','precision','labels','seal','counts'])
def test_coherent_saved_report_mutations(tmp_path,monkeypatch,change):
    output,h=full_saved_bundle(tmp_path,monkeypatch);rp=output/'report.json';r=m.read(rp,h)
    if change=='score':
        path=output/'scores.json';v=m.saved.decode(path.read_bytes());v[0]['A_minus_R_MSE']+=1
        path.write_bytes(m.canonical(v));r['scores_sha256']=r['artifact_sha256']['scores.json']=m.file_sha(path)
    elif change=='unknown':(output/'unknown.json').write_bytes(b'{}')
    elif change=='model':r['model_after']['original_state_dict']='0'*64
    elif change=='precision':r['precision_after']['matmul_TF32']=True
    elif change=='seal':r['prediction_seal_sha256']='0'*64
    elif change=='counts':r['resource_receipt']['completed']['predict_API']=7
    elif change=='labels':
        path=output/'events.jsonl';es=[m.saved.decode(line)['payload'] for line in path.read_bytes().splitlines()]
        es[-2],es[-3]=es[-3],es[-2];tail='0'*64;lines=[]
        for i,e in enumerate(es):
            b=dict(index=i,previous=tail,payload=e);tail=m.digest(b);lines.append(m.canonical(dict(b,sha256=tail))+b'\n')
        path.write_bytes(b''.join(lines));r['journal_tail']=tail;r['artifact_sha256']['events.jsonl']=m.file_sha(path)
    rp.write_bytes(m.canonical(r));h=m.file_sha(rp)
    with pytest.raises((ValueError,KeyError)):m.consume(output,h)
