"""RAM-only native runner tests: no neural, native physics or production I/O."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_wallcapped_physical_20261029_v1 as m
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, MemoryStore as OldMemoryStore, Env, fake_place, h, reseal
from test_agent_allmode_prefix_20261028_v1 import slot as model_slot


def fixture(missing=()):
    _,data,*_=old_fixture()
    compiled=dict(schema=m.PRODUCER+'.prepare',expected_model_hash=h('model'),slots=[],
        root_arm_order=[[r,a] for r,_,a in m.ORDER],bindings_sha256={})
    for root in (0,1):
        s=model_slot(root*2)
        s.update(root_slot=root,condition='off',task_id=h(('task',root)),task_sha256=h(('taskfile',root)),
            warm_raw10=m.moving.warmup_commands().tolist())
        compiled['slots'].append(s)
    plans=[]
    for i,(root,_,arm) in enumerate(m.ORDER):
        s=compiled['slots'][root]
        commands=np.full((15,2),.125*(i%2),np.float32).tolist()
        cp=dict(schema=m.PRODUCER+'.plan',root_slot=root,arm=arm,task_id=s['task_id'],
            task_sha256=s['task_sha256'],root_context=deepcopy(s['packet']['root_context']),
            goal=deepcopy(s['packet']['goal']),warm_raw10=s['warm_raw10'],commands=commands,
            raw_float32_bits=np.asarray(commands,np.float32).tobytes().hex(),
            contexts=[deepcopy(s['packet']['root_context']) for _ in range(3)],
            model_hash=compiled['expected_model_hash'],model_Q=.25,report_sha256=h('planner report'),
            selection_uses_physics=False,P1_admission=False,P2_admission=False)
        cp['selection_digest']=m.planner_digest(cp);plans.append(None if i in missing else cp)
    consumed=dict(schema=m.PRODUCER+'.consume',status='wallcapped_records_verified_not_admitted',
        plans=plans,accounts=[dict(root=r,arm=a,export_missing=i in missing) for i,(r,_,a) in enumerate(m.ORDER)],
        report_sha256=h('planner report'),fixed_denominator=4,known_roots=2,fresh_roots=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False)
    args=dict(planner_output='/RAM/planner',planner_report_sha=h('planner report'),
        planner_receipt='/RAM/receipt.json',planner_receipt_sha=h('receipt'),planner_source_sha=h('source'))
    p=m.assemble(args,compiled,consumed,data)
    return p,data,compiled,consumed


def test_actual_producer_model_plan_abi_fixed4_full15_and_distinct_codecs():
    p,_,_,u=fixture();m.validate_execution(p)
    assert [(x['root_slot'],x['arm']) for x in p['plans']]==[(s,a) for s,_,a in m.ORDER]
    assert m.digest([1,2])!=m.planner_digest([1,2])
    assert all(len(x['commands'])==15 for x in u['plans'])
    assert m.contract()['caps']['public_step']==100
    assert m.contract()['caps']['physics_step']==1016
    assert m.contract()['caps']['explicit_render']==324


def test_full15_raw_literal_no_repair_and_negative_zero_bits():
    x=np.zeros((15,2),np.float32);x[0,0]=-0.
    assert m.raw15(x.tolist()).tobytes()==x.tobytes()
    for y in (x[:10].tolist(),np.full((15,2),1.01).tolist(),[[True,0.]]*15):
        with pytest.raises(ValueError):m.raw15(y)


def test_authorization_before_reads_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('read'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/o',lambda:pytest.fail('factory'),authorization='bad')


class MemoryStore(OldMemoryStore):
    def check(self,size=4096,final=False):
        m.require(sum(map(len,self.files.values()))+size<=m.CAP-(0 if final else m.FINAL), 'RAM own64MiB cap')


def execute(*,fault=None,at=11,close_error=False,missing=()):
    p,data,_,_=fixture(missing);store=MemoryStore();envs=[]
    def factory():
        e=Env(fault=fault,at=at,close_error=close_error);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        # Only this new module's dependency injection seams; frozen modules untouched.
        mp.setattr(m,'place',fake_place);mp.setattr(m,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM/own_prepare.json',prepare_sha256=h('own prepare'))
    return r,store,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_full4_full15_native_loop_and_complete_strict_lifecycle():
    r,s,p,data,envs=complete()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([n for n in s.files if n.endswith('.png')])==324
    for row,plan,e in zip(r['rows'],p['plans'],envs,strict=True):
        assert e.closed and e.dispatch_flags==[False]*25 and e.with_target is False
        assert np.asarray(e.commands[10:],np.float32).tobytes()==m.raw15(plan['commands']).tobytes()
        assert [x['step'] for x in row['checkpoints']]==[0,5,10,15] and row['missing_checkpoints']==[]
        assert len(row['responses'])==row['own_returned']==15 and row['warmup_returned']==10
        assert row['prefix51']['complete'] and '_set_state' not in e.__dict__ and 'reset' not in e.__dict__


@pytest.mark.parametrize('field',['order','raw','goal','context','task','warm','contexts','model','digest','partial','missing'])
def test_actual_plan_abi_tampering_rejected(field):
    p,data,c,u=fixture()
    if field=='order':u['plans'].reverse()
    elif field=='raw':u['plans'][0]['commands'][14][1]=.25
    elif field=='goal':u['plans'][0]['goal'][0]+=.1
    elif field=='context':u['plans'][0]['root_context']['latent_history'][0][0]+=.1
    elif field=='task':u['plans'][0]['task_id']=h('bad')
    elif field=='warm':u['plans'][0]['warm_raw10'][0][0]+=.001
    elif field=='contexts':u['plans'][0]['contexts'].pop()
    elif field=='model':u['plans'][0]['model_hash']=h('other model')
    elif field=='digest':u['plans'][0]['selection_digest']=h('bad')
    elif field=='partial':u['status']='partial'
    else:u['plans'].pop()
    with pytest.raises(ValueError):m.assemble(p['arguments'],c,u,data)


@pytest.mark.parametrize('fault,at',[('truncate',5),('terminal',11),('truncate',15),('truncate',20),('terminal',25)])
def test_native_stop_full4_no_padding_and_cp15_missing_when_not_reached(fault,at):
    r,s,p,data,envs=execute(fault=fault,at=at)
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4
    assert all(len(e.commands)==at and e.closed for e in envs)
    assert all(x['own_returned']==max(at-10,0) for x in r['rows'])
    assert all((15 in x['missing_checkpoints'])==(at<25) for x in r['rows'])


@pytest.mark.parametrize('fault',['throw','guard','substitute','render'])
def test_failed_incident_all4_no_action_repair_or_extra_dispatch(fault):
    r,s,p,data,envs=execute(fault=fault,at=11,close_error=True)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    assert all(e.closed for e in envs)
    assert all(x['first_error'] is not None for x in r['rows'])
    # Original zero command remains zero under this fixture's substitution;
    # those arms continue, never delete either the changed or unchanged action.
    expected=[25,11,25,11] if fault=='substitute' else [11]*4
    assert [len(e.commands) for e in envs]==expected
    if fault=='throw':assert len(m.validate_saved(r,s.events,p))==4
    else:
        # Preserve incident data/denominator, but never certify invalid native
        # action/RGB invariants merely because the driver recorded a failure.
        with pytest.raises(ValueError):m.validate_saved(r,s.events,p)


@pytest.mark.parametrize('field',['unknown','late_raw','cp15','reward','warmgoal','goalbind','renderer','count','arm','missing_return'])
def test_strict_saved_mutations_rejected_even_resealed(field):
    orig,s,prior,_,_=complete();r=deepcopy(orig);p=deepcopy(prior);events=deepcopy(s.events)
    if field=='unknown':events.insert(-1,dict(kind='unknown_event'))
    elif field=='late_raw':
        e=next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
            and e['receipt'].get('phase')=='plan' and e['receipt']['step']==15);e['receipt']['raw_bits']='00'
    elif field=='cp15':r['rows'][0]['checkpoints'].pop();r['rows'][0]['missing_checkpoints']=[15]
    elif field=='reward':
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==15);e['reward']=0.
    elif field=='warmgoal':
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='warmup');e['goal_state']=p['source_roots'][0]['goal']
    elif field=='goalbind':
        e=next(e for e in events if e['kind']=='goal_attached');e['goal'][0]+=.01
    elif field=='renderer':
        e=next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan' and e['receipt']['step']==15)
        e['pre']['renderer']['render_size']=512;e['post']['renderer']['render_size']=512
    elif field=='count':r['counts']['physics_step']['completed']-=1
    elif field=='arm':r['rows'][0]['arm']=m.ARMS[1]
    else:
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==15);events.remove(e)
    reseal(events,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,events,p)


def test_full_saved_consume_all324_png_and_unknown_artifact(monkeypatch):
    r,s,p,data,_=complete()
    # RAM bytes from the real adapter loop, not fabricated downstream schema.
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(s.files))
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    out=m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())
    assert out['saved_PNG_verified']==324 and out['fixed_denominator']==4 and not out['P2_admission']
    monkeypatch.setattr(m,'saved_artifacts',lambda _:dict(s.files,unknown=b'bad'))
    with pytest.raises(ValueError,match='unknown extra'):m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())


def test_saved_cap_is_own64_not_old32(tmp_path):
    # Sparse fixture bytes; no native/model work and no production artifact.
    p=tmp_path/'RAM_only.bin'
    with p.open('xb') as f:f.truncate(32*1024**2+1)
    assert len(m.saved_artifacts(tmp_path)[p.name])==32*1024**2+1


def test_literal_signed_zero_digest_is_not_silently_lost():
    p,data,c,u=fixture();x=u['plans'][0];x['commands'][0][0]=-0.
    # Previous positive-zero action bytes/digest cannot authenticate negative zero.
    with pytest.raises(ValueError):m.assemble(p['arguments'],c,u,data)


@pytest.mark.parametrize('missing',[(0,),(1,3),(0,1,2,3)])
def test_missing_exports_close_without_constructor_or_replacement(missing):
    r,s,p,data,envs=execute(missing=missing)
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4
    assert len(envs)==4-len(missing)
    assert r['counts']['constructor']['attempted']==4-len(missing)
    for i in missing:
        assert r['rows'][i]['plan'] is None and r['rows'][i]['status']=='export_missing'
        assert not any(op['branch']==i for op in r['actual_operation_receipts'])
    assert [e['receipt']['index'] for e in s.events if e['kind']=='branch_closed']==list(range(4))


def test_missing_slot_invented_execution_rejected():
    r,s,p,_,_=execute(missing=(0,))
    r['rows'][0]['own_returned']=1
    with pytest.raises(ValueError):m.validate_saved(r,s.events,p)


def test_real_store_emit_put_native_journal_abi(tmp_path):
    store=m.Store(str(tmp_path/(m.NAME+'_realcodec')))
    store.put('binding.json',dict(x=1))
    store.emit('first',payload=dict(x=1))
    store.emit('second',payload=dict(x=2))
    payload=(store.path/'journal.jsonl').read_bytes()
    assert payload.endswith(b'\n') and b'\n\n' not in payload
    events=[m.native.decode(x) for x in payload.splitlines()]
    assert len(events)==2 and events[0]['previous_sha256'] is None
    assert events[1]['previous_sha256']==events[0]['event_sha256']
    for e in events:
        assert e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
    assert m.read(store.path/'binding.json')==dict(x=1)


def test_material_source_seal_before_import_and_consume(monkeypatch):
    p,*_=fixture()
    monkeypatch.setattr(m,'sha',lambda _:h('not supplied'))
    monkeypatch.setattr(m.importlib,'import_module',lambda _:pytest.fail('import before source seal'))
    with pytest.raises(ValueError,match='source seal'):m.material(p['arguments'])


def test_export_presence_cannot_disagree_with_actual_ledger():
    p,data,c,u=fixture(missing=(0,))
    u['accounts'][0]['export_missing']=False
    with pytest.raises(ValueError,match='export presence'):m.assemble(p['arguments'],c,u,data)
    p,data,c,u=fixture()
    u['plans'][0]=None
    with pytest.raises(ValueError,match='export presence'):m.assemble(p['arguments'],c,u,data)


def test_actual_producer_compact_newline_digest_is_not_physical_codec():
    from experiments.spectral_reachability import agent_p2_wallcapped_cuda_20261029_v1 as runtime
    p,*_=fixture()
    for plan in p['plans']:
        chosen=plan['planner_plan'];body={k:v for k,v in chosen.items() if k!='selection_digest'}
        assert runtime.old.digest(body)==m.planner_digest(body)==chosen['selection_digest']
        assert m.digest(body)!=chosen['selection_digest']

