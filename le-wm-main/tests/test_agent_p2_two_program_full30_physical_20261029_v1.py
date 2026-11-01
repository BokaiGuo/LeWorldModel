"""RAM-only native runner tests: no neural, native physics or production I/O."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_two_program_full30_physical_20261029_v1 as m
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, MemoryStore as OldMemoryStore, Env, fake_place, h, reseal
from test_agent_allmode_prefix_20261028_v1 import slot as model_slot


def fixture():
    _,data,*_=old_fixture()
    compiled=dict(schema=m.PRODUCER+'.prepare',expected_model_hash=h('model'),slots=[],bindings_sha256={})
    for root in (0,1):
        s=model_slot(root*2)
        ctx=deepcopy(s['packet']['root_context'])
        ctx['action_history']=m.normalization.normalized_history(m.moving.warmup_commands()).tolist()
        programs=[dict(id=f'r{root}/c{j}',program_sha256=h((root,j)),
            scope=dict(normalization=dict(mean=m.normalization.ACTION_MEAN.tolist(),std=m.normalization.ACTION_STD.tolist()))) for j in (1,2)]
        compiled['slots'].append(dict(root_slot=root,condition='off',task_id=h(('task',root)),
            task_sha256=h(('taskfile',root)),blind_sha256=data['exports'][(root,'off')]['sha256'],
            root_context=ctx,root_warm_raw10=m.moving.warmup_commands().tolist(),goal=s['packet']['goal'],programs=programs))
    plans=[]
    for i,(root,_,arm) in enumerate(m.ORDER):
        s=compiled['slots'][root];u=np.full(60,.125*(i%2),np.float64)
        commands=m.K.chart(u).reshape(30,2);ctx=deepcopy(s['root_context']);contexts=[]
        for j in range(6):
            ctx=deepcopy(ctx)
            ctx['latent_history']=ctx['latent_history'][1:]+[deepcopy(ctx['latent_history'][-1])]
            ctx['action_history']=ctx['action_history'][1:]+m.normalization.normalized_history(commands[j*5:j*5+5]).tolist()
            contexts.append(ctx)
        cp=dict(schema=m.PRODUCER+'.model_plan',root_slot=root,arm=arm,task_id=s['task_id'],
            task_sha256=s['task_sha256'],root_context=deepcopy(s['root_context']),goal=deepcopy(s['goal']),
            root_warm_raw10=s['root_warm_raw10'],commands=commands.tolist(),raw_float32_bits=commands.tobytes().hex(),
            source_programs=[dict(id=x['id'],sha256=x['program_sha256']) for x in s['programs']],
            normalization=s['programs'][0]['scope']['normalization'],new_raw_steps=30,logits=u.tolist(),
            program_count=2 if arm==m.ARMS[0] else 1,program_start_controls=[0,15] if arm==m.ARMS[0] else [0],
            new_domain=dict(logit_dim=60,logit_cap=m.K.C30,raw_bounds=[-1,1]),
            contexts=contexts,model_hash=compiled['expected_model_hash'],predicted_Q=m.K.objective(contexts[-1],s['goal']),**m.K.FLAGS)
        cp['witness_sha256']=m.K.digest(cp);plans.append(cp)
    consumed=dict(schema=m.PRODUCER+'.consume',status='full30_transfer_and_joint_verified_NOT_admitted',
        plans=plans,report_sha256=m.PINS['planner_report_sha'],prepare_sha256=m.PINS['planner_prepare_sha'],
        errors=[],**m.K.FLAGS)
    args=deepcopy(m.PINS);p=m.assemble(args,compiled,consumed,data)
    return p,data,compiled,consumed


def test_actual_producer_model_plan_abi_fixed4_full30_and_distinct_codecs():
    p,_,_,u=fixture();m.validate_execution(p)
    assert [(x['root_slot'],x['arm']) for x in p['plans']]==[(s,a) for s,_,a in m.ORDER]
    assert m.digest([1,2])!=m.planner_digest([1,2])
    assert all(len(x['commands'])==30 for x in u['plans'])
    assert m.contract()['caps']['public_step']==160
    assert m.contract()['caps']['physics_step']==1616
    assert m.contract()['caps']['explicit_render']==504


def test_full30_raw_literal_no_repair_and_negative_zero_bits():
    x=np.zeros((30,2),np.float32);x[0,0]=-0.
    assert m.raw30(x.tolist()).tobytes()==x.tobytes()
    for y in (x[:10].tolist(),np.full((30,2),1.01).tolist(),[[True,0.]]*30):
        with pytest.raises(ValueError):m.raw30(y)


def test_authorization_before_reads_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('read'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/o',lambda:pytest.fail('factory'),authorization='bad')


class MemoryStore(OldMemoryStore):
    def check(self,size=4096,final=False):
        final=final or getattr(self,'finalizing',False)
        m.require(sum(map(len,self.files.values()))+size<=m.CAP-(0 if final else m.FINAL), 'RAM own64MiB cap')


def execute(*,fault=None,at=11,close_error=False):
    p,data,_,_=fixture();store=MemoryStore();envs=[]
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


def test_full4_full30_native_loop_and_complete_strict_lifecycle():
    r,s,p,data,envs=complete()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([n for n in s.files if n.endswith('.png')])==504
    for row,plan,e in zip(r['rows'],p['plans'],envs,strict=True):
        assert e.closed and e.dispatch_flags==[False]*40 and e.with_target is False
        assert np.asarray(e.commands[10:],np.float32).tobytes()==m.raw30(plan['commands']).tobytes()
        assert [x['step'] for x in row['checkpoints']]==[0,5,10,15,20,25,30] and row['missing_checkpoints']==[]
        assert len(row['responses'])==row['own_returned']==30 and row['warmup_returned']==10
        assert row['prefix51']['complete'] and '_set_state' not in e.__dict__ and 'reset' not in e.__dict__


@pytest.mark.parametrize('field',['order','raw','goal','context','task','warm','contexts','model','digest','partial','missing'])
def test_actual_plan_abi_tampering_rejected(field):
    p,data,c,u=fixture()
    if field=='order':u['plans'].reverse()
    elif field=='raw':u['plans'][0]['commands'][14][1]=.25
    elif field=='goal':u['plans'][0]['goal'][0]+=.1
    elif field=='context':u['plans'][0]['root_context']['latent_history'][0][0]+=.1
    elif field=='task':u['plans'][0]['task_id']=h('bad')
    elif field=='warm':u['plans'][0]['root_warm_raw10'][0][0]+=.001
    elif field=='contexts':u['plans'][0]['contexts'].pop()
    elif field=='model':u['plans'][0]['model_hash']=h('other model')
    elif field=='digest':u['plans'][0]['witness_sha256']=h('bad')
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
    expected=[40,11,40,11] if fault=='substitute' else [11]*4
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
            and e['receipt'].get('phase')=='plan' and e['receipt']['step']==30);e['receipt']['raw_bits']='00'
    elif field=='cp15':r['rows'][0]['checkpoints'].pop();r['rows'][0]['missing_checkpoints']=[15]
    elif field=='reward':
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==30);e['reward']=0.
    elif field=='warmgoal':
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='warmup');e['goal_state']=p['source_roots'][0]['goal']
    elif field=='goalbind':
        e=next(e for e in events if e['kind']=='goal_attached');e['goal'][0]+=.01
    elif field=='renderer':
        e=next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan' and e['receipt']['step']==30)
        e['pre']['renderer']['render_size']=512;e['post']['renderer']['render_size']=512
    elif field=='count':r['counts']['physics_step']['completed']-=1
    elif field=='arm':r['rows'][0]['arm']=m.ARMS[1]
    else:
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==30);events.remove(e)
    reseal(events,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,events,p)


def test_full_saved_consume_all324_png_and_unknown_artifact(monkeypatch):
    r,s,p,data,_=complete()
    # RAM bytes from the real adapter loop, not fabricated downstream schema.
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(s.files))
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    out=m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())
    assert out['saved_PNG_verified']==504 and out['fixed_denominator']==4 and not out['P2_admission']
    monkeypatch.setattr(m,'saved_artifacts',lambda _:dict(s.files,unknown=b'bad'))
    with pytest.raises(ValueError,match='unknown extra'):m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())


def test_metadata_group_cap_not_just_whole_output_cap(tmp_path):
    # Sparse fixture bytes; no native/model work and no production artifact.
    p=tmp_path/'RAM_only.bin'
    with p.open('xb') as f:f.truncate(m.OTHER_CAP+1)
    with pytest.raises(ValueError,match='budget proof'):m.saved_artifacts(tmp_path)


def test_literal_signed_zero_digest_is_not_silently_lost():
    p,data,c,u=fixture();x=u['plans'][0];x['commands'][0][0]=-0.
    # Previous positive-zero action bytes/digest cannot authenticate negative zero.
    with pytest.raises(ValueError):m.assemble(p['arguments'],c,u,data)


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


def test_budget_proof_full_RAM_bytes_and_no_control15_reset():
    r,s,p,_,envs=complete();c=m.byte_census(s.files)
    assert c['PNG']==504 and c['other_bytes']<=m.OTHER_CAP and c['journal_bytes']<=m.JOURNAL_CAP
    assert m.contract()['budget_proof']['total_reserved_bytes']==214*1024**2<m.CAP
    for i in range(4):
        ops=[o for o in r['actual_operation_receipts'] if o['branch']==i]
        assert sum(o['resource']=='reset' for o in ops)==2
        plan=[o for o in ops if o['resource']=='public_step' and o.get('phase')=='plan']
        assert [o['step'] for o in plan]==list(range(1,31))
        assert not any(o['resource'] in ('reset','set_state','goal_bind') and o['index']>plan[0]['index'] for o in ops)


@pytest.mark.parametrize('mutation',['missing','late','raw','pixels'])
def test_native_history_guard_durable_before_plan_intent(mutation):
    orig,s,p,_,_=complete();r=deepcopy(orig);events=deepcopy(s.events)
    guard=next(e for e in events if e['kind']=='native_history_guard')
    first=next(e for e in events if e['kind']=='operation_dispatch_intent'
        and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan')
    assert guard['index']<first['index']
    if mutation=='missing':events.remove(guard)
    elif mutation=='late':
        events.remove(guard);events.insert(events.index(first)+1,guard)
    elif mutation=='raw':guard['guard']['warm_raw_float32_bits']='00'
    else:guard['guard']['observed_history_pixel_sha256'][0]=h('other')
    reseal(events,r)
    with pytest.raises(ValueError):m.validate_saved(r,events,p)


@pytest.mark.parametrize('field',['action_history','latent_history','normalizer','program','chart'])
def test_coherent_resealed_model_seam_mutation_rejected(field):
    p,data,c,u=fixture();x=u['plans'][0]
    if field in ('action_history','latent_history'):x['contexts'][3][field][0][0]+=.125
    elif field=='normalizer':x['normalization']['mean'][0]=0.
    elif field=='program':x['source_programs'][1]['sha256']=h('different')
    else:x['logits'][0]=.01
    x['witness_sha256']=m.K.digest({k:v for k,v in x.items() if k!='witness_sha256'})
    with pytest.raises(ValueError):m.assemble(p['arguments'],c,u,data)


def test_wrong_authority_refused_before_any_saved_reads(monkeypatch):
    a=deepcopy(m.PINS);a['planner_prepare_sha']=h('other')
    monkeypatch.setattr(m,'sha',lambda _:pytest.fail('must reject different authority before reads'))
    with pytest.raises(ValueError,match='pinned'):m.material(a)


def test_bound_file_tamper_and_symlink_rejected(tmp_path):
    p=tmp_path/'fixture.txt';p.write_bytes(b'fixture')
    seals={str(p):m.sha(p)};m.rehash(seals)
    p.write_bytes(b'mutation')
    with pytest.raises(ValueError,match='immutable binding'):m.rehash(seals)
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):m.rehash({str(link):m.sha(p)})


def test_verify_time_exhaustion_no_store_or_dispatch(monkeypatch):
    p,data,*_=fixture();clock=[0.]
    def verify(*args):clock[0]=m.WALL;return p,data
    monkeypatch.setattr(m,'verify',verify);monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(m,'Store',lambda _:pytest.fail('no output after verify timeout'))
    with pytest.raises(ValueError,match='verify exhausted'):
        m.run('/RAM/p',h('p'),'/RAM/out',lambda:pytest.fail('no physics'),authorization=m.AUTH)


@pytest.mark.parametrize('field',['wall','bool','reporthash','unknown','blank'])
def test_completion_or_journal_mutations_rejected(monkeypatch,field):
    _,s,p,data,_=complete();files=deepcopy(s.files)
    if field=='blank':files['journal.jsonl']+=b'\n'
    elif field=='unknown':files['terminal_failure.json']=b'{}\n'
    else:
        c=m.native.decode(files['completion.json'])
        if field=='wall':c['wall_seconds']=m.WALL+1
        elif field=='bool':c['wall_seconds']=True
        else:c['report_sha256']=h('wrong')
        files['completion.json']=m.raw(c)
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(files))
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    with pytest.raises(ValueError):m.consume('/RAM/physical',hashlib.sha256(files['report.json']).hexdigest())


def test_real_disk_whole_fixture_reader_and_PNG_mutation(tmp_path,monkeypatch):
    _,ram,p,data,_=complete();s=m.Store(str(tmp_path/(m.NAME+'_diskfixture')))
    for n,b in ram.files.items():s.write(n,b,append=(n=='journal.jsonl'),final=True)
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    out=m.consume(str(s.path),m.sha(s.path/'report.json'))
    assert out['saved_PNG_verified']==504 and out['all_four_physically_executed']
    assert out['full_upstream_rebuilt_here'] is False and out['own_native_saved_reconstructed_here'] is True
    f=next(s.path.glob('*.png'));f.write_bytes(b'corrupt')
    with pytest.raises(ValueError):m.consume(str(s.path),m.sha(s.path/'report.json'))


def test_real_store_caps_reserve_exclusive_and_closed_fsync(tmp_path,monkeypatch):
    s=m.Store(str(tmp_path/(m.NAME+'_caps')))
    s.put('one.json',dict(scalar=1.))
    with pytest.raises(ValueError,match='exclusive'):s.put('one.json',dict(scalar=2.))
    with pytest.raises(ValueError,match='PNG'):s.write('huge.png',b'0'*(m.PNG_CAP+1))
    with pytest.raises(ValueError,match='metadata'):s.write('huge.json',b'0'*(m.OTHER_CAP+1))
    s.bytes=m.CAP-m.FINAL
    with pytest.raises(ValueError):s.check(1)
    s.check(1,final=True)


def test_timeout_still_closes_environment_and_writes_partial(monkeypatch):
    p,data,*_=fixture();store=MemoryStore();clock=[0.];envs=[]
    def factory():
        e=Env();envs.append(e);clock[0]=m.WALL+1;return e
    monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(m,'validate_env',lambda _:None)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
    r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM/p',prepare_sha256=h('p'),started=0.)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    assert len(envs)==1 and envs[0].closed
    assert r['counts']['constructor']['completed']==r['counts']['close']['completed']==1
    assert r['counts']['public_step']['attempted']==0 and 'report.json' in store.files
    assert 'terminal_failure.json' in store.files and 'completion.json' not in store.files



