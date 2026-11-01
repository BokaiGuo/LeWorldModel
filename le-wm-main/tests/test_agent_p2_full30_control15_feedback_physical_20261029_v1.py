"""RAM callbacks and temporary disk only; no model/native physics/production."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_physical_20261029_v1 as m
from test_agent_p2_two_program_full30_physical_20261029_v1 import complete as parent_complete
from test_agent_allmode_physical_20261028_v1 import Env, fake_place, h, reseal


def fixture(job_index=0):
    old,store,op,data,_=parent_complete()  # Synthetic parent RAM authority, not production consume.
    compiled=dict(schema=m.PRODUCER+'.prepare',expected_model_hash=op['expected_model_hash'],jobs=[],
                  bases=[dict(base_index=i) for i in range(4)],inherited_resources={'fixture':True})
    new=[]
    for base,original in enumerate(op['plans']):
        cp=original['planner_plan'];u=np.asarray(cp['logits']);a=m.raw30(cp['commands'])
        cap=m.K.C15 if base%2==0 else float(np.sqrt(m.K.C30**2-u[:30]@u[:30]))
        for pol in ('I','F'):
            job=dict(base_index=base,root_slot=base//2,arm=cp['arm'],policy=pol,
                task_id=cp['task_id'],task_sha256=cp['task_sha256'],parent_witness_sha256=cp['witness_sha256'],
                parent_logits=u.tolist(),parent_raw_bits=cp['raw_float32_bits'],prefix_raw=a[:15].tolist(),
                incumbent_logits=u[30:].tolist(),suffix_cap=cap,goal=cp['goal'],input_context=cp['contexts'][2])
            compiled['jobs'].append(job)
            result=dict(commands=a.tolist(),raw_float32_bits=a.tobytes().hex(),suffix_logits=u[30:].tolist(),
                        new_suffix_Q=m.K.objective(cp['contexts'][-1],cp['goal']),contexts=cp['contexts'][3:])
            new.append(m.planner.row_plan(compiled,job,dict(result=result)))
    receipt=dict(schema=m.PRODUCER+'.consume',status='control15_feedback_verified_NOT_admitted',errors=[],plans=new,
        report_sha256=m.PINS['report']['sha256'],prepare_sha256=m.PINS['prepared']['sha256'],
        completion_sha256=m.PINS['completion']['sha256'],policy_slots=[{}]*12,actual_new=m.planner.contract()['total'])
    reference=[m.normalized_prefix_event(e) for e in store.events if e.get('branch')==job_index and m.is_prefix_event(e)]
    p=m.assemble(job_index,compiled,receipt,op,old['rows'],reference)
    return p,data,compiled,receipt


class MemoryStore(m.Store):
    def __init__(self):
        self.path=Path('/RAM_ONLY');self.files={};self.events=[];self.prefix_events={};self.previous=None;self.index=0
        self.sizes={};self.bytes=0;self.finalizing=False
    def check(self,size=4096,final=False):
        m.require(self.bytes+size<=m.CAP-(0 if final or self.finalizing else m.FINAL),'RAM total/reserve')
    def write(self,name,value,append=False,final=False):
        self.check(len(value),final)
        m.require(append or name not in self.files,'exclusive RAM artifact')
        self.files[name]=self.files.get(name,b'')+value if append else value
        self.bytes+=len(value);self.sizes[name]=len(self.files[name])
        return hashlib.sha256(self.files[name]).hexdigest()
    def emit(self,kind,**fields):
        start=len(self.files.get('journal.jsonl',b''));super().emit(kind,**fields)
        self.events.append(m.native.decode(self.files['journal.jsonl'][start:]))


def execute(*,fault=None,at=11,close_error=False,prefix_fault=False):
    p,data,_,_=fixture();store=MemoryStore();envs=[]
    if prefix_fault:
        e=next(e for e in p['prefix_reference'] if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan'
               and e['receipt']['step']==15)
        e['receipt']['pixel_sha256']=h('wrong actual RGB')
    def factory():
        e=Env(fault=fault,at=at,close_error=close_error);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m,'place',fake_place);mp.setattr(m,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        report=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM/own_prepare.json',prepare_sha256=h('prepare'))
    return report,store,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_complete_two_trajectory_field_transport_budget():
    r,s,p,_,envs=complete()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==2
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    budget=m.byte_census(s.files);print('COMPLETE_TWO_TRAJECTORY_TRANSPORT',budget)
    print('COMPLETE_TWO_TRAJECTORY_SHAPE_BOUND',m.shape_budget(s.files))
    assert budget['PNG']==252 and budget['journal_bytes']<=m.JOURNAL_CAP and budget['other_bytes']<=m.OTHER_CAP
    assert len(r['policy_roster'])==12 and all(e.closed and len(e.commands)==40 for e in envs)
    assert m.contract()['budget_proof']['total_reserved_bytes']==119*1024**2
    assert len(p['prefix_reference'])==106
    for i,row in enumerate(r['rows']):
        assert row['own_returned']==30 and row['warmup_returned']==10
        assert row['feedback_guard']==m.feedback_guard_value(p['plans'][i],p)
        assert row['missing_checkpoints']==[]


def test_saved_consume_ram_full252_png(monkeypatch):
    r,s,p,data,_=complete();monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(s.files))
    monkeypatch.setattr(m,'verify',lambda *_:(deepcopy(p),data))
    result=m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest(),hashlib.sha256(s.files['completion.json']).hexdigest())
    assert result['saved_PNG_verified']==252 and result['job_index']==0 and len(result['policy_roster'])==12


def test_prefix_mismatch_never_dispatches_suffix():
    r,s,p,_,envs=execute(prefix_fault=True)
    assert r['status']=='partial_or_failed'
    assert all(len(e.commands)==25 and e.closed for e in envs)
    assert all(row['own_returned']==15 and row['feedback_guard'] is None for row in r['rows'])
    assert not any(e['kind']=='feedback_prefix_guard' for e in s.events)


@pytest.mark.parametrize('fault,at',[('truncate',5),('truncate',20),('truncate',25),('truncate',35)])
def test_native_stop_missing_no_padding(fault,at):
    r,s,p,_,envs=execute(fault=fault,at=at)
    assert all(len(e.commands)==at and e.closed for e in envs)
    assert all(row['own_returned']==max(0,at-10) for row in r['rows'])
    assert all((row['feedback_guard'] is None)==(at<=25) for row in r['rows'])
    assert all((30 in row['missing_checkpoints']) for row in r['rows'])


@pytest.mark.parametrize('mutation',['unknown','remove_guard','late_guard','guard_hash','prefix_state','policy','count'])
def test_strict_resealed_feedback_mutations(mutation):
    orig,s,p,_,_=complete();r=deepcopy(orig);events=deepcopy(s.events)
    if mutation=='unknown':events.insert(-1,dict(kind='unknown_event'))
    elif mutation=='remove_guard':events.remove(next(e for e in events if e['kind']=='feedback_prefix_guard'))
    elif mutation=='late_guard':
        i=next(i for i,e in enumerate(events) if e['kind']=='feedback_prefix_guard');guard=events.pop(i)
        after=next(i for i,e in enumerate(events) if e['kind']=='operation_dispatch_intent' and
                   e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt']['step']==16)
        events.insert(after+1,guard)
    elif mutation=='guard_hash':next(e for e in events if e['kind']=='feedback_prefix_guard')['guard']['prefix_reference_sha256']=h('bad')
    elif mutation=='prefix_state':
        e=next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan' and e['receipt']['step']==15)
        e['receipt']['state'][0]+=.1
    elif mutation=='policy':r['rows'][0]['policy']='F'
    else:r['counts']['physics_step']['completed']-=1
    reseal(events,r)
    with pytest.raises(ValueError):m.validate_saved(r,events,p)


def test_wrong_auth_before_reads_or_callbacks(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *_:pytest.fail('read before auth'))
    with pytest.raises(ValueError):m.run('x','y','z',lambda:pytest.fail('factory'),authorization='bad')


def test_actual_disk_store_codec_and_exclusive(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_codec'));s.put('metadata.json',dict(a=1))
    s.emit('one');s.emit('two')
    data=(s.path/'journal.jsonl').read_bytes();assert b'\n\n' not in data
    events=[m.native.decode(line) for line in data.splitlines()]
    assert events[0]['event_sha256']==m.source.digest({k:v for k,v in events[0].items() if k!='event_sha256'})
    assert events[1]['previous_sha256']==events[0]['event_sha256']
    with pytest.raises(ValueError):s.put('metadata.json',dict(a=2))


@pytest.mark.parametrize('job',[True,-1,4,None])
def test_invalid_fixed_job_index(job):
    p,data,c,u=fixture()
    with pytest.raises(ValueError):m.assemble(job,c,u,{'plans':[]},[],[])


@pytest.mark.parametrize('group',['png','journal','metadata','whole'])
def test_actual_store_group_caps_write_before_dispatch(tmp_path,group):
    s=m.Store(tmp_path/(m.NAME+'_caps'))
    if group=='png':name,data='x.png',b'x'*(m.PNG_CAP+1)
    elif group=='journal':name,data='journal.jsonl',b'x'*(m.JOURNAL_CAP+1)
    elif group=='metadata':name,data='x.json',b'x'*(m.OTHER_CAP+1)
    else:
        s.bytes=m.CAP-m.FINAL;name,data='x.png',b'x'
    with pytest.raises(ValueError):s.write(name,data)
    assert not (s.path/name).exists()


def test_verify_elapsed_before_any_output_or_env(tmp_path,monkeypatch):
    p,data,*_=fixture();times=iter((0.,901.))
    monkeypatch.setattr(m.time,'monotonic',lambda:next(times,901.))
    monkeypatch.setattr(m,'verify',lambda *_:(p,data))
    out=tmp_path/(m.NAME+'_expired')
    with pytest.raises(ValueError):m.run('p',h('p'),out,lambda:pytest.fail('env'),authorization=m.AUTH)
    assert not out.exists()


def test_postguard_failure_keeps_attempt_and_cleanup():
    r,s,p,_,envs=execute(fault='throw',at=26,close_error=True)
    assert r['status']=='partial_or_failed' and len(r['policy_roster'])==12
    assert all(e.closed and len(e.commands)==26 for e in envs)
    assert all(row['feedback_guard'] is not None and row['own_attempted']==16 and row['own_returned']==15
               and row['first_error'] and row['secondary_errors'] for row in r['rows'])


@pytest.mark.parametrize('field',['raw','context','domain','witness','Q','order'])
def test_new_plan_no_repair_or_reselection(field):
    p,data,c,u=fixture();u=deepcopy(u)
    if field=='raw':u['plans'][0]['commands'][16][0]+=.01
    elif field=='context':u['plans'][0]['input_context']['latent_history'][0][0]+=.1
    elif field=='domain':u['plans'][0]['suffix_cap']*=2
    elif field=='witness':u['plans'][0]['new_witness_sha256']=h('bad')
    elif field=='Q':u['plans'][0]['new_suffix_Q']+=.1
    else:u['plans'][0],u['plans'][1]=u['plans'][1],u['plans'][0]
    old,_,op,_,_=parent_complete()
    with pytest.raises(ValueError):m.assemble(0,c,u,op,old['rows'],p['prefix_reference'])


def disk_build_fixture(tmp_path,monkeypatch):
    """Actual material ABI: saved JSON/pins/journal/inventory, no fullstrict."""
    _,data,c,u=fixture()
    old,store,op,_,_=parent_complete()
    c=deepcopy(c);u=deepcopy(u);op=deepcopy(op);old=deepcopy(old);data=deepcopy(data)
    def put(path,value,encoded=False):
        body=value if encoded else m.raw(value)
        path.write_bytes(body)
        return dict(path=str(path),sha256=hashlib.sha256(body).hexdigest())
    dependency=tmp_path/'original_task.bin';dep=put(dependency,b'fixture source bytes',True)
    c['bindings_sha256']={dep['path']:dep['sha256']}
    op['bindings']={dep['path']:dep['sha256']};data['bindings']={dep['path']:dep['sha256']}
    parent_dir=tmp_path/'parent';parent_dir.mkdir()
    jp=parent_dir/'journal.jsonl';put(jp,store.files['journal.jsonl'],True)
    old['journal_sha256']=m.sha(jp)
    parent_prepare=put(tmp_path/'parent_prepare.json',op)
    parent_report=put(parent_dir/'report.json',old)
    parent_receipt=put(tmp_path/'parent_receipt.json',dict(rows=old['rows'],
        physical_report_sha256=parent_report['sha256'],bindings=op['bindings']))
    parent_pins=dict(physical_prepared=parent_prepare['path'],physical_prepare_sha=parent_prepare['sha256'],
        physical_output=str(parent_dir),physical_report_sha=parent_report['sha256'],
        physical_receipt=parent_receipt['path'],physical_receipt_sha=parent_receipt['sha256'])
    model_dir=tmp_path/'model';model_dir.mkdir()
    artifact=put(model_dir/'native_io.json',dict(real_saved_ABI=True))
    prepared=put(tmp_path/'model_prepare.json',c)
    report=put(model_dir/'report.json',dict(schema=m.PRODUCER+'.report',
        status='all8_feedback_sealed_NOT_admitted',prepared=prepared['path'],prepare_sha256=prepared['sha256'],
        errors=[],model_before=c['expected_model_hash'],model_after=c['expected_model_hash'],
        artifact_sha256={'native_io.json':artifact['sha256']}))
    completion=put(model_dir/'completion.json',dict(report_sha256=report['sha256'],elapsed_seconds=1.))
    u.update(report_sha256=report['sha256'],prepare_sha256=prepared['sha256'],completion_sha256=completion['sha256'])
    receipt=put(tmp_path/'model_receipt.json',u)
    monkeypatch.setattr(m,'PINS',dict(prepared=prepared,report=report,completion=completion,receipt=receipt))
    monkeypatch.setattr(m,'PARENT_PINS',parent_pins)
    monkeypatch.setattr(m,'source_data',lambda:deepcopy(data))
    return artifact,dependency


def test_complete_build_real_disk_pins_inventory_source_and_verify(tmp_path,monkeypatch):
    disk_build_fixture(tmp_path,monkeypatch)
    for job in range(4):
        p,data=m.material(job)
        assert p['job_index']==job and len(p['plans'])==2 and len(p['prefix_reference'])==106
        assert p['planner_authority']['full_upstream_rebuilt_here'] is False
        assert all(str(f) in p['bindings'] for f in m.own_files())
        out=tmp_path/(m.NAME+f'_prepare_J{job}.json')
        descriptor=m.prepare(out,job)
        rebuilt,_=m.verify(out,descriptor['sha256'])
        assert m.raw(rebuilt)==m.raw(p) and out.stat().st_size<m.META
        print('FULL_BUILD_JOB',job,'PREPARE_BYTES',out.stat().st_size)


@pytest.mark.parametrize('mutation',['extra_artifact','missing_artifact','artifact_hash','task_hash','own_prepare'])
def test_complete_build_disk_tamper_failclosed(tmp_path,monkeypatch,mutation):
    artifact,task=disk_build_fixture(tmp_path,monkeypatch)
    out=tmp_path/(m.NAME+'_prepare.json');descriptor=m.prepare(out,0)
    if mutation=='extra_artifact':(Path(artifact['path']).parent/'unknown.bin').write_bytes(b'unknown')
    elif mutation=='missing_artifact':Path(artifact['path']).unlink()
    elif mutation=='artifact_hash':Path(artifact['path']).write_bytes(b'changed model IO')
    elif mutation=='task_hash':task.write_bytes(b'changed task')
    else:out.write_bytes(out.read_bytes()+b' ')
    with pytest.raises(ValueError):m.verify(out,descriptor['sha256'])


def test_four_job_receipt_join_counts_and_no_missing_or_reordered(tmp_path):
    r,_,p,_,_=complete();paths=[];hashes=[]
    for job in range(4):
        # Synthetic saved own-receipt ABI only, not claimed native evidence.
        value=dict(schema=m.NAME+'.consume',job_index=job,rows=deepcopy(r['rows']),
            planner_authority=p['planner_authority'],global_policy_slots=12,bindings={},
            counts=r['counts'],full_stage_wall_seconds=float(job+1))
        path=tmp_path/f'job{job}.json';path.write_bytes(m.raw(value))
        paths.append(str(path));hashes.append(m.sha(path))
    joined=m.join(paths,hashes)
    assert joined['fixed_denominator']==len(joined['policy_slots'])==12
    assert joined['sum_job_whole_wall_seconds']==10.
    assert joined['counts']['public_step']==dict(attempted=320,completed=320)
    assert joined['counts']['physics_step']==dict(attempted=3232,completed=3232)
    assert joined['inherited_O']==4 and joined['new_trajectories']==8
    with pytest.raises(ValueError):m.join(paths[:3],hashes[:3])
    with pytest.raises(ValueError):m.join(paths[::-1],hashes[::-1])
    with pytest.raises(ValueError):m.join(paths,[h('bad')]+hashes[1:])
