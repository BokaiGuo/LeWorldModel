"""Real temporary writer/reader with inert RAM callbacks; no native environment."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from experiments.spectral_reachability import agent_p3_settling_tail_physical_20261029_v1 as m


def helpers():
    path=m.ROOT/'tests/test_agent_p3_same_time_capability_probe_physical_20261029_v2.py'
    spec=importlib.util.spec_from_file_location('stage_c_fixture_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def data():
    h=helpers();inputs=h.inputs.__wrapped__();original_observation=h.observation
    def observation(root,phase,step,overlay):
        # The old synthetic fixture reset plan0 to warm0; actual plan0 is the
        # existing warm10 state after an inert goal attachment.
        if phase=='plan' and step==0:
            e=original_observation(root,'warmup',10,overlay)
            e['receipt']['phase']='plan';e['receipt']['step']=0
            return e
        return original_observation(root,phase,step,overlay)
    h.observation=observation
    for row in inputs[0]['rows']:
        row['observations'][33:36]=[observation(row['root_slot'],'plan',0,i) for i in range(3)]
    return h,inputs,h.prepared(inputs)


def old_fixture(tmp_path,h,inputs,p,i):
    store,ref,seal=h.drive_fixture(tmp_path/f'old{i}',p,inputs[0],i)
    r=m.old.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,permit_fixture=True)
    return dict(job=p['jobs'][i],old_row=r['row'],old_seal=seal)


def drive(tmp_path,h,inputs,p,original,i=0,stop=None,fail_guard=False,fail_fsync=False,corrupt_cp15=False,output=None):
    job=p['jobs'][i];ref=inputs[0]['rows'][job['endpoint']]
    p2=dict(authority=p['authority'],limits=m.limits([original]))
    binding=m.binding_for(p2,job,'/fixture/prepare.json','a'*64,fixture=True)
    store=m.Store(str(output if output is not None else tmp_path/f'new{i}'),binding,p2['limits']);store.emit('run_started',binding_sha256=m.digest(binding))
    meter=m.Meter(store,job,ref,original);closed=dict(status='horizon_completed',error=None,cleanup_errors=[])
    if fail_guard:
        real_emit=store.emit
        def emit(kind,**kw):
            if kind=='guard' and kw['n']==81:raise OSError('fsync fixture')
            return real_emit(kind,**kw)
        store.emit=emit
    if fail_fsync:
        real_write=store.write
        def write(name,body,**kwargs):
            if name=='journal.jsonl' and b'"n": 81' in body and b'"kind": "guard"' in body:
                fsync=m.os.fsync
                def reject(_):raise OSError('actual journal fsync fixture')
                try:
                    m.os.fsync=reject
                    return real_write(name,body,**kwargs)
                finally:m.os.fsync=fsync
            return real_write(name,body,**kwargs)
        store.write=write
    def reset_call():
        for _ in range(2):meter.call('set_state',lambda:meter.call('physics_step',lambda:None))
        meter.call('goal_setter',lambda:None);meter.call('goal_render',lambda:None)
        return dict(goal_state=[0.]*7,state=h.state(job['root_slot'],0),snapshot=h.snapshot(h.state(job['root_slot'],0)))
    try:
        meter.call('constructor',lambda:None)
        meter.call('reset',reset_call,role='seed',seed=51+job['root_slot'])
        meter.call('reset',reset_call,role='placement',seed=0,requested_state=h.state(job['root_slot'],0))
        for phase,commands in (('warmup',job['warm10']),('plan',job['raw15']+m.gate.TAIL)):
            if phase=='plan':meter.call('goal_bind',lambda:meter.call('goal_setter',lambda:None))
            for step in range(len(commands)+1):
                stopping=stop==(phase,step)
                if step:
                    raw=commands[step-1]
                    def public_call():
                        for _ in range(10):meter.call('physics_step',lambda:None)
                        return h.result(job['root_slot'],step,raw,truncated=stopping)
                    meter.call('public_step',public_call,phase=phase,step=step,raw_bits=''.join(m.old.bits([raw],1)))
                for overlay in range(3):
                    meter.call('explicit_render',lambda:None)
                    event=h.observation(job['root_slot'],phase,step,overlay);rc=event['receipt']
                    rc['png']=f'b{i}_{phase}_{step:02}_{overlay}.png';rc['file_sha256']=store.frame(rc['png'],h.PIXELS)
                    store.emit('fresh_observation',**meter.where,receipt=rc,pre=event['pre'],post=event['post'])
                if corrupt_cp15 and phase=='plan' and step==15:
                    (store.path/f'b{i}_plan_15_0.png').write_bytes(b'altered after capture')
                if stopping:closed['status']='native_stopped';raise m.old.NativeStop()
    except m.old.NativeStop:closed['error']=None
    except BaseException as e:closed.update(status='engineering_failed',error=m.old.error(e))
    finally:
        meter.call('close',lambda:None,cleanup=True)
        store.emit('branch_closed',result=closed);store.emit('run_finished',result=closed)
    try:seal=m.publish(store,ref,original)
    except BaseException as e:seal=m.failed_publication(store,meter,closed,e)
    return store,ref,seal,meter


@pytest.mark.parametrize('index',range(10))
def test_full_disk_each_fixed_branch(data,tmp_path,index):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,index)
    store,ref,seal,meter=drive(tmp_path,h,inputs,p,original,index)
    result=m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,original,permit_fixture=True)
    row=result['row']
    assert result['event_count']==990 and row['guards']==[51,66,81]
    assert len(row['observations'])==96 and len(row['operations'])==444
    assert row['endpoint15']['step']==15 and row['endpoint20']['step']==20
    assert row['missing_checkpoints']==[]
    assert row['counts']=={k:dict(attempted=n,completed=n) for k,n in m.COUNTS.items()}
    with pytest.raises(ValueError):m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,original)


@pytest.mark.parametrize('step',[5,10,15,16,19,20])
def test_actual_stop_preserved_no_forced_tail(data,tmp_path,step):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    store,ref,seal,_=drive(tmp_path,h,inputs,p,original,stop=('plan',step))
    row=m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,original,permit_fixture=True)['row']
    assert row['status']=='native_stopped' and row['counts']['public_step']['attempted']==10+step
    assert (row['endpoint20'] is None)==(step<20)
    assert (81 in row['guards'])==(step>=16)


def test_81_guard_write_failure_never_dispatches_step16(data,tmp_path):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    store,ref,seal,meter=drive(tmp_path,h,inputs,p,original,fail_guard=True)
    assert meter.counts['public_step']['attempted']==25 and 81 not in meter.guarded
    result=m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,original,permit_fixture=True)
    assert result['row']['status']=='engineering_failed' and result['row']['endpoint20'] is None


@pytest.mark.parametrize('index,expected_public',[(50,15),(65,20)])
def test_original_guard_mismatch_never_reaches_next_action(data,tmp_path,index,expected_public):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    inputs[0]['rows'][0]['observations'][index]['pre']['renderer']['fixed_original_unknown']='tampered'
    store,ref,seal,meter=drive(tmp_path,h,inputs,p,original)
    assert meter.counts['public_step']['attempted']==expected_public
    row=m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,original,permit_fixture=True)['row']
    assert row['status']=='engineering_failed' and row['endpoint20'] is None


def test_actual_guard_fsync_failure_retains_unverified_partial(data,tmp_path):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    store,ref,seal,meter=drive(tmp_path,h,inputs,p,original,fail_fsync=True)
    assert meter.counts['public_step']['attempted']==25 and 81 not in meter.guarded
    assert seal['status']=='transport_failed_NOT_verified'
    r=m.consume_failure(str(store.path),seal['failure_sha256'],store.binding)
    assert r['counts'] is None and r['endpoint20'] is None and r['unknown_final_callbacks']


def test_actual_png_changed_after_capture_blocks_step16(data,tmp_path):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    store,ref,seal,meter=drive(tmp_path,h,inputs,p,original,corrupt_cp15=True)
    assert meter.counts['public_step']['attempted']==25 and 81 not in meter.guarded
    assert seal['status']=='transport_failed_NOT_verified'
    result=m.consume_failure(str(store.path),seal['failure_sha256'],store.binding)
    assert result['counts'] is None and result['endpoint20'] is None


def test_incompressible_96_png_actual_writer_bound(data,tmp_path):
    import numpy as np
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    store=m.Store(str(tmp_path/'random_png'),dict(fixture_only=True),m.limits([original]))
    rng=np.random.default_rng(20261029)
    for i in range(96):
        pixels=rng.integers(0,256,size=(224,224,3),dtype=np.uint8)
        name=f'frame{i:02}.png'; actual=store.frame(name,pixels)
        assert actual==m.sha(str(store.path/name))
        m.old.check_png_stream((store.path/name).read_bytes())
    with pytest.raises(ValueError,match='96 bounded PNG'):store.frame('extra.png',pixels)


@pytest.mark.parametrize('tamper',['png','journal','extra'])
def test_saved_tamper_rejected(data,tmp_path,tamper):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    store,ref,seal,_=drive(tmp_path,h,inputs,p,original)
    if tamper=='png':next(store.path.glob('*.png')).write_bytes(b'bad')
    elif tamper=='journal':(store.path/'journal.jsonl').write_bytes(b'bad')
    else:(store.path/'extra').write_bytes(b'bad')
    with pytest.raises((ValueError,FileNotFoundError)):m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,original,permit_fixture=True)


def test_partial_failure_is_unknown(data,tmp_path):
    h,inputs,p=data;original=old_fixture(tmp_path,h,inputs,p,0)
    store,ref,_,meter=drive(tmp_path,h,inputs,p,original)
    # Simulate an isolated publication failure with already-created new store.
    path=tmp_path/'failure';binding=dict(fixture_only=True)
    s=m.Store(str(path),binding,m.limits([original]));s.emit('run_started',binding_sha256=m.digest(binding))
    counter=m.Meter(s,p['jobs'][0],ref,original);counter.call('constructor',lambda:None)
    seal=m.failed_publication(s,counter,dict(status='engineering_failed',error={'type':'fixture'},cleanup_errors=[]),ValueError('write'))
    r=m.consume_failure(str(path),seal['failure_sha256'],binding)
    assert r['counts'] is None and r['unknown_final_callbacks'] and r['endpoint20'] is None


def test_wrong_authorization_never_constructs(tmp_path):
    with pytest.raises(ValueError,match='authorization'):m.run('/missing','0'*64,str(tmp_path/'run'),'wrong')
    assert not (tmp_path/'run').exists()


def test_full_batch_disk_reservation_before_native_import(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(m,'verify',lambda *_:(dict(fixture_only=False),None,None))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=10*m.old.CAP+m.old.FLOOR-1))
    monkeypatch.setattr(m.old,'native_helpers',lambda _:pytest.fail('no native import before batch budget'))
    with pytest.raises(ValueError,match='full ten-job maximum'):
        m.run('/fixture','0'*64,str(tmp_path/'run'),m.AUTH)
    assert not (tmp_path/'run').exists()


def test_unpublished_start_failures_keep_all_ten_slots(data,tmp_path,monkeypatch):
    _,inputs,old_p=data
    prepared=tmp_path/'prepare.json';prepared.write_text('{}')
    prepare_sha=m.sha(str(prepared));jobs=old_p['jobs']
    p=dict(jobs=jobs,fixture_only=False,authority={},bindings={})
    monkeypatch.setattr(m,'verify',lambda *_:(p,inputs[0],[{} for _ in jobs]))
    monkeypatch.setattr(m.old,'native_helpers',lambda _:object())
    def fail(*_):raise OSError('fixture startup')
    monkeypatch.setattr(m,'native_job',fail)
    root=tmp_path/'batch';seal=m.run(str(prepared),prepare_sha,str(root),m.AUTH)
    out=m.consume(str(root),str(prepared),prepare_sha,seal['sha256'])
    assert out['status']=='saved_partial_transport_NOT_verified' and len(out['jobs'])==10
    assert all(x['counts'] is None and x['endpoint20'] is None for x in out['jobs'])
    batch=m.read(str(root/'batch.json'));batch['P1_admission']=True
    (root/'batch.json').write_bytes(m.raw(batch)+b'\n')
    with pytest.raises(ValueError,match='science flags'):
        m.consume(str(root),str(prepared),prepare_sha,m.sha(str(root/'batch.json')))


def test_batch_all10_writer_to_independent_reader(data,tmp_path,monkeypatch):
    h,inputs,old_p=data
    originals=[old_fixture(tmp_path,h,inputs,old_p,i) for i in range(10)]
    prepared=tmp_path/'prepare.json';prepared.write_text('{}');prepared_sha=m.sha(str(prepared))
    p=dict(jobs=old_p['jobs'],fixture_only=False,authority=old_p['authority'],bindings={})
    monkeypatch.setattr(m,'verify',lambda *_:(p,inputs[0],originals))
    monkeypatch.setattr(m.old,'native_helpers',lambda _:object())
    def run_fixture(p0,w,original,job,output,prep,prep_sha,helper):
        store,_,seal,_=drive(tmp_path,h,inputs,old_p,original,i=job['index'],output=output)
        # Fixture-only binding and saved result are accepted solely by this
        # explicitly scoped test adapter; production consumer remains strict.
        return seal
    monkeypatch.setattr(m,'native_job',run_fixture)
    real_consume=m.consume_job
    def consume_fixture(output,report_sha,completion_sha,binding,ref,original):
        fixture_binding=deepcopy(binding);fixture_binding.update(fixture_only=True,
            prepared='/fixture/prepare.json',prepare_sha256='a'*64)
        return real_consume(output,report_sha,completion_sha,fixture_binding,ref,original,permit_fixture=True)
    monkeypatch.setattr(m,'consume_job',consume_fixture)
    root=tmp_path/'batch';seal=m.run(str(prepared),prepared_sha,str(root),m.AUTH)
    result=m.consume(str(root),str(prepared),prepared_sha,seal['sha256'])
    assert result['fixed_denominator']==10 and len(result['jobs'])==10
    assert all(x['row']['guards']==[51,66,81] and x['event_count']==990 for x in result['jobs'])
