"""Only RAM native-array fixtures, fake clocks and temporary-file ABI tests."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p3_equal_cap_parent_20261029_v1 as m

fixtures=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_parent_native_cuda_20261029_v1.py'))
NumpyNative=fixtures['NumpyNative']


def material():
    original=fixtures['material']()
    return dict(schema=m.NAME+'.prepare',tasks=original['tasks'],profile=m.profile(),roster=m.base.roster(),
        precision=m.base.precision_contract(),protocol_sha256=m.PROTOCOL_SHA,
        fixed_denominator=dict(arms=4,requests_per_arm=144,total_requests=576),**m.FLAGS)


class Memory:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


class Clock:
    def __init__(self):self.now=0;self.syncs=0
    def __call__(self):self.now+=1000;return self.now
    def sync(self):self.syncs+=1


def guard():
    return dict(precision=m.base.precision_contract(),peak_bytes=[0,0],output_bytes=0,whole_job_elapsed_seconds=0.)


def run_ram(p=None,fail=False,expire=None):
    p=material() if p is None else p;store=Memory();clock=Clock()
    def check():
        if expire and expire(store.events):clock.now+=3601*10**9
        return guard()
    rows=m.execute(p,lambda t,s:NumpyNative(t,s,fail=fail),store,sync=clock.sync,guard=check,clock=clock,permit_ram=True)
    return p,dict(rows=rows,profile=p['profile'],fixed_denominator=p['fixed_denominator']),store.events,clock


@pytest.fixture(scope='module')
def saved():return run_ram()


def test_common_caps_source_defaults_untouched():
    p=material();m.validate_prepare(p)
    assert m.profile()['resource_caps']['fixed']==m.profile()['resource_caps']['adaptive']==m.parent.default_caps('adaptive')
    assert m.parent.default_caps('fixed')['predict_API']==114
    assert m.profile()['resource_caps']['fixed']['predict_API']==562
    assert m.profile()['resource_caps']['fixed']['jacobian_API']==34
    assert m.profile()['arm_wall_seconds']==3600 and m.profile()['wall_seconds']==18000
    assert m.sha(m.PROTOCOL)==m.PROTOCOL_SHA and m.sha(m.base.__file__)==m.BASE_SHA


def test_complete_four576_original_R_no_padding_and_real_probes(saved):
    p,r,events,clock=saved
    out=m.validate_saved(p,r,events,permit_fixture=True)
    assert len(out)==4 and sum(len(row['result']['jobs']) for row in r['rows'])==576
    assert sum(len(row['timing']['probes']) for row in r['rows'])==clock.syncs
    refused=r['rows'][3]
    assert refused['status']=='refused_unsupported_original_type'
    assert sum(j['status']=='refused_unsupported_original_type' for j in refused['result']['jobs'])==8
    assert refused['released_plan'] is None and refused['plan_available'] is False
    fixed=r['rows'][0]
    assert fixed['result']['resource_receipt']['completed']['predict_API']==114
    assert fixed['result']['resource_receipt']['completed']['jacobian_API']==18
    assert fixed['plan_available'] and fixed['released_plan']==fixed['result']['model_plan']
    assert fixed['result']['resource_receipt']['completed']!=r['rows'][1]['result']['resource_receipt']['completed']
    assert len({row['timing']['elapsed_ns'] for row in r['rows']})>1  # wall is a response, not equality gate


def test_real_native_failure_no_refund_or_later_arms():
    p,r,events,_=run_ram(fail=True)
    out=m.validate_saved(p,r,events,permit_fixture=True)
    assert len(out)==1 and all(x['status']=='blocked_after_runtime_failure' for x in r['rows'][1:])
    rr=r['rows'][0]['result']['resource_receipt']
    assert rr['attempted']['predict_API']==1 and rr['completed']['predict_API']==0
    assert r['rows'][0]['released_plan'] is None


@pytest.mark.parametrize('kind',('predict','jacobian','raw_backward'))
def test_deadline_after_returned_native_work_is_preserved(kind):
    fired=[False]
    def expired(es):
        if fired[0]:return False
        if es and es[-1]['kind']=='arm_event' and es[-1]['payload']['kind']=='resource_return' and es[-1]['payload']['entry']['resource']==kind:
            fired[0]=True;return True
        return False
    p,r,events,_=run_ram(expire=expired)
    assert fired[0] and r['rows'][0]['timing']['failed']
    assert r['rows'][0]['released_plan'] is None
    assert all(x['status']=='blocked_after_runtime_failure' for x in r['rows'][1:])
    m.validate_saved(p,r,events,permit_fixture=True)
    rr=r['rows'][0]['result']['resource_receipt']
    assert any(e['resource']==kind and e['status']=='returned' for e in rr['events'])


def test_core_predispatch_deadline_without_native_call():
    fired=[False]
    def expired(es):
        if not fired[0] and es and es[-1]['kind']=='arm_event' and es[-1]['call'] is None and es[-1]['payload']['kind']=='resource_attempt':
            fired[0]=True;return True
        return False
    p,r,events,_=run_ram(expire=expired)
    m.validate_saved(p,r,events,permit_fixture=True)
    assert r['rows'][0]['result']['resource_receipt']['attempted']['signal_compile_calls']==1
    assert not any(e['kind']=='arm_event' and e['payload']['kind']=='native_call_attempt' for e in events)
    assert r['rows'][0]['result']['backend_kind'] is None


def test_deadline_only_at_final_sync_withholds_complete_core_plan(saved):
    _,r0,_,_=saved
    store=Memory();clock=Clock()
    timer=m.Timer(store,0,clock.sync,guard,clock);timer.probe('arm_start')
    clock.now+=3601*10**9;timer.probe('arm_final',raise_error=False)
    assert timer.failed and timer.receipt()['probes'][-1]['overrun_ns']>0
    row=dict(r0['rows'][0],timing=timer.receipt(),plan_available=False,released_plan=None)
    assert row['result']['model_plan'] is not None and not row['plan_available']
    m.audit_timer(row,store.events,permit_fixture=True)


def test_execute_final_guard_suppresses_real_completed_plan():
    p=material();store=Memory();clock=Clock();fired=[False]
    def check():
        if store.events and store.events[-1]['kind']=='arm_event' and store.events[-1]['payload']['kind']=='resource_return':
            e=store.events[-1]['payload']['entry']
            if e['resource']=='replay' and e['scope'].get('replay_mode')=='segmented' and not fired[0]:
                clock.now+=3601*10**9;fired[0]=True
        return guard()
    rows=m.execute(p,lambda t,s:NumpyNative(t,s),store,sync=clock.sync,guard=check,clock=clock,permit_ram=True)
    assert fired[0] and rows[0]['result']['model_plan'] is not None
    assert rows[0]['timing']['failed'] and rows[0]['released_plan'] is None
    assert all(x['status']=='blocked_after_runtime_failure' for x in rows[1:])
    m.validate_saved(p,dict(rows=rows,profile=p['profile'],fixed_denominator=p['fixed_denominator']),store.events,permit_fixture=True)


def test_start_sync_failure_keeps_rows_and_sync_attempts():
    p=material();store=Memory();clock=Clock()
    def broken():raise RuntimeError('RAM sync failure')
    rows=m.execute(p,lambda t,s:pytest.fail('no backend allowed'),store,sync=broken,guard=guard,clock=clock,permit_ram=True)
    r=dict(rows=rows,profile=p['profile'],fixed_denominator=p['fixed_denominator'])
    m.validate_saved(p,r,store.events,permit_fixture=True)
    assert rows[0]['result'] is None and rows[0]['timing']['synchronization_attempts']==2
    assert rows[0]['timing']['synchronization_returned']==0


def test_guard_failure_after_successful_sync_not_lost():
    store=Memory();clock=Clock()
    def bad():raise RuntimeError('RAM guard failure')
    timer=m.Timer(store,0,clock.sync,bad,clock)
    timer.probe('arm_start',raise_error=False);timer.probe('arm_final',raise_error=False)
    assert timer.receipt()['synchronization_returned']==2 and timer.receipt()['failed']
    row=dict(slot=0,timing=timer.receipt())
    m.audit_timer(row,store.events,permit_fixture=True)


def test_store_real_two_events_to_disk_chain(tmp_path):
    store=m.Store(tmp_path/'saved',m.profile(),free=lambda:10**12)
    store.event(dict(kind='source_before',sha256='a'*64));store.event(dict(kind='source_after',sha256='a'*64))
    tail=store.tail;store.close()
    events=m.base.load_events(store.output,dict(journal_tail=tail))
    assert [e['kind'] for e in events]==['source_before','source_after']
    path=store.output/'events.jsonl';path.write_bytes(path.read_bytes().replace(b'source_before',b'fake___before'))
    with pytest.raises(ValueError):m.base.load_events(store.output,dict(journal_tail=tail))


def test_real_store_partial_execute_chain_reader_validate(tmp_path):
    p=material();store=m.Store(tmp_path/'partial',p['profile'],free=lambda:10**12);clock=Clock()
    rows=m.execute(p,lambda t,s:NumpyNative(t,s,fail=True),store,sync=clock.sync,guard=guard,clock=clock,permit_ram=True)
    report=dict(rows=rows,profile=p['profile'],fixed_denominator=p['fixed_denominator'],journal_tail=store.tail)
    store.put('binding.json',dict(prepared='/RAM/not-production',prepare_sha256='a'*64))
    store.close();store.put('report.json',report)
    loaded=m.signal.read(store.output/'report.json');events=m.base.load_events(store.output,loaded)
    assert m.validate_saved(p,loaded,events,permit_fixture=True)[0]['plan_available'] is False
    assert {f.name for f in store.output.iterdir()}=={'binding.json','events.jsonl','report.json'}


def test_real_store_normal_allowance_loss_is_partial_only(tmp_path):
    p=material();clock=Clock()
    class Exhausted(m.Store):
        def event(self,value):
            if not self.recovery and value.get('kind')=='arm_event' and value['payload']['kind']=='resource_attempt':
                # The original ledger already appended its paid attempt. No
                # record bytes can fit the ordinary partition at this point.
                self.limits=dict(self.limits,output_bytes=self.bytes+self.limits['final_reserve_bytes'])
            return super().event(value)
    store=Exhausted(tmp_path/'exhausted',p['profile'],free=lambda:10**12)
    rows=m.execute(p,lambda t,s:pytest.fail('no native dispatch after lost attempt'),store,
        sync=clock.sync,guard=lambda:(store.check(),guard())[1],clock=clock,permit_ram=True)
    assert store.recovery and rows[0]['released_plan'] is None
    assert all(r['status']=='blocked_after_runtime_failure' for r in rows[1:])
    with pytest.raises(ValueError,match='no dispatch'):
        store.event(dict(kind='arm_attempt',identity=m.base.roster()[1]))
    with pytest.raises(RuntimeError,match='recovery only'):store.check()
    report=dict(rows=rows,profile=p['profile'],fixed_denominator=p['fixed_denominator'],journal_tail=store.tail)
    store.close();store.put('report.json',report)
    loaded=m.signal.read(store.output/'report.json');events=m.base.load_events(store.output,loaded)
    rr=rows[0]['result']['resource_receipt']
    assert rr['attempted']['signal_compile_calls']==1
    assert not any(e['kind']=='arm_event' and e['payload']['kind']=='resource_attempt' for e in events)
    with pytest.raises(ValueError):m.validate_saved(p,loaded,events,permit_fixture=True)
    assert store.bytes<=store.limits['output_bytes']


@pytest.mark.parametrize('damage',('caps','denominator','timer','overrun','guard','late_work','release','unknown'))
def test_saved_mutations_fail_closed(saved,damage):
    p0,r0,es0,_=saved;p=deepcopy(p0)
    # Copy only affected row/event fragments, not hundreds of native arrays.
    r=dict(r0,rows=list(r0['rows']));events=list(es0)
    if damage=='caps':p['profile']['resource_caps']['fixed']['predict_API']=114
    elif damage=='denominator':p['fixed_denominator']['total_requests']=575
    elif damage=='timer':
        i=next(i for i,e in enumerate(events) if e['kind']==m.TIMING and e['stage']=='native_postreturn');events.pop(i)
    elif damage=='overrun':
        r['rows'][0]=deepcopy(r['rows'][0]);r['rows'][0]['timing']['probes'][-1]['overrun_ns']=1
    elif damage=='guard':
        i=next(i for i,e in enumerate(events) if e['kind']==m.TIMING and e['stage']=='native_predispatch');events[i]=dict(events[i],stage='native_postreturn')
    elif damage=='late_work':events.append(next(e for e in events if e['kind']=='arm_event'))
    elif damage=='release':r['rows'][3]=dict(r['rows'][3],plan_available=True,released_plan={'fake':True})
    else:events.append(dict(kind='unknown_equal_cap'))
    with pytest.raises(ValueError):m.validate_saved(p,r,events,permit_fixture=True)


def test_ram_never_accepted_by_production(saved):
    p,r,events,_=saved
    with pytest.raises(ValueError,match='no implicit RAM'):m.validate_saved(p,r,events)


def test_run_requires_authorization_before_weights():
    with pytest.raises(ValueError,match='explicit MAIN'):m.run('absent','f'*64,'absent',authorization='no')


def test_import_cli_no_torch_physics():
    code=f'import sys; from experiments.spectral_reachability import {m.NAME} as m; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; m.main(["--help"])'
    p=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert p.returncode==0 and all(x in p.stdout for x in ('prepare','verify','run','consume'))
