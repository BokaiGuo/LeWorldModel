"""RAM mocks only: no neural, native simulator, prepare or runtime result files."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_mixedmode_zero_time_control_20261028_v1 as m
from test_agent_mixedmode_capability_physical_20261028_v1 import (
    MemoryStore, fixture as parent_fixture, Env, fake_place, file_store)


def h(value):return hashlib.sha256(str(value).encode()).hexdigest()


def fixture():
    _,p,data=parent_fixture()
    package=dict(schema=m.NAME+'.prepare',contract=m.contract(),jobs=m.jobs_for(p),
        probe_digest=m.probe.digest(p),probe_prepared='/RAM/parent_prepare.json',
        probe_prepare_sha256=h('parent prepare'),bindings={str(Path(m.__file__).resolve()):h('own source')})
    return package,p,data


def execute(monkeypatch,*,factory=Env,store=None,started=None):
    package,p,data=fixture();store=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:None)
    r=m.execute(package,p,data,store,make,authorization=m.AUTH,prepared='/RAM/own.json',
        prepare_sha=h('own prepare'),started=started)
    return r,store,envs,package,p


@lru_cache(maxsize=1)
def completed():
    with pytest.MonkeyPatch.context() as mp:return execute(mp)


def test_fixed20_actual_zero5_counts_and_full_raw_history():
    r,s,envs,package,p=completed()
    assert r['status']=='complete',r['errors'][:3]
    assert len(r['rows'])==len(envs)==20
    assert r['counts']=={('internal_physics_step' if k=='physics_step' else k):dict(attempted=v,completed=v)
        for k,v in m.CAPS.items()}
    assert len([k for k in s.blobs if k.endswith('.png')])==1320
    assert len([k for k in s.blobs if k.endswith('.npz')])==400
    for j,row,env in zip(package['jobs'],r['rows'],envs,strict=True):
        commands=np.asarray(env.commands,np.float32)
        assert len(commands)==20 and env.closed
        assert commands[:10].tobytes()==m.moving.warmup_commands().tobytes()
        assert commands[10:15].tobytes()==np.zeros((5,2),dtype='<f4').tobytes()
        assert commands[15:].tobytes()==m.probe.raw(j['probe_commands'],5).tobytes()
        assert row['zero5_executed'] and not row['model_selected_witness']
        assert 'selection_digest' not in row and row['parent_selection_digest']==p['tasks'][j['task']]['plan']['selection_digest']
        assert row['attempted']==row['completed']==5 and len(row['frames'])==6
        assert env.dispatch_flags==[False]*20 and env.with_target is False
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__
    assert m.helper.CAPS['public_step']==700 and m.helper.NAME!=m.NAME


def test_own_trace_payloads_and_hashchain_in_memory():
    r,s,_,package,p=completed();m.validate_trace(package,p,r,s.events)
    m.validate_payloads(package,p,r,s.events,lambda n:bytes(s.blobs[n]))
    tail=None
    for i,e in enumerate(s.events):
        assert e['index']==i and e['previous_sha256']==tail
        assert e['event_sha256']==m.probe.digest({k:v for k,v in e.items() if k!='event_sha256'})
        tail=e['event_sha256']
    assert tail==r['journal_tail'] and len(s.blobs['journal.jsonl'])<m.JOURNAL_CAP
    assert len(m.canonical(r))<m.REPORT_CAP


@pytest.mark.parametrize('field',['schema','denominator','counts','ancestry','context','zero','probe','dispatch','return',
    'root','goal','endpoint','frame','guard','flags'])
def test_strict_trace_rejects_even_resealed_scientific_or_dispatch_changes(field):
    r0,s,_,package,p=completed();r=deepcopy(r0);events=deepcopy(s.events)
    if field=='schema':r['schema']=m.probe.NAME+'.evidence'
    elif field=='denominator':r['rows'].pop()
    elif field=='counts':r['counts']['public_step']['completed']-=1
    elif field=='ancestry':r['rows'][0]['parent_selection_digest']=h('bad')
    elif field=='context':r['rows'][0]['context_digest']=h('bad')
    elif field=='zero':r['rows'][0]['intervention_commands'][0][0]=float(np.float32(.1))
    elif field=='probe':r['rows'][0]['probe_commands'][0][0]=float(np.float32(.01))
    elif field=='dispatch':next(e for e in events if e['kind']=='operation_attempted' and
        e['resource']=='public_step' and e['fields']['phase']=='zero')['fields']['raw_bits']='wrong'
    elif field=='return':next(e for e in events if e['kind']=='raw_return' and e['phase']=='zero')['actual_bits']='wrong'
    elif field=='root':r['rows'][0]['root_state'][0]+=1
    elif field=='goal':r['rows'][0]['goal'][0]+=1
    elif field=='endpoint':r['rows'][0]['end_state'][0]+=1
    elif field=='frame':r['rows'][0]['frames'][0]['pre']['renderer']['action_scale']=1
    elif field=='guard':r['rows'][0]['guards']['task']['primitive_commands_authorized']=5
    else:r['P1_admission']=True
    with pytest.raises((ValueError,KeyError)):m.validate_trace(package,p,r,events)


@pytest.mark.parametrize('field',['PNG','pixel','triplet','warmup','NPZ','framejoin'])
def test_payloads_strict_frame_info_and_original_warmup(field):
    r0,s,_,package,p=completed();r=deepcopy(r0);events=deepcopy(s.events)
    def load(n):
        raw=bytes(s.blobs[n])
        return raw+b'bad' if field=='PNG' and n.endswith('.png') else raw
    if field=='pixel':next(e for e in events if e['kind']=='fresh_observation')['receipt']['pixel_sha256']=h('bad')
    elif field=='triplet':next(e for e in events if e['kind']=='fresh_observation')['receipt']['overlay_index']=1
    elif field=='warmup':p=deepcopy(p);p['warm_traces'][0][0]['receipt']['pixel_sha256']=h('bad')
    elif field=='NPZ':next(e for e in events if e['kind']=='raw_return')['info']['sha256']=h('bad')
    elif field=='framejoin':r['rows'][0]['frames'][0]['file']='different.png'
    with pytest.raises((ValueError,KeyError)):m.validate_payloads(package,p,r,events,load)


def test_parent_ancestry_fixed_all_tasks_not_selection():
    package,p,_=fixture();jobs=package['jobs']
    assert [(j['task'],j['probe_id']) for j in jobs]==[(t,k) for t in range(4) for k,_ in m.probe.PROBES]
    assert all(j['model_selected_witness'] is False and 'selection_digest' not in j for j in jobs)
    changed=deepcopy(p);changed['tasks'][0]['plan']['prefix_commands'][0][0]=.25
    with pytest.raises(ValueError):m.jobs_for(changed)
    # Old-selected zero prefixes are still physically dispatched, never skipped.
    assert sum(np.count_nonzero(m.probe.raw(j['intervention_commands'],5)) for j in jobs)==0


@pytest.mark.parametrize('fault',['terminal','truncate','throw','substitute'])
def test_all20_failed_or_stopped_denominator_no_replacement(monkeypatch,fault):
    class FaultEnv(Env):
        def step(self,c):
            result=super().step(c)
            if fault=='substitute' and len(self.commands)==11:self.latest_action=np.asarray(c,np.float32)+np.float32(.125)
            return result
    r,s,envs,package,p=execute(monkeypatch,factory=lambda:FaultEnv(fault=fault,at=11))
    assert r['status']=='partial' and len(r['rows'])==len(envs)==20
    assert all(e.closed and len(e.commands)==11 for e in envs)
    assert all(row['end_state'] is None for row in r['rows'])
    assert all(row['status']==('native_stopped' if fault in ('terminal','truncate') else 'failed') for row in r['rows'])
    m.validate_trace(package,p,r,s.events)
    assert r['counts']['public_step']['attempted']==220


def test_timeout_full_unstarted20_and_no_simulator(monkeypatch):
    r,s,envs,package,p=execute(monkeypatch,started=time.monotonic()-m.WALL-1)
    assert not envs and r['counts']=={} and len(r['rows'])==20 and r['status']=='partial'
    assert all(row['status']=='unstarted' and row['end_state'] is None for row in r['rows'])
    m.validate_trace(package,p,r,s.events)


def test_meter_attempt_before_callback_and_cleanup_nonphysical_only():
    s=MemoryStore();meter=m.Meter(s);meter.where={'job_digest':h('j')};order=[]
    meter.call('constructor',lambda:order.append(s.events[-1]['kind']))
    assert order==['operation_attempted']
    meter.counts['public_step']['attempted']=400
    with pytest.raises(ValueError):meter.call('public_step',lambda:order.append('bad'))
    with pytest.raises(ValueError):meter.call('reset',lambda:None,cleanup=True)
    meter.fault=True;meter.call('close',lambda:order.append('close'),cleanup=True)
    assert order[-1]=='close'


def test_callback_error_lowerbound_and_close():
    s=MemoryStore();meter=m.Meter(s);meter.where={'job_digest':h('j')}
    with pytest.raises(RuntimeError):meter.call('reset',lambda:(_ for _ in ()).throw(RuntimeError('primary')))
    assert meter.receipt()['reset']==dict(attempted=1,completed=0)
    assert s.events[-1]['kind']=='operation_failed'
    meter.call('close',lambda:None,cleanup=True)


def test_no_authority_no_input_output_or_factory(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('not authorized')
    monkeypatch.setattr(m,'verify',forbidden);monkeypatch.setattr(m,'Store',forbidden)
    with pytest.raises(ValueError):m.run('x','x','x',forbidden,authorization='wrong')


def test_early_source_error_never_masks_first_execution_error(monkeypatch):
    package,p,data=fixture();store=MemoryStore()
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:(_ for _ in ()).throw(FileNotFoundError('LATE')))
    def fail():raise RuntimeError('FIRST')
    r=m.execute(package,p,data,store,fail,authorization=m.AUTH,prepared='/RAM/p',prepare_sha=h('p'))
    assert r['status']=='partial' and r['errors'][0]['message']=='FIRST'
    assert any(e['phase']=='final_source_hash' for e in r['errors']) and len(r['rows'])==20


def test_store_actual_partial_and_append_accounting(monkeypatch):
    # Exercise the exact inherited low-level writer, not a replacement implementation.
    parent,fs=file_store(monkeypatch);s=object.__new__(m.Store);s.__dict__.update(parent.__dict__)
    s.check=lambda *a,**kw:None
    s.write('journal.jsonl',b'abc',append=True);s.write('journal.jsonl',b'def',append=True)
    assert s.used==6
    fs.partial=True
    with pytest.raises(OSError):s.write('partial.bin',b'abcdefgh')
    assert s.used==9 and s.files['partial.bin'][2]==3


def test_cli_help_and_run_authority(capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    output=capsys.readouterr().out
    assert e.value.code==0 and all(k in output for k in ('prepare','verify','run','consume'))
    with pytest.raises(ValueError):m.main(['run','--prepared','x','--prepare-sha','x','--output','x'])


def test_prepare_contract_json_roundtrip_and_size():
    package,_,_=fixture();assert json.loads(m.canonical(package))==package
    assert len(m.canonical(package))<m.PREP_CAP
    assert m.contract()['caps']['reset']==40 and m.contract()['caps']['explicit_render']==1320
    assert m.contract()['time_matched'] and not m.contract()['state_velocity_contact_mediation_isolated']


def test_own_consume_full_RAM(monkeypatch):
    r,s,_,package,p=completed();root=Path('/RAM/control');report_sha=hashlib.sha256(s.blobs['report.json']).hexdigest()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m,'verify',lambda *_:(package,p,None))
        mp.setattr(m.probe,'verify_bindings',lambda _:None)
        mp.setattr(m.probe,'read',lambda path:json.loads(s.blobs[Path(path).name]))
        mp.setattr(m,'sha',lambda path:hashlib.sha256(s.blobs[Path(path).name]).hexdigest())
        mp.setattr(m.helper,'read_events',lambda *_:s.events)
        mp.setattr(Path,'is_symlink',lambda _:False)
        mp.setattr(Path,'is_file',lambda _:True)
        mp.setattr(Path,'iterdir',lambda _:[root/n for n in s.blobs])
        mp.setattr(Path,'stat',lambda path,**kw:SimpleNamespace(st_size=len(s.blobs.get(path.name,b''))))
        mp.setattr(Path,'read_bytes',lambda path:bytes(s.blobs[path.name]))
        out=m.consume(root,report_sha)
    assert out['fixed_denominator']==20 and out['new_physics_calls']==out['new_neural_calls']==0
    assert out['status']=='zero_time_control_responses_verified_not_admitted' and out['P1_admission'] is False


def test_material_assembly_saved_seam_no_output(monkeypatch):
    _,p,data=fixture()
    monkeypatch.setattr(m.probe,'verify',lambda *_:p)
    monkeypatch.setattr(m.adapter,'source_material',lambda:data)
    monkeypatch.setattr(m.adapter,'constructor_contract',lambda:('/RAM/native.py',h('native')))
    monkeypatch.setattr(m,'sha',lambda path:h(str(Path(path).resolve())))
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:None)
    package,_,_=m.material('/RAM/p.json',h('p'))
    assert len(package['jobs'])==20 and package['schema']==m.NAME+'.prepare'
    assert json.loads(m.canonical(package))==package
