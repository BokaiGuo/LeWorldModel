"""CPU-only host wiring and failure tests; no model, CUDA, or physical run."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
from experiments.spectral_reachability import p3_matched_order_host_codex_20261029 as h
import agent_p3_fullcommit_live_20261028_v1_tests as fixture


class Memory:
    def __init__(self): self.events=[]
    def event(self,value): self.events.append(deepcopy(value))


class FailNative:
    kind='RAM_FIXTURE'
    def signal(self,req,ledger): raise RuntimeError('synthetic native failure')


def test_fixed_roster_partial_and_failed_attempts():
    root,_=fixture.request();store=Memory()
    rows=h.execute(dict(tasks=[root,root]),lambda t,s:FailNative(),store,permit_ram=True)
    assert len(rows)==4 and rows[0]['status']=='failed'
    assert [r['status'] for r in rows[1:]]==['blocked_after_failure']*3
    ledger=rows[0]['result']['resource_receipt']
    assert ledger['attempted']['signal_compile_calls']==1 and ledger['completed']['signal_compile_calls']==0
    assert [e['kind'] for e in store.events if e['kind']!='arm_event']==['denominator','arm_attempt','arm_return']
    assert next(e for e in store.events if e['kind']=='arm_return')['row']==rows[0]


def test_backend_without_fixture_permission_fails_closed():
    root,_=fixture.request();store=Memory()
    rows=h.execute(dict(tasks=[root,root]),lambda t,s:FailNative(),store)
    assert rows[0]['status']=='failed' and rows[0]['result']['resource_receipt']['attempted']['signal_compile_calls']==0


def test_consumer_rejects_ram_and_partial():
    root,_=fixture.request();store=Memory()
    rows=h.execute(dict(tasks=[root,root]),lambda t,s:FailNative(),store,permit_ram=True)
    with pytest.raises(ValueError,match='complete native unit'):
        h.audit_arm(root,rows[0],[e for e in store.events if e['kind']=='arm_event'])


def test_guard_refuses_before_host_import():
    with pytest.raises(ValueError,match='authorization'):
        h.run('not-read','0'*64,'unused',False)
    proc=subprocess.run([sys.executable,'-c',
        'import sys;import experiments.spectral_reachability.p3_matched_order_host_codex_20261029;assert "torch" not in sys.modules'],capture_output=True)
    assert proc.returncode==0,proc.stderr.decode()


def test_durable_store_bound_and_failure_record(tmp_path):
    limits=dict(h.limits(),output_bytes=50000,final_reserve_bytes=10000,disk_floor_bytes=0)
    store=h.signal.Store(tmp_path/'capture',limits,free=lambda:10**9)
    store.event(dict(kind='begin'))
    with pytest.raises(ValueError,match='output engineering cap'):store.event(dict(large='x'*50000))
    store.close()
    events,tail=h.signal.journal(tmp_path/'capture'/'events.jsonl')
    assert events==[dict(kind='begin')] and tail==store.tail


def test_prepare_and_tampering_read_only():
    path=h.ROOT/'results/p3_matched_order_host_codex_20261029_prepare_MAIN03.json'
    if not path.exists():pytest.skip('real CPU-only prepare not available')
    value=h.verify_prepare(path,'85e393983b1437630bd74243afe17c6764aff9d8505c5dd9d44d0603bc0066a4')
    assert len(value['tasks'])==2 and len(value['roster'])==4
    assert value['per_unit_cap']['predict_API']==52 and not value['equal_FLOPs']
    with pytest.raises(ValueError,match='input SHA'):h.verify_prepare(path,'0'*64)


def test_saved_native_auditor_on_synthetic_transcripts(monkeypatch):
    # HOST_CUDA label is synthetic in this fixture, not device evidence. Buffer
    # only so the fixture can populate its invented derivative arrays first.
    class Buffered:
        kind='HOST_CUDA'
        def __init__(self,task,sink):self.task=task;self.sink=sink
        def invoke(self,name,*args):
            buffers=fixture.Events();backend=fixture.RamBackend(self.task,buffers)
            out=getattr(backend,name)(*args)
            for event in buffers.events:self.sink.event(event)
            return out
        def signal(self,req,ledger):return self.invoke('signal',req,ledger)
        def replay(self,context,actions,segmented,ledger):return self.invoke('replay',context,actions,segmented,ledger)
    root,_=fixture.request();store=Memory()
    # Only one pair is needed for unit-level native transcript verification.
    sink=h.prior.ArmSink(store,0)
    result=h.design.run_unit(root,'fixed_order',lambda t,s:h.prior.RecordingBackend(Buffered(t,s),t,s,lambda:None),sink)
    assert result['status']=='complete_measured_axes_only'
    row=dict(slot=0,root=0,policy='fixed_order',status=result['status'],result=result)
    wrapped=store.events
    assert h.audit_arm(root,row,wrapped)['native_calls']>0
    corrupted=deepcopy(wrapped)
    victim=next(e for e in corrupted if e['payload']['kind']=='native_call_return')
    victim['payload']['result']['nominal'][0]+=1
    with pytest.raises(ValueError):h.audit_arm(root,row,corrupted)


def test_complete_consumer_synthetic_integration(tmp_path,monkeypatch):
    import json
    class Buffered:
        kind='HOST_CUDA'  # synthetic label; never a real CUDA certificate
        def __init__(self,task,sink):self.task=task;self.sink=sink
        def invoke(self,name,*args):
            buffers=fixture.Events();backend=fixture.RamBackend(self.task,buffers)
            out=getattr(backend,name)(*args)
            for event in buffers.events:self.sink.event(event)
            return out
        def signal(self,req,ledger):return self.invoke('signal',req,ledger)
        def replay(self,context,actions,segmented,ledger):return self.invoke('replay',context,actions,segmented,ledger)
    roots=[fixture.request(slot=i)[0] for i in (0,1)]
    p=dict(tasks=roots,sources={},precision=h.prior.precision_contract(),limits=h.limits(),expected_model_hash='a'*64)
    store=Memory();model=dict(original_state_dict='a'*64);classes=[]
    store.event(dict(kind='source_before',sha256=h.core.digest(p['sources'])))
    store.event(dict(kind='model_before',hashes=model,classes=classes))
    store.event(dict(kind='precision_before',value=p['precision']))
    rows=h.execute(p,lambda t,s:Buffered(t,s),store)
    assert all(r['status']=='complete_measured_axes_only' for r in rows)
    store.event(dict(kind='precision_after',value=p['precision']))
    store.event(dict(kind='model_after',hashes=model,classes=classes))
    store.event(dict(kind='source_after',sha256=h.core.digest(p['sources'])))
    prepared=str(tmp_path/'prepare.json');prep_sha='b'*64
    (tmp_path/'binding.json').write_bytes(h.core.canonical(dict(prepared=prepared,prepare_sha256=prep_sha)))
    (tmp_path/'events.jsonl').write_bytes(b'synthetic fixture')
    report=dict(schema=h.NAME+'.report',status='complete_engineering_only',errors=[],prepared=prepared,prepare_sha256=prep_sha,
        rows=rows,sources_verified=True,model_before=model,model_after=model,classes_before=classes,classes_after=classes,
        precision=[p['precision']]*2,wall_seconds=1,host=dict(device='cuda:0',allocation_sum=4.0,visible_nvidia_nodes=['synthetic']),
        CUDA_peak_bytes=[1,1],journal_tail='fixture',artifact_sha256={n:h.sha(tmp_path/n) for n in ('binding.json','events.jsonl')})
    path=tmp_path/'report.json';path.write_bytes(h.core.canonical(report))
    monkeypatch.setattr(h,'verify_prepare',lambda *_:p)
    monkeypatch.setattr(h.signal,'journal',lambda _:(store.events,'fixture'))
    result=h.consume(tmp_path,h.sha(path),prepared,prep_sha)
    assert result['status']=='saved_native_IO_engineering_only' and not result['scientific_pass']
    assert all(pair['matched_hook_shape_multiset'] for pair in result['pairs'])
    damaged=deepcopy(store.events);damaged[0]['sha256']='0'*64
    monkeypatch.setattr(h.signal,'journal',lambda _:(damaged,'fixture'))
    with pytest.raises(ValueError,match='top journal'):h.consume(tmp_path,h.sha(path),prepared,prep_sha)
