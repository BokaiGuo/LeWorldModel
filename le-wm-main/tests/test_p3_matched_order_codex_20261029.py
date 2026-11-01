"""Synthetic native callback only; no GPU or environment execution."""
from copy import deepcopy
import hashlib
import subprocess
import sys
import pytest
from experiments.spectral_reachability import p3_matched_order_codex_20261029 as m
import agent_p3_fullcommit_live_20261028_v1_tests as fixture


def execute(policy, backend=fixture.RamBackend):
    root,_=fixture.request(); store=fixture.Events()
    result=m.run_unit(root,policy,lambda task,sink:backend(task,sink),store,permit_ram=True)
    return result,store


def test_real_native_api_ram_integration_matched():
    fixed,fs=execute('fixed_order'); dynamic,ds=execute('signal_order')
    root,_=fixture.request()
    assert fixed['status']==dynamic['status']=='complete_measured_axes_only'
    assert [x['selected_horizon'] for x in fixed['blocks']]==[1,3]
    assert sorted(x['selected_horizon'] for x in dynamic['blocks'])==[1,3]
    assert fixed['resource_receipt']['attempted']==dynamic['resource_receipt']['completed']==m.expected()
    assert m.expected()['predict_API']==52 and m.expected()['VJP_directions']==768
    assert m.compare(fixed,dynamic)['matched_measured_axes'] is True
    assert m.compare(fixed,dynamic)['matched_hook_shape_multiset'] is True
    assert m.verify_unit(root,fixed) and m.verify_unit(root,dynamic)
    assert fs.events[-1]['kind']==ds.events[-1]['kind']==m.NAME+'.final'


def test_failed_attempt_is_persisted_not_matched():
    class Broken(fixture.RamBackend):
        def replay(self,*args): raise RuntimeError('fixture error')
    failed,store=execute('fixed_order',Broken)
    assert failed['status']=='failed'
    assert failed['resource_receipt']['attempted']['replay_calls']==1
    assert failed['resource_receipt']['completed']['replay_calls']==0
    assert store.events[-1]['result']==failed
    fixed,_=execute('signal_order')
    assert m.compare(failed,fixed)['matched_measured_axes'] is None


def test_unsupported_decision_does_not_spend_remaining():
    decision=dict(continue_expanding=True,program_type='reconfiguration',extension_type='reconfiguration',horizon=3)
    with pytest.raises(ValueError,match='unsupported'):m.choose('signal_order',decision,[1,3],0)
    assert m.choose('fixed_order',decision,[1,3],0)==1
    assert m.choose('signal_order',decision,[3],1)==3


def test_implicit_ram_forbidden():
    root,_=fixture.request(); store=fixture.Events()
    r=m.run_unit(root,'fixed_order',lambda t,s:fixture.RamBackend(t,s),store)
    assert r['status']=='failed' and not r['blocks']


def test_pair_rejects_identity_and_tampering():
    fixed,_=execute('fixed_order');dynamic,_=execute('signal_order')
    changed=deepcopy(dynamic);changed['root_sha256']='0'*64
    with pytest.raises(ValueError,match='pair identity'):m.compare(fixed,changed)
    changed=deepcopy(dynamic);changed['resource_receipt']['completed']['predict_API']-=1
    with pytest.raises(ValueError):m.compare(fixed,changed)
    root,_=fixture.request();changed=deepcopy(dynamic);changed['blocks'][0]['winner_index']=99
    with pytest.raises(ValueError,match='winner'):m.verify_unit(root,changed)


def test_saved_consumer_recomputes_and_requires_external_pins(tmp_path):
    root,_=fixture.request();fixed,_=execute('fixed_order');dynamic,_=execute('signal_order')
    args=[]
    for name,value in [('root',root),('fixed',fixed),('signal',dynamic)]:
        path=tmp_path/(name+'.json');raw=m.core.canonical(value);path.write_bytes(raw)
        args += ['--'+name,str(path),'--'+name+'-sha',hashlib.sha256(raw).hexdigest()]
    output=tmp_path/'receipt.json'
    cmd=[sys.executable,'-m','experiments.spectral_reachability.p3_matched_order_codex_20261029',*args,'--output',str(output)]
    subprocess.run(cmd,check=True,capture_output=True)
    receipt=m.read_pinned(output,hashlib.sha256(output.read_bytes()).hexdigest())
    assert receipt['accounting']['matched_measured_axes'] and receipt['backend_kinds']==['RAM_FIXTURE','RAM_FIXTURE']
    assert not receipt['scientific_pass'] and not receipt['source_provenance_certified']
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    (tmp_path/'signal.json').write_bytes(b'{}')
    assert subprocess.run(cmd[:-1]+[str(tmp_path/'bad.json')],capture_output=True).returncode!=0
