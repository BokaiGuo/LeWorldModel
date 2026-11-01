"""Read-only terminal saved regression and RAM tamper tests; no neural/physics."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_fixed_execution_saved_20261028_v1 as m


@lru_cache(maxsize=1)
def saved():
    assert m.sha(m.ACTUAL_REPORT)==m.ACTUAL_REPORT_SHA
    files=m.saved_artifacts(m.ACTUAL_REPORT.parent)
    report=m.decode(files['report.json']);binding=report['execution_binding']
    assert binding['prepare_sha256']==m.ACTUAL_PREPARE_SHA==m.sha(binding['prepared'])
    prepared=m.physical.read(binding['prepared'],m.physical.PREP_CAP)
    assert hashlib.sha256(files['journal.jsonl']).hexdigest()==report['journal_sha256']
    assert hashlib.sha256(files['execution_binding.json']).hexdigest()==report['execution_binding_sha256']
    assert m.decode(files['execution_binding.json'])==binding
    events=[m.decode(line) for line in files['journal.jsonl'].splitlines()]
    return report,events,prepared,files


def boundary(slot=0):
    _,events,_,_=saved()
    frame=next(e['receipt'] for e in events if e['kind']=='fresh_observation' and e['root_slot']==slot and
        e['receipt']['phase']=='warmup' and e['receipt']['step']==0 and e['receipt']['overlay_index']==0)
    recipe=next(e for e in events if e['kind']=='placement_recipe' and e['branch']==slot*2)
    return deepcopy(frame),deepcopy(recipe)


def test_real_frozen_saved_failure_reproduced_then_only_boundary_repaired():
    report,events,p,_=saved()
    assert m.pin_frozen()
    with pytest.raises(ValueError,match='actual placement state'):m.physical.validate_saved(report,events,p)
    assert m.validate_saved(report,events,p)==report['rows']
    assert [r['own_returned'] for r in report['rows']]==[5]*4 and report['errors']==[]
    assert all(report[k] is False for k in m.FLAGS)


@pytest.mark.parametrize('slot',[0,1])
def test_real_requested_vs_observed_native_modulo_and_unwrapped_bits(slot):
    frame,recipe=boundary(slot)
    requested=np.asarray(recipe['requested_state'],np.float64)
    observed=np.asarray(frame['state'],np.float64)
    expected=requested.copy();expected[4] %= (2*np.pi)
    assert observed.tobytes()==expected.tobytes()
    assert observed[[0,1,2,3,5,6]].tobytes()==requested[[0,1,2,3,5,6]].tobytes()
    assert frame['snapshot']['bodies']['block']['unwrapped_angle']==requested[4]
    assert m.validate_initial_observation(frame,recipe)
    if slot==0:
        assert requested[4]==-5.587802913760431 and observed[4]==0.6953823934191554
        assert observed[4]-requested[4]==2*np.pi


@pytest.mark.parametrize('fault',['obs_plus_2pi','obs_minus_2pi','snapshot_plus_2pi','recipe_plus_2pi','bits_plus_2pi',
    'obs_ulp','request_ulp','agent_x','agent_y','block_x','block_y','velocity_x','velocity_y','unwrapped_ulp'])
def test_boundary_no_modulo_equivalence_or_tolerance_for_physical_identity(fault):
    frame,recipe=boundary();block=frame['snapshot']['bodies']['block']
    if fault=='obs_plus_2pi':frame['state'][4]+=2*np.pi
    elif fault=='obs_minus_2pi':frame['state'][4]-=2*np.pi
    elif fault=='snapshot_plus_2pi':block['unwrapped_angle']+=2*np.pi
    elif fault=='recipe_plus_2pi':recipe['requested_state'][4]+=2*np.pi
    elif fault=='bits_plus_2pi':
        bits=np.frombuffer(bytes.fromhex(block['float64_bits']),dtype='<f8').copy();bits[4]+=2*np.pi;block['float64_bits']=bits.tobytes().hex()
    elif fault=='obs_ulp':frame['state'][4]=float(np.nextafter(frame['state'][4],np.inf))
    elif fault=='request_ulp':recipe['requested_state'][4]=float(np.nextafter(recipe['requested_state'][4],np.inf))
    elif fault=='unwrapped_ulp':block['unwrapped_angle']=float(np.nextafter(block['unwrapped_angle'],np.inf))
    else:
        index={'agent_x':0,'agent_y':1,'block_x':2,'block_y':3,'velocity_x':5,'velocity_y':6}[fault]
        frame['state'][index]=float(np.nextafter(frame['state'][index],np.inf))
    with pytest.raises(ValueError):m.validate_initial_observation(frame,recipe)


def reseal(events,report):
    tail=None
    for i,e in enumerate(events):
        e.update(index=i,previous_sha256=tail)
        e['event_sha256']=m.source.digest({k:v for k,v in e.items() if k!='event_sha256'});tail=e['event_sha256']
    report['journal_chain_tail']=tail


@pytest.mark.parametrize('fault',['recipe_2pi','snapshot_2pi','obs_2pi','raw','metric','count','gate','denominator','checkpoint','journal','source'])
def test_real_saved_resealed_tamper_still_rejected(fault):
    original,original_events,p0,_=saved();r,events,p=deepcopy(original),deepcopy(original_events),deepcopy(p0)
    if fault=='recipe_2pi':next(e for e in events if e['kind']=='placement_recipe')['requested_state'][4]+=2*np.pi
    elif fault in ('snapshot_2pi','obs_2pi'):
        e=next(e for e in events if e['kind']=='fresh_observation')
        if fault=='snapshot_2pi':e['receipt']['snapshot']['bodies']['block']['unwrapped_angle']+=2*np.pi
        else:e['receipt']['state'][4]+=2*np.pi
    elif fault=='raw':r['rows'][0]['commands'][0][0]=.5
    elif fault=='metric':r['rows'][0]['responses'][0]['reward']=0
    elif fault=='count':r['counts']['public_step']['completed']-=1
    elif fault=='gate':r['P3_admission']=True
    elif fault=='denominator':r['rows'].pop()
    elif fault=='checkpoint':r['rows'][0]['checkpoints'][0]['pixel_sha256']='1'*64
    elif fault=='journal':events[1]['kind']='not_allowed'
    else:p['source_roots'][0]['goal'][0]+=1
    for e in events:
        if e['kind']=='branch_closed' and e['receipt']['index']<len(r['rows']):e['receipt']=deepcopy(r['rows'][e['receipt']['index']])
    reseal(events,r)
    with pytest.raises((ValueError,KeyError)):m.validate_saved(r,events,p)


def test_copied_validator_has_exactly_the_one_boundary_change():
    path=Path(m.physical.__file__);body=path.read_text()
    old=body[body.index('def validate_saved('):body.index('\ndef consume(')].strip()
    current=Path(m.__file__).read_text();current=current[current.index('def validate_saved('):current.index('\ndef consume(')].strip()
    expected=old.replace("NAME+'.report'","PHYSICAL+'.report'").replace(
        "elif f['phase']=='warmup':require(len(recipe)==1 and f['state']==recipe[0]['requested_state'],'actual placement state')",
        "elif f['phase']=='warmup':\n                require(len(recipe)==1,'one actual placement recipe')\n                validate_initial_observation(f,recipe[0])")
    assert current==expected


def test_real_artifact_bytes_saved_consume_RAM_boundary(monkeypatch):
    report,_,p,files=saved();calls=[]
    monkeypatch.setattr(m,'verify',lambda path,h:(calls.append((path,h)) or (p,{})))
    monkeypatch.setattr(m,'saved_artifacts',lambda _:dict(files))
    out=m.consume(str(m.ACTUAL_REPORT.parent),m.ACTUAL_REPORT_SHA)
    assert calls==[(report['execution_binding']['prepared'],m.ACTUAL_PREPARE_SHA)]
    assert out['physical_report_schema']==m.PHYSICAL+'.report' and out['rows']==report['rows']
    assert out['fixed_denominator']==4 and out['known_roots']==2 and out['new_physics_calls']==out['new_neural_calls']==0
    assert out['schema']==m.NAME+'.consume' and all(out[k] is False for k in m.FLAGS)
    assert all(out['bindings'][str(p)]==h for p,h in zip(m.physical.own_files(),m.FROZEN_SHAS,strict=True))


@pytest.mark.parametrize('fault',['png','extra','binding','journal','report','verify','drift'])
def test_saved_consume_bytes_and_upstream_failures(monkeypatch,fault):
    _,_,p,original=saved();files=dict(original);expected=m.ACTUAL_REPORT_SHA;calls=[]
    def verify(*a):
        if fault=='verify':raise ValueError('real frozen verification rejected')
        return p,{}
    monkeypatch.setattr(m,'verify',verify)
    if fault=='png':files[next(n for n in files if n.endswith('.png'))]=b'bad'
    elif fault=='extra':files['orphan.bin']=b'bad'
    elif fault=='binding':files['execution_binding.json']=b'{}'
    elif fault=='journal':files['journal.jsonl']=files['journal.jsonl'][:-1]
    elif fault=='report':expected='1'*64
    def read(_):
        calls.append(1);result=dict(files)
        if fault=='drift' and len(calls)>1:result['late.bin']=b'bad'
        return result
    monkeypatch.setattr(m,'saved_artifacts',read)
    with pytest.raises(ValueError):m.consume(str(m.ACTUAL_REPORT.parent),expected)


@pytest.mark.parametrize('method',['verify','material'])
def test_real_frozen_entrypoints_delegated_without_globals_patch(monkeypatch,method):
    calls=[];result=object()
    def fn(*a):calls.append(a);return result
    monkeypatch.setattr(m,'frozen_'+method,fn)
    args=('/RAM/prepared','1'*64) if method=='verify' else ({'RAM':'arguments'},)
    assert getattr(m,method)(*args) is result and calls==[args]


@pytest.mark.parametrize('index',[0,1,2])
def test_all_frozen_file_hashes_checked_before_any_real_verification(monkeypatch,index):
    original=m.sha;path=m.physical.own_files()[index]
    monkeypatch.setattr(m,'sha',lambda p:'1'*64 if str(p)==str(path) else original(p))
    monkeypatch.setattr(m,'frozen_verify',lambda *a:pytest.fail('must reject before frozen verify'))
    with pytest.raises(ValueError,match='exact SHA'):m.verify('/missing','1'*64)


def test_cli_readonly_has_no_prepare_run_or_neural_entry(monkeypatch,capsys):
    for command in ('prepare','run','verify'):
        with pytest.raises(SystemExit):m.main([command])
    calls=[]
    monkeypatch.setattr(m,'consume',lambda *a:(calls.append(a) or dict(status=m.RECORDED)))
    assert m.main(['consume','--output',str(m.ACTUAL_REPORT.parent),'--report-sha',m.ACTUAL_REPORT_SHA])==0
    assert calls==[(str(m.ACTUAL_REPORT.parent),m.ACTUAL_REPORT_SHA)]
