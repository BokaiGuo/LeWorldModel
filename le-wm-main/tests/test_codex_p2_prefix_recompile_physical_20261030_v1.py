from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_prefix_recompile_physical_20261030_v1 as h
from test_agent_p2_full30_control15_feedback_physical_20261029_v1 import fixture
from test_agent_allmode_physical_20261028_v1 import Env, fake_place, reseal


class MemoryStore(h.Store):
    def __init__(self):
        self.path=Path("/RAM_ONLY");self.files={};self.events=[];self.prefix_events={}
        self.previous=None;self.index=0;self.sizes={};self.bytes=0;self.finalizing=False

    def check(self,size=4096,final=False):
        h.require(self.bytes+size<=h.CAP-(0 if final or self.finalizing else h.FINAL),'RAM cap')

    def write(self,name,value,append=False,final=False):
        self.check(len(value),final)
        h.require(append or name not in self.files,'exclusive')
        self.files[name]=self.files.get(name,b'')+value if append else value
        self.bytes+=len(value);self.sizes[name]=len(self.files[name])
        return hashlib.sha256(self.files[name]).hexdigest()

    def emit(self,kind,**fields):
        start=len(self.files.get('journal.jsonl',b''))
        super().emit(kind,**fields)
        self.events.append(h.native.decode(self.files['journal.jsonl'][start:]))


def synthetic(fault=None,at=11,mismatch=False):
    upstream,data,_,_=fixture()
    parent=upstream['plans'][1]
    bits=np.asarray(parent['commands'][:25],dtype='<f4').tobytes()
    entries=[]
    for arm in ('joint','prefix_chain'):
        entries.append(dict(base=0,arm=arm,task_id=parent['task_id'],root=parent['root_slot'],
            prefix_sha256=hashlib.sha256(bits[:120]).hexdigest(),horizon=25,
            winner_suffix_sha256=hashlib.sha256(bits[120:]).hexdigest(),
            winner_suffix_float32_hex=bits[120:].hex(),full25_raw_float32_hex=bits.hex(),
            winner_index=0,program_ids=['a','b']))
    b=dict(schema=h.program_binding.NAME+'.binding',base=0,plans=entries,
        observed_context=deepcopy(upstream['observed_context']),
        prefix_reference=deepcopy(upstream['prefix_reference']),original_bindings=deepcopy(upstream['bindings']),
        physical_ready=False,scientific_pass=False)
    p=h.h25_assemble(upstream,b); store=MemoryStore(); envs=[]
    if mismatch:
        event=next(e for e in p['prefix_reference'] if e['kind']=='fresh_observation'
            and e['receipt']['phase']=='plan' and e['receipt']['step']==15)
        event['receipt']['pixel_sha256']='bad'
        p['program_binding']['prefix_reference']=deepcopy(p['prefix_reference'])
    def factory():
        env=Env(fault=fault,at=at);envs.append(env);return env
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(h,'place',fake_place);mp.setattr(h,'validate_env',lambda _:None)
        mp.setattr(h,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        result=h.execute(p,data,store,factory,authorization=h.AUTH,
            prepared='/RAM/prepare.json',prepare_sha256='a'*64)
    return result,store,p,envs


def test_complete_and_tampered_guard():
    report,store,p,envs=synthetic()
    assert report['status']==h.RECORDED,report['errors']
    assert all(row['own_returned']==25 and row['feedback_guard'] for row in report['rows'])
    assert len(h.validate_saved(report,store.events,p))==2
    assert all(e.closed for e in envs)
    events=deepcopy(store.events);bad=deepcopy(report)
    next(e for e in events if e['kind']=='feedback_prefix_guard')['guard']['new_witness_sha256']='bad'
    reseal(events,bad)
    with pytest.raises(ValueError):
        h.validate_saved(bad,events,p)


def test_stop_and_prefix_mismatch_never_pad():
    report,store,p,envs=synthetic(fault='truncate',at=20)
    assert all(x['own_returned']==10 and x['status']=='native_stopped' for x in report['rows'])
    assert len(h.validate_saved(report,store.events,p))==2
    assert all(len(e.commands)==20 and e.closed for e in envs)
    report,_,_,envs=synthetic(mismatch=True)
    assert report['status']=='partial_or_failed'
    assert all(len(e.commands)==25 and e.closed for e in envs)
    assert all(x['feedback_guard'] is None for x in report['rows'])


def test_saved_consumer_rejects_png_tamper(monkeypatch):
    _,store,p,_=synthetic()
    files=deepcopy(store.files)
    monkeypatch.setattr(h,'saved_artifacts',lambda _:deepcopy(files))
    monkeypatch.setattr(h,'verify',lambda *_:(deepcopy(p),None))
    report_sha=hashlib.sha256(files['report.json']).hexdigest()
    completion_sha=hashlib.sha256(files['completion.json']).hexdigest()
    receipt=h.consume('/RAM/physical',report_sha,completion_sha)
    assert receipt['saved_PNG_verified']==222
    png=next(name for name in files if name.endswith('.png'))
    files[png]=b'tampered'
    with pytest.raises(ValueError):
        h.consume('/RAM/physical',report_sha,completion_sha)


def test_h25_literal_shape_and_domain():
    a=np.zeros((25,2),dtype=np.float32).astype(float).tolist()
    assert h.raw25(a).shape==(25,2)
    for bad in (a+[[0.,0.]]*5, a[:-1], [[2.,0.]]+a[1:], [[float('nan'),0.]]+a[1:]):
        with pytest.raises(Exception):
            h.raw25(bad)


@pytest.mark.parametrize('base',range(4))
def test_real_material_reconstructs_and_rejects_tampering(base):
    p,_=h.material(base)
    h.validate_execution(p)
    assert [x['policy'] for x in p['plans']]==['joint','prefix_chain']
    assert all(len(bytes.fromhex(x['raw_float32_hex']))==200 for x in p['plans'])
    bad=deepcopy(p)
    bad['plans'][0]['commands'][0][0]=0.123
    bad['plans'][0]['selection_digest']=h.digest({k:v for k,v in bad['plans'][0].items() if k!='selection_digest'})
    bad['planner_authority']['plans_sha256']=h.digest(bad['plans'])
    with pytest.raises(Exception):
        h.validate_execution(bad)


@pytest.mark.parametrize('base',range(4))
def test_actual_winner_packets_and_resealed_identity(base):
    p,_=h.material(base)
    for i,plan in enumerate(p['plans']):
        branch=json.loads((h.program_binding.RUN/f'branch_{2*base+i}.json').read_bytes())
        winner=branch['winner']
        suffix=(winner['candidate']['raw_float32_hex'] if i==0 else
                winner['first_raw_float32_hex']+winner['second_raw_float32_hex'])
        assert bytes.fromhex(plan['raw_float32_hex'])[120:].hex()==suffix
        assert plan['new_model_plan']['winner_index']==(
            winner['candidate']['index'] if i==0 else winner['index'])
    bad=deepcopy(p)
    bad['plans'][1]['new_model_plan']['winner_index']+=1
    w=bad['plans'][1]['new_model_plan']
    w['new_witness_sha256']=h.planner.digest({key:value for key,value in w.items() if key!='new_witness_sha256'})
    bad['plans'][1]['selection_digest']=h.digest({key:value for key,value in bad['plans'][1].items() if key!='selection_digest'})
    bad['planner_authority']['plans_sha256']=h.digest(bad['plans'])
    with pytest.raises(ValueError):
        h.validate_execution(bad)
