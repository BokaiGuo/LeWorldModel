"""RAM saved arrays/BytesIO PNG only, no real output, model or environment."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from experiments.spectral_reachability import agent_kinodynamic_matched_physical_consume_20261028_v1 as m
from test_agent_kinodynamic_matched_physical_20261028_v1 import execute,fixture,ArrayEnv


def pack(factory=ArrayEnv):
    report,store,_=execute(factory);p,_=fixture();return report,store,p


def test_all_actual_two_rows_and_six_checkpoints():
    r,s,p=pack();cp=m.assemble(r,s.events,p)
    assert len(cp)==6 and [(c['root_slot'],c['checkpoint']['step']) for c in cp]==[(i,n) for i in (0,1) for n in (0,5,10)]
    assert all(c['arm']==m.ARM and len(c['checkpoint']['state'])==7 for c in cp)


@pytest.mark.parametrize('fault',['terminal','truncate','guard','render','substitute','throw'])
def test_partial_does_not_invent_missing_checkpoints(fault):
    r,s,p=pack(lambda:ArrayEnv(fault=fault));cp=m.assemble(r,s.events,p)
    assert len(r['rows'])==2 and all(c['checkpoint']['step']==0 for c in cp)
    assert len(cp)==2
    if fault=='throw':
        assert all(row['own_returned']==0 for row in r['rows'])
        assert all(row['last_observation']['step']==0 for row in r['rows'])


@pytest.mark.parametrize('change',['subset','oldarm','opmissing','opextra','opindex','counter','raw','reward','velocity','terminal','final','checkpoint','goal','PNG','selection'])
def test_saved_crosswire_rejected(change):
    r,s,p=pack();events=deepcopy(s.events)
    if change=='subset':r['rows'].pop()
    elif change=='oldarm':r['rows'][0]['arm']='CEM'
    elif change in ('opmissing','opextra','opindex'):
        i=next(i for i,e in enumerate(events) if e['kind']=='operation_returned')
        if change=='opmissing':events.pop(i)
        elif change=='opextra':events.insert(i,deepcopy(events[i]))
        else:events[i]['receipt']['operation_index']=999
    elif change=='counter':r['counts']['public_step']['completed']-=1
    elif change=='raw':
        i=next(i for i,e in enumerate(events) if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step')
        events[i]['receipt']['raw_action'][0]+=.01
    elif change in ('reward','velocity','terminal'):
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan')
        if change=='reward':e['reward']+=1.
        elif change=='velocity':e['state'][-1]+=100.
        else:e['terminated']=not e['terminated']
    elif change=='final':r['rows'][0]['final_state'][-1]+=1.
    elif change=='checkpoint':r['rows'][0]['checkpoints'][1]['state'][0]+=.01
    elif change=='goal':r['rows'][0]['goal'][0]+=1.
    elif change=='PNG':r['rows'][0]['PNG']-=1
    else:r['rows'][0]['selection_digest']='f'*64
    with pytest.raises((ValueError,KeyError)):m.assemble(r,events,p)


def test_native_full7_velocity_and_angle_semantics():
    g=np.zeros(7);state=np.zeros(7);state[-1]=3.;state[-2]=4.
    label=m.metric(g,state)
    assert label['native_distance']==5. and label['native_success']
    state[4]=2*np.pi+1.;label=m.metric(g,state)
    assert label['negative_native_angle'] and label['native_success'] and not label['wrapped_allxy_success']
    assert label==m.kernel.metric(g,state)
    with pytest.raises(ValueError):m.metric(np.zeros(5),state)
    with pytest.raises(ValueError):m.metric(g,np.zeros(5))


def test_physical_seals_remain_frozen():
    for f,h in m.physical_seals().items():assert m.sha(f)==h


def consume_memory(monkeypatch,corrupt=None):
    r,s,p=pack();root=Path('/RAM_ONLY_SAVED_BEAM')
    r['prepared']='/RAM_ONLY_PREPARED.json';r['prepare_sha256']='a'*64
    # execute fixture already used these same execution identity bytes.
    files={}
    for name,value in s.files.items():
        if name=='report.json':continue
        if isinstance(value,np.ndarray):
            b=io.BytesIO();Image.fromarray(value).save(b,format='PNG');files[name]=b.getvalue()
    events=deepcopy(s.events)
    events[0]['execution_binding']=dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'])
    for e in events:
        if e['kind']=='fresh_observation':
            c=e['receipt'];c['file_sha256']=hashlib.sha256(files[c['png']]).hexdigest()
    for index,row in enumerate(r['rows']):
        for c in row['checkpoints']:c['file_sha256']=hashlib.sha256(files[c['png']]).hexdigest()
        if 'last_observation' in row:
            c=row['last_observation'];c['file_sha256']=hashlib.sha256(files[c['png']]).hexdigest()
        next(e for e in events if e['kind']=='branch_closed' and e['receipt']['branch_index']==index)['receipt']=deepcopy(row)
    previous=None
    for e in events:
        e['previous_sha256']=previous;e.pop('event_sha256',None);e['event_sha256']=m.source.digest(e);previous=e['event_sha256']
    r['journal_chain_tail']=previous
    files['journal.jsonl']=b''.join(m.physical.raw(e).rstrip(b'\n')+b'\n' for e in events)
    r['journal_sha256']=hashlib.sha256(files['journal.jsonl']).hexdigest()
    files['execution_binding.json']=m.physical.raw(dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']))
    files['report.json']=m.physical.raw(r);expected=hashlib.sha256(files['report.json']).hexdigest()
    if corrupt=='png':files['b0_plan_10.png']+=b'corrupt'
    elif corrupt=='foreign':files['foreign.json']=b'{}'
    elif corrupt=='chain':files['journal.jsonl']=files['journal.jsonl'].replace(b'run_started',b'run_altered',1)
    original_sha=m.sha;original_bytes=Path.read_bytes;original_file=Path.is_file;original_link=Path.is_symlink
    original_stat=Path.stat;original_iter=Path.iterdir;image_open=Image.open
    def fake_file(path):return path.parent==root
    monkeypatch.setattr(m,'verify',lambda *a:p)
    # verify() is an explicit mocked boundary here; its virtual prepare seal
    # must also remain available to the consumer's final closure recheck.
    monkeypatch.setattr(m,'sha',lambda path:'a'*64 if str(path)==r['prepared'] else
        hashlib.sha256(files[Path(path).name]).hexdigest() if fake_file(Path(path)) else original_sha(path))
    monkeypatch.setattr(m,'read',lambda path,*a:json.loads(files[Path(path).name]) if fake_file(Path(path)) else m.physical.read(path,*a))
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(root/n for n in files) if path==root else original_iter(path))
    monkeypatch.setattr(Path,'read_bytes',lambda path:files[path.name] if fake_file(path) else original_bytes(path))
    monkeypatch.setattr(Path,'is_file',lambda path:True if fake_file(path) else original_file(path))
    monkeypatch.setattr(Path,'is_symlink',lambda path:False if fake_file(path) else original_link(path))
    monkeypatch.setattr(Path,'stat',lambda path,*a,**kw:NS(st_size=len(files[path.name])) if fake_file(path) else original_stat(path,*a,**kw))
    monkeypatch.setattr(m,'absolute',lambda value:value)  # RAM path, no actual inode/symlink resolution.
    monkeypatch.setattr(Image,'open',lambda path,*a,**kw:image_open(io.BytesIO(files[Path(path).name])) if isinstance(path,Path) and fake_file(path) else image_open(path,*a,**kw))
    return m.consume(str(root),expected)


def test_entire_consume_bytesio_png_and_chain(monkeypatch):
    r=consume_memory(monkeypatch)
    assert r['fixed_branch_denominator']==2 and r['checkpoint_complete'] and len(r['checkpoints'])==6
    assert len(r['consumer_sources'])==3 and not r['P1_admission']


@pytest.mark.parametrize('change',['png','foreign','chain'])
def test_consume_file_corruption_rejected(monkeypatch,change):
    with pytest.raises(ValueError):consume_memory(monkeypatch,change)
