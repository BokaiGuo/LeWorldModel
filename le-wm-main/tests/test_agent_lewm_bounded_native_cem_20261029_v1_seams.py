"""Additional RAM regressions; frozen producer/tests/doc remain unchanged."""
import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent_lewm_bounded_native_cem_20261029_v1 import saved, FixtureTensor
from experiments.spectral_reachability import agent_lewm_bounded_native_cem_20261029_v1 as m


def persist_report(store, report):
    (store.path/'report.json').write_bytes(m.raw(report))
    return m.sha(store.path/'report.json')


def reseal_journal(store, report, events):
    """Valid new hashchain: mutations must fail schedule, not integrity checks."""
    tail=None;lines=[]
    for sequence,event in enumerate(events):
        row=dict(sequence=sequence,previous_sha256=tail,**event)
        tail=hashlib.sha256(m.raw(row)).hexdigest();row['event_sha256']=tail
        lines.append(m.raw(row))
    path=store.path/'journal.jsonl';path.write_bytes(b''.join(lines))
    report['journal_tail']=tail
    report['artifacts']['journal.jsonl']=dict(sha256=m.sha(path),file_bytes=path.stat().st_size)
    return persist_report(store,report)


def test_frozen_inputs_and_full_store_roundtrip(saved):
    store,report,_=saved
    assert m.sha(m.__file__)=='d0f249632793fb1ceca98849e7e716b7cd4d382d092a0adc1ba7c8fa99bab47f'
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert [r['root'] for r in receipt['rows']]==[0,1]
    assert all(r['candidate_count']==8 for r in receipt['rows'])
    assert receipt['resources']==m.resources()
    assert not receipt['physical_release'] and not receipt['scientific_admission']


@pytest.mark.parametrize('root',[0,1])
@pytest.mark.parametrize('iteration',[0,1])
@pytest.mark.parametrize('mode',['score_before_seal','missing_seal','update_before_score'])
def test_valid_hashchain_planning_barrier_rejected(saved,root,iteration,mode):
    store,report,_=saved;events=m.expected_events(report['prepare_sha256'],report['status'])
    find=lambda kind:next(i for i,e in enumerate(events) if e['kind']==kind and e.get('root')==root and e.get('iteration')==iteration)
    seal=find('planning_predictions_sealed');score=find('criterion_intent');update=find('cem_update')
    if mode=='missing_seal':events.pop(seal)
    elif mode=='score_before_seal':events[seal],events[score]=events[score],events[seal]
    else:events[score],events[update]=events[update],events[score]
    h=reseal_journal(store,report,events)
    with pytest.raises(ValueError,match='seal-before-criterion orders'):m.consume(store.path,h)


@pytest.mark.parametrize('root',[0,1])
@pytest.mark.parametrize('layout',['whole','segmented'])
@pytest.mark.parametrize('mode',['score_before_all4_seal','missing_layout_return','missing_all4_seal'])
def test_valid_hashchain_all4_barrier_rejected(saved,root,layout,mode):
    store,report,_=saved;events=m.expected_events(report['prepare_sha256'],report['status'])
    seal=next(i for i,e in enumerate(events) if e['kind']=='all_final_predictions_sealed')
    score=next(i for i,e in enumerate(events) if e==dict(kind='criterion_intent',root=root,layout=layout,iteration=None))
    if mode=='score_before_all4_seal':events[seal],events[score]=events[score],events[seal]
    elif mode=='missing_all4_seal':events.pop(seal)
    else:events.remove(dict(kind='layout_return',root=root,layout=layout))
    h=reseal_journal(store,report,events)
    with pytest.raises(ValueError,match='seal-before-criterion orders'):m.consume(store.path,h)


@pytest.mark.parametrize('root',[0,1])
def test_final_reload_second_tanh_rejected_with_valid_new_arrays(saved,root):
    store,report,p=saved;delivery=report['rows'][root]['delivery']
    record=delivery['normalization'];raw=m.old.load_array(store.path,record['raw'],report['artifacts'])
    remapped=np.tanh(raw).astype(np.float32)
    assert not m.exact(raw,remapped)  # witness nonzero, cannot hide idempotence at zero
    slot=p['slots'][root]
    tokens=(remapped.reshape(1,2,10)-np.tile(np.asarray(slot['mean'],np.float32),5))/np.tile(np.asarray(slot['std'],np.float32),5)
    record['raw']=store.array(remapped);record['tokens']=store.array(tokens)
    delivery['inputs']['actions']=store.array(tokens)
    report['artifacts']={k:v for k,v in store.inventory.items() if k!='report.json'}
    # journal was separately registered before report, and all new NPY hashes valid.
    with pytest.raises(ValueError,match='actual supplied raw normalizer input'):
        m.consume(store.path,persist_report(store,report))


def test_extra_final_mapping_event_rejected(saved):
    store,report,_=saved;events=m.expected_events(report['prepare_sha256'],report['status'])
    i=events.index(dict(kind='raw_reload_return',root=0))+1
    events[i:i]=[dict(kind=kind,root=0,iteration=None,final_mapping=True) for kind in ('mapping_intent','mapping_return')]
    with pytest.raises(ValueError,match='seal-before-criterion orders'):
        m.consume(store.path,reseal_journal(store,report,events))


@pytest.mark.parametrize('samples',[4,300])
@pytest.mark.parametrize('final',[False,True])
def test_packet_normalizer_full_indexed_shape(saved,monkeypatch,samples,final):
    store,_,p=saved
    fake=SimpleNamespace(Tensor=FixtureTensor,float32=np.float32,
        isfinite=lambda a:FixtureTensor(np.isfinite(a.a)),
        as_tensor=lambda value,**kw:FixtureTensor(np.asarray(value,dtype=kw['dtype'])))
    monkeypatch.setitem(sys.modules,'torch',fake)
    shape=(1,2,5,2) if final else (1,samples,2,5,2)
    raw=np.linspace(-.99,.99,np.prod(shape),dtype=np.float32).reshape(shape)
    out,record=m.native_normalize(FixtureTensor(raw),p['slots'][0],store,0,None if final else 0)
    assert out.shape==shape[:-2]+(10,)
    expected=np.empty_like(out.a)
    for packet in range(2):
        for step in range(5):
            for axis in range(2):
                expected[...,packet,2*step+axis]=(raw[...,packet,step,axis]-np.float32(p['slots'][0]['mean'][axis]))/np.float32(p['slots'][0]['std'][axis])
    assert m.exact(out.a,expected)
    assert m.exact(m.old.load_array(store.path,record['raw'],store.inventory),raw)
    assert m.exact(m.normalization_check(record,raw,p['slots'][0],lambda r:m.old.load_array(store.path,r,store.inventory)),expected)


def test_runtime_raw_reload_loop_has_no_mapping_call():
    tree=ast.parse(Path(m.__file__).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loops=[n for n in ast.walk(run) if isinstance(n,ast.For) and any(
        isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='load_array' for x in ast.walk(n))]
    assert len(loops)==1
    names=[x.func.id for x in ast.walk(loops[0]) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
    assert 'native_normalize' in names and 'native_map' not in names
