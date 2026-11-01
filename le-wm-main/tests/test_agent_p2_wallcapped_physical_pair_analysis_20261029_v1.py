"""Saved fixture-only numeric analysis; no NN or native physics workers."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_wallcapped_physical_pair_analysis_20261029_v1 as m
from test_agent_p2_wallcapped_physical_20261029_v1 import execute as ram_execute


@lru_cache(maxsize=3)
def bundle(missing=(),fault=None,at=11):
    r,s,p,_,_=ram_execute(missing=missing,fault=fault,at=at)
    return r,s.events,p,s.files


def rows(missing=(),fault=None,at=11):
    r,events,p,_=bundle(missing,fault,at)
    paid=m.counters(events,r)
    return [m.summarize_slot(i,p['plans'][i],r['rows'][i],events,p,paid['per_slot'][i]) for i in range(4)]


def test_literal_native_metrics_and_not_object_only():
    goal=[0.]*7;s=[3.,4.,6.,8.,np.pi/2,1.,2.]
    x=m.metrics(goal,s)
    assert x['agent_xy']==5 and x['object_xy']==10
    assert x['joint_xy']==np.sqrt(125) and x['velocity_error']==np.sqrt(5)
    assert x['native7_distance']==np.linalg.norm(s) and x['native7_reward']==-np.linalg.norm(s)
    assert x['native7_delta']==s
    assert x['native_success'] is False and x['object_only_success'] is None
    assert m.metrics(goal,[20.,0.,0.,0.,0.,0.,0.])['native_success'] is False
    assert m.metrics(goal,[0.,0.,0.,0.,np.pi/9,0.,0.])['native_success'] is False


def test_native_nonmod_angle_is_preserved_not_replaced_by_wrapped_gate():
    x=m.metrics([0.]*7,[0.,0.,0.,0.,2*np.pi+1.,0.,0.])
    assert x['native_angle_negative'] and x['native_success']
    assert x['angle_wrapped']>np.pi/9


@pytest.mark.parametrize('value',[[True]*7,[float('nan')]*7,[0.]*6])
def test_literal_metrics_reject_bad_vectors(value):
    with pytest.raises(ValueError):m.metrics([0.]*7,value)


def test_all_steps_cps_two_pairs_complete_no_selection():
    x=rows();r,_,p,_=bundle();pair=m.paired(x,p['plans'])
    assert len(x)==4 and len(pair)==2
    assert all(len(row['steps'])==16 and len(row['warmup'])==11 and row['full15'] for row in x)
    assert all([c['step'] for c in row['checkpoints']]==[0,5,10,15] for row in x)
    assert all(a['dispatch_bits_match'] is a['actual_bits_match'] is True for row in x for a in row['actions'])
    assert all(len(z['steps'])==16 and z['terminal15'] is not None for z in pair)
    assert all(row['actual_latent_Q'] is None for row in x)
    for z in pair:
        j,c=x[z['joint_index']],x[z['composition_index']]
        for step in range(16):
            d=z['steps'][step]['metrics']['composition_minus_joint']
            assert d['object_xy']==c['steps'][step]['metrics']['object_xy']-j['steps'][step]['metrics']['object_xy']
    assert sum(t['public_step']['completed'] for t in m.counters(bundle()[1],r)['per_slot'])==100


@pytest.mark.parametrize('missing',[(0,),(1,3),(0,1,2,3)])
def test_missing_exports_keep_all_rows_steps_pairs_null(missing):
    x=rows(missing);_,_,p,_=bundle(missing);pair=m.paired(x,p['plans'])
    assert len(x)==4 and len(pair)==2
    for i in missing:
        assert x[i]['steps']==[None]*16 and x[i]['final_observed'] is None
        assert not any(a['attempted'] for a in x[i]['actions'])
    for z in pair:
        if z['joint_index'] in missing or z['composition_index'] in missing:
            assert z['terminal15'] is None and not z['both_full15']


def test_native_early_stop_no_endpoint_carryforward():
    x=rows(fault='terminal',at=15);_,_,p,_=bundle((), 'terminal',15)
    assert all(z['final_observed']['step']==5 and z['steps'][6:]==[None]*10 for z in x)
    assert all(z['checkpoints'][2]['value'] is None for z in x)
    assert all(z['terminal15'] is None for z in m.paired(x,p['plans']))


def test_one_arm14_other15_common_time_only_and_separate_eventtime():
    # Pure pairing of independently validated RAM rows, not production labels.
    x=deepcopy(rows());stopped=rows(fault='terminal',at=24)
    x[3]=deepcopy(stopped[3]);_,_,p,_=bundle()
    pair=m.paired(x,p['plans'])[1]
    assert pair['joint_index']==3 and pair['composition_index']==2
    assert x[3]['status']=='native_stopped' and x[3]['final_observed']['step']==14
    assert x[2]['final_observed']['step']==15
    assert all(v['metrics'] is not None for v in pair['steps'][:15])
    assert pair['steps'][15]['metrics'] is pair['terminal15'] is None
    assert x[3]['steps'][15] is x[3]['checkpoints'][3]['value'] is None
    assert not pair['both_full15']


@pytest.mark.parametrize('field',['reward','actual_bits','goal','action','drop_frame','cp','endpoint','row_order','source_root'])
def test_numeric_mutations_rejected_without_producer_consumer(field):
    r,events,p,_=deepcopy(bundle());row=r['rows'][0];plan=p['plans'][0]
    local=[e for e in events if e.get('branch')==0]
    t=next(e for e in local if e['kind']=='returned_transition' and e['phase']=='plan')
    if field=='reward':t['reward']+=1.
    elif field=='actual_bits':t['actual_bits']='00'
    elif field=='goal':t['goal_state'][0]+=1.
    elif field=='action':plan['commands'][0][0]=.5
    elif field=='drop_frame':
        e=next(e for e in events if e['kind']=='fresh_observation' and e.get('branch')==0
            and e['receipt']['phase']=='plan' and e['receipt']['step']==1 and e['receipt']['overlay_index']==0)
        events.remove(e)
    elif field=='cp':row['checkpoints'][1]['state'][0]+=1.
    elif field=='endpoint':row['final_state'][0]+=1.
    elif field=='row_order':row['arm']='composition'
    else:p['source_roots'][0]['root_state'][0]+=1.
    with pytest.raises(ValueError):m.summarize_slot(0,plan,row,events,p,m.counters(bundle()[1],bundle()[0])['per_slot'][0])


def test_exact_real_adapter_codec_hashchain_and_empty_record_rejection():
    r,_,_,files=bundle();ev=m.journal(files['journal.jsonl'],r)
    assert len(ev)>2
    payload=files['journal.jsonl'].replace(b'\n',b'\n\n',1);bad=deepcopy(r)
    bad['journal_sha256']=hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError):m.journal(payload,bad)


def disk_fixture(tmp_path):
    r,events,p,files=deepcopy(bundle())
    directory=tmp_path/'physical';directory.mkdir();prepared=tmp_path/'prepared.json'
    prepared.write_bytes(m.canonical(p));psha=m.sha(prepared)
    r['execution_binding'].update(prepared=str(prepared),prepare_sha256=psha)
    events[0]['prepare_sha256']=psha
    previous=None
    for i,e in enumerate(events):
        e.update(index=i,previous_sha256=previous)
        e['event_sha256']=m.digest({k:v for k,v in e.items() if k!='event_sha256'});previous=e['event_sha256']
    files['journal.jsonl']=b''.join(m.canonical(e)+b'\n' for e in events)
    r['journal_chain_tail']=previous;r['journal_sha256']=hashlib.sha256(files['journal.jsonl']).hexdigest()
    files['execution_binding.json']=m.canonical(r['execution_binding'])
    r['execution_binding_sha256']=hashlib.sha256(files['execution_binding.json']).hexdigest()
    files['report.json']=m.canonical(r)
    for name,body in files.items():(directory/name).write_bytes(body)
    rsha=m.sha(directory/'report.json')
    authority=dict(schema=m.PHYSICAL+'.consume',physical_report_sha256=rsha,rows=r['rows'],plans=p['plans'],
        counts=r['counts'],errors=r['errors'],status=r['status'],fixed_denominator=4,known_roots=2,fresh_roots=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,scientific_admission=False,physical_release=False,
        planner_authority=p['planner_authority'],bindings={str(prepared):psha})
    authority['bindings'].update({str(directory/name):m.sha(directory/name) for name in files})
    receipt=tmp_path/'strict.json';receipt.write_bytes(m.canonical(authority))
    return directory,rsha,receipt,m.sha(receipt)


def test_full_saved_disk_analysis_and_exclusive_writer(tmp_path):
    args=disk_fixture(tmp_path);out=m.analyze(*args)
    assert len(out['rows'])==4 and len(out['pairs'])==2 and not out['P2_mechanism']
    assert out['upstream_full_strict_rebuilt_here'] is False
    path=tmp_path/(m.NAME+'_RAM_ONLY.json');written=m.write_new(path,out)
    assert m.read(path,written['sha256'])==out
    with pytest.raises(ValueError):m.write_new(path,out)


@pytest.mark.parametrize('target',['report_sha','receipt_sha','bound_png','bound_prepare','strict_row'])
def test_saved_authority_and_bindings_tampering_failclosed(tmp_path,target):
    directory,rsha,receipt,csha=disk_fixture(tmp_path)
    if target=='report_sha':rsha='0'*64
    elif target=='receipt_sha':csha='0'*64
    elif target=='bound_png':next(directory.glob('*.png')).write_bytes(b'corrupted')
    elif target=='bound_prepare':(tmp_path/'prepared.json').write_bytes(b'{}')
    else:
        a=m.read(receipt);a['rows'][0]['own_returned']=14
        receipt.write_bytes(m.canonical(a));csha=m.sha(receipt)
    with pytest.raises(ValueError):m.analyze(directory,rsha,receipt,csha)


def test_counters_cannot_refund_or_move_operations():
    r,e,_,_=deepcopy(bundle());r['counts']['public_step']['attempted']-=1
    with pytest.raises(ValueError):m.counters(e,r)
    r,e,_,_=deepcopy(bundle());j=next(i for i,v in enumerate(e) if v['kind']=='operation_returned')
    e[j],e[j-1]=e[j-1],e[j]
    with pytest.raises(ValueError):m.counters(e,r)
