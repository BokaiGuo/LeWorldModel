from copy import deepcopy
import hashlib
from pathlib import Path
import pytest

from experiments.spectral_reachability import agent_p2_intersection_physical_analysis_20261029_v2 as m
from test_agent_p2_intersection_physical_20261029_v2 import complete, execute


def receipt(r,p):
    flags=('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness','equal_FLOPs','equal_total_compute','physical_release','independent_confirmation')
    return dict(schema=m.physical.NAME+'.consume',status=r['status'],physical_report_sha256='a'*64,
        fixed_denominator=6,known_roots=2,fresh_roots=0,inherited=0,rows=deepcopy(r['rows']),plans=deepcopy(p['plans']),
        counts=deepcopy(r['counts']),errors=deepcopy(r['errors']),planner_authority=deepcopy(p['planner_authority']),
        new_neural_calls=0,new_physics_calls=0,**{k:False for k in flags})


def test_full6_endpoint_and_exact_operation_census():
    r,s,p,_,_=complete(); q=receipt(r,p); out=m.analyze(r,q,'a'*64,'b'*64)
    assert len(out['rows'])==6 and out['fixed_denominator']==6
    assert all(x['endpoint_step']==15 and len(x['endpoint_metrics']['residual7'])==7 for x in out['rows'])
    assert all(x['native_operations']['public_step']==dict(attempted=25,completed=25,failed=0) for x in out['rows'])
    assert out['endpoint_success_counts']['known']==6
    assert len(out['same_time_pairs'])==18 and all(x['known'] for x in out['same_time_pairs'])
    assert all(len(x['checkpoints'])==3 for x in out['rows'])


def test_actual_early_stop_and_missing_no_padding():
    r,s,p,_,_=execute(fault='truncate',at=15,missing=(2,)); out=m.analyze(r,receipt(r,p),'a'*64,'b'*64)
    assert out['rows'][2]['endpoint_metrics'] is None and out['rows'][2]['native_success'] is None
    assert all(x['endpoint_step']==5 and not x['endpoint_same_nominal_time'] for x in out['rows'] if x['index']!=2)
    assert out['endpoint_success_counts']['fixed_denominator']==6 and out['endpoint_success_counts']['unknown']==1
    assert all(not x['known'] and x['delta_b_minus_a'] is None for x in out['same_time_pairs'] if x['step'] in (10,15))
    assert all(not x['known'] for x in out['same_time_pairs'] if x['root_slot']==0 and 'grasp' in (x['a'],x['b']))


def test_engineering_failure_keeps_unknown():
    r,s,p,_,_=execute(fault='throw',at=11);out=m.analyze(r,receipt(r,p),'a'*64,'b'*64)
    assert len(out['rows'])==6 and out['endpoint_success_counts']['unknown']==6
    assert all(x['native_operations']['public_step']['failed']==1 for x in out['rows'])


@pytest.mark.parametrize('field',['order','count','op_branch','reward','flags','receipt','axis','account','contract'])
def test_reject_inconsistent_input(field):
    orig,s,p,_,_=complete();r=deepcopy(orig);q=receipt(r,p)
    if field=='order':r['rows'].reverse();q['rows'].reverse()
    elif field=='count':r['counts']['physics_step']['completed']-=1;q['counts']=deepcopy(r['counts'])
    elif field=='op_branch':r['actual_operation_receipts'][0]['branch']=9
    elif field=='reward':r['rows'][0]['responses'][-1]['reward']+=1;q['rows']=deepcopy(r['rows'])
    elif field=='flags':q['equal_FLOPs']=True
    elif field=='receipt':q['physical_report_sha256']='c'*64
    elif field=='axis':r['counts'].pop('physics_step');q['counts']=deepcopy(r['counts'])
    elif field=='account':q['planner_authority']['accounts'][0]['arm']='grasp';r['execution_binding']['planner_authority']=deepcopy(q['planner_authority'])
    else:r['contract']['fixed_denominator']=4
    with pytest.raises((ValueError,KeyError)):m.analyze(r,q,'a'*64,'b'*64)


def test_sha_pin_rejects_changed_input(tmp_path):
    p=tmp_path/'input.json'; p.write_bytes(b'{"a":1}')
    assert m.read_bound(str(p),hashlib.sha256(p.read_bytes()).hexdigest(),100)=={'a':1}
    with pytest.raises(ValueError):m.read_bound(str(p),'0'*64,100)


def test_build_reconstructs_pinned_temp_report_and_receipt(tmp_path):
    r,_,p,_,_=complete();q=receipt(r,p)
    report_path=tmp_path/'report.json';report_path.write_bytes(m.raw(r));report_sha=hashlib.sha256(report_path.read_bytes()).hexdigest()
    q['physical_report_sha256']=report_sha;q['bindings']={str(report_path):report_sha}
    receipt_path=tmp_path/'receipt.json';receipt_path.write_bytes(m.raw(q));receipt_sha=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    out=m.build(str(report_path),report_sha,str(receipt_path),receipt_sha)
    assert len(out['rows'])==6 and len(out['same_time_pairs'])==18
    q['bindings'][str(report_path)]='0'*64;receipt_path.write_bytes(m.raw(q))
    with pytest.raises(ValueError,match='authenticates'):
        m.build(str(report_path),report_sha,str(receipt_path),hashlib.sha256(receipt_path.read_bytes()).hexdigest())

