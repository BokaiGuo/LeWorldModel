"""Synthetic ledger tests; no GPU or physical execution."""
import copy
import pytest
from experiments.spectral_reachability import p3_k17_compute_ledger_codex_20261029 as m


def event(resource='predict', shape=None, directions=None):
    e=dict(index=0,scope=dict(mode='adaptive',K=17,phase='signal'),resource=resource,
           shape=[2,3,192] if shape is None and resource=='predict' else shape,
           directions=directions,status='returned',started_ns=1,ended_ns=3)
    e['delta']=m.delta(e)
    return e


def ledger(e):
    d=e['delta']; caps={k: max(10,v) for k,v in d.items()}
    return dict(caps=caps,attempted=copy.copy(d),completed=copy.copy(d),events=[e],wall_ns=4,
                inclusive_operation_wall_ns={e['resource']:2},FLOPs=None,cuda_ns=None,
                backward_engine_calls=None,physics_steps=0)


def test_reconstruct_shape_and_phase():
    r=m.recount(ledger(event()),'adaptive')
    assert (r['completed']['predict_API'],r['completed']['predict_samples'],r['completed']['predict_context_tokens'])==(1,2,6)
    assert r['phases']['signal']['completed']['predict_context_tokens']==6


@pytest.mark.parametrize('field', ['delta','status','index','scope','time','total','unknown'])
def test_corruption_rejected(field):
    r=ledger(event()); e=r['events'][0]
    if field=='delta': e['delta']['predict_API']=3
    if field=='status': e['status']='attempted'
    if field=='index': e['index']=1
    if field=='scope': e['scope']['mode']='fixedH1'
    if field=='time': e['ended_ns']=9
    if field=='total': r['completed']['predict_API']=0
    if field=='unknown': r['FLOPs']=0
    with pytest.raises(ValueError): m.recount(r,'adaptive')


def test_failed_operation_counts_attempted_not_completed():
    e=event(); e['status']='failed'; r=ledger(e); r['completed']=dict.fromkeys(m.AXES,0)
    result=m.recount(r,'adaptive')
    assert result['attempted']['predict_API']==1 and result['completed']['predict_API']==0


def test_blocked_not_dispatched_or_charged():
    e=event('vjp',directions=12); e.update(status='blocked_before_dispatch',ended_ns=None)
    r=ledger(e); r['caps']['VJP_directions']=11
    r['attempted']=dict.fromkeys(m.AXES,0); r['completed']=dict.fromkeys(m.AXES,0); r['inclusive_operation_wall_ns']={}
    assert m.recount(r,'adaptive')['attempted']['VJP_directions']==0
    r['caps']['VJP_directions']=12
    with pytest.raises(ValueError,match='blocked event'): m.recount(r,'adaptive')


@pytest.mark.parametrize('resource,shape,directions', [('predict',[2,4,192],None),('vjp',None,0),('invented',[1,3,2],None),('jacobian',[],None)])
def test_reject_unknown_or_bad_operation(resource,shape,directions):
    with pytest.raises(ValueError): event(resource,shape,directions)


def test_input_anchor_rejects_tampering(tmp_path):
    p=tmp_path/'receipt.json'; p.write_text('{}')
    with pytest.raises(ValueError,match='anchor'): m.pinned(p,'0'*64)
