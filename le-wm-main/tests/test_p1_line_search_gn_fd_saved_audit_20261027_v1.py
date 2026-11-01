"""Independent RAM corruption tests and the externally SHA-bound actual FD audit."""
from copy import deepcopy
import gzip
import json
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p1_line_search_gn_fd_saved_audit_20261027_v1 as m


A=np.sin(np.arange(1920).reshape(192,10)/37)/10
B=np.cos(np.arange(1920).reshape(192,10)/41)/10


def context(z,h):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=z.tolist(),action_history=h.tolist())


def path(task,raw):
    z=np.asarray(task['root_context']['latent_history']); h=np.asarray(task['root_context']['action_history'],np.float32)
    out=[]
    for k in (0,1):
        token=(np.asarray(raw,np.float32)[5*k:5*k+5].reshape(10)-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(np.asarray(task['action_std'],np.float32),5)
        terminal=A@token if k==0 else .7*z[-1]+.3*A@h[-1]+B@token
        z=np.vstack((z[1:],terminal.astype(np.float32).astype(float))); h=np.vstack((h[1:],token))
        out.append(context(z,h))
    return out


def fixture():
    tasks={r:dict(root_context=context(np.zeros((3,192)),np.zeros((2,10))),action_mean=[.1,-.2],action_std=[.7,1.3]) for r in (0,1)}
    points=[]; matrices={}
    for r in (0,1):
        for it in (1,2,3,4):
            x=(np.linspace(-.6,.5,20).reshape(2,10)*it/4).astype(np.float32)
            raw=np.tanh(x).reshape(10,2); contexts=path(tasks[r],raw)
            j=np.column_stack((A,B))/np.tile(tasks[r]['action_std'],10)[None,:]*(1-raw.astype(float).reshape(20)**2)[None,:]
            point=dict(point_order=len(points),root_index=r,root_sha256=str(r)*64,task_id='toy'+str(r),iteration=it,
                at_logits=x.astype(float).tolist(),at_terminal=contexts[-1]['latent_history'][-1],saved_context_sha256=[m.digest(c) for c in contexts])
            matrices[len(points)]=j; points.append(point)
    events=[]; index=0
    for req in m.requests(points):
        events.append(dict(kind='evaluation_attempt',request=deepcopy(req))); contexts=path(tasks[req['root_index']],req['commands'])
        for packet,c in enumerate(contexts):
            for resource in ('action_encoder','predict'):
                e=dict(index=index,resource=resource,samples=1,completed=False,evaluation_order=req['order'],packet=packet)
                events.extend([dict(kind='module_attempt',entry=e),dict(kind='module_completed',entry=dict(e,completed=True))]); index+=1
            events.append(dict(kind='native_packet',evaluation_order=req['order'],packet=packet,context_sha256=m.digest(c),terminal=c['latent_history'][-1]))
        events.append(dict(kind='evaluation_complete',evaluation=dict(req,terminal=contexts[-1]['latent_history'][-1],context_sha256=[m.digest(c) for c in contexts])))
    return points,matrices,tasks,events


def test_independent_fixed_request_rounding_no_projection():
    points,_,_,_=fixture(); rows=m.requests(points)
    assert len(rows)==152 and sum(r['mode']=='nominal' for r in rows)==8
    d=m.directions()[0]; first=rows[1]
    x=(np.asarray(points[0]['at_logits'])-1e-3*np.asarray(d['vector']).reshape(2,10)).astype(np.float32)
    assert np.array_equal(first['logits'],x)
    assert np.array_equal(first['commands'],np.tanh(x).reshape(10,2))


def test_full152_context_and608_module_mock_census():
    points,js,tasks,events=fixture(); rows,counts=m.consume(events,points,tasks)
    result=m.recompute(points,js,rows)
    assert len(events)==1824 and len(rows)==152 and len(result['comparisons'])==72
    assert result['nominal_anchors']==8 and len(result['epsilon_stability'])==24
    assert counts=={k:dict(attempted=304,completed=304) for k in ('predict','action_encoder')}
    assert result['scientific_pass_threshold'] is None and result['P1_admission'] is False


@pytest.mark.parametrize('damage',('drop','extra','reorder','raw','normalizer','history_hash','packet','terminal','resources','duplicate_return','bool_count','anchor'))
def test_saved_receipt_corruptions_cannot_disappear(damage):
    points,_,tasks,events=fixture()
    if damage=='drop': events.pop()
    elif damage=='extra': events.append(deepcopy(events[-1]))
    elif damage=='reorder': events[1],events[2]=events[2],events[1]
    elif damage=='raw': events[0]['request']['commands'][0][0]+=.01
    elif damage=='normalizer': tasks[0]['action_std'][0]*=2
    elif damage=='history_hash': events[5]['context_sha256']='0'*64
    elif damage=='packet': events[10]['packet']=0
    elif damage=='terminal': events[5]['terminal'][0]+=.01
    elif damage=='resources': events[1]['entry']['resource']='predict'
    elif damage=='duplicate_return': events[4]=deepcopy(events[2])
    elif damage=='bool_count': events[1]['entry']['samples']=True
    else: points[0]['at_terminal'][0]+=.01
    with pytest.raises(ValueError): m.consume(events,points,tasks)


def test_native_context_normalizer_and_retained_history_independently_rebuilt():
    _,_,tasks,_=fixture(); task=tasks[0]
    raw=np.arange(10,dtype=np.float32).reshape(5,2)/20
    c=m.reconstruct_context(task,task['root_context'],raw,np.ones(192,np.float32))
    expected=(raw.reshape(10)-np.tile(np.array([.1,-.2],np.float32),5))/np.tile(np.array([.7,1.3],np.float32),5)
    assert np.array_equal(c['action_history'][-1],expected)
    assert c['latent_history'][:2]==task['root_context']['latent_history'][1:]
    with pytest.raises(ValueError,match='float32'):
        m.reconstruct_context(task,task['root_context'],raw,np.full(192,.1,dtype=float))


@pytest.mark.parametrize('damage',('missing_tanh','double_tanh','missing_normalizer','detach_history','detach_latent'))
def test_actual_FD_vs_incorrect_J_not_just_raw_row_self_consistency(damage):
    points,js,tasks,events=fixture(); rows,_=m.consume(events,points,tasks)
    bad={k:v.copy() for k,v in js.items()}
    for point in points:
        k=point['point_order']; chain=1-np.tanh(np.asarray(point['at_logits'],np.float32)).astype(float).reshape(20)**2
        if damage=='missing_tanh': bad[k]/=chain
        elif damage=='double_tanh': bad[k]*=chain
        elif damage=='missing_normalizer': bad[k]*=np.tile(tasks[point['root_index']]['action_std'],10)
        elif damage=='detach_history': bad[k][:,:10]*=.7
        else: bad[k][:,:10]*=.3
    correct=m.recompute(points,js,rows)['comparisons']; incorrect=m.recompute(points,bad,rows)['comparisons']
    good=[c['effective_direction_metrics']['absolute_l2'] for c in correct if c['direction']=='fixed20_sign_unit']
    wrong=[c['effective_direction_metrics']['absolute_l2'] for c in incorrect if c['direction']=='fixed20_sign_unit']
    # Mock sensitivity assertion only; NOT a new real-data derivative threshold.
    assert min(wrong)>20*max(good)


def test_null_reference_ratios_stay_unknown_and_all_denominators_retained():
    z=m.error_metrics(np.zeros(192),np.zeros(192))
    assert z['relative_l2'] is z['symmetric_relative_l2'] is z['cosine'] is None
    c=dict(requested_direction_metrics=z,effective_direction_metrics=z)
    s=m.summary([c,c]); assert s['count']==2
    assert s['effective_relative_l2']['undefined']==2 and s['effective_relative_l2']['minimum'] is None


@pytest.mark.parametrize('damage',('truncated','trailing','concatenated','limit'))
def test_gzip_partial_or_extra_stream_never_complete(damage):
    raw=gzip.compress(b'{"kind":"event"}\n',mtime=0)
    if damage=='truncated': raw=raw[:-1]
    elif damage=='trailing': raw+=b'extra'
    elif damage=='concatenated': raw+=raw
    with pytest.raises(ValueError): m.gunzip(raw,limit=1 if damage=='limit' else 1024)


@pytest.mark.parametrize('raw',(b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'))
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError): m.decode(raw)


def test_numeric_vectors_and_scope_mutations_rejected():
    expected=dict(fd=[0.,1.,2.],verdict=None,P1_admission=False,count=72)
    m.numeric_equal(deepcopy(expected),expected)
    for key,value in (('fd',[0.,1.1,2.]),('fd',[0.,1.]),('verdict','passed'),('P1_admission',0),('count',71)):
        bad=deepcopy(expected);bad[key]=value
        with pytest.raises(ValueError): m.numeric_equal(bad,expected)


@pytest.fixture(scope='module')
def actual(): return m.execute()


def test_actual_report_externally_bound_and_all72_recomputed(actual):
    assert actual['input_report_sha256']=='1dd13d4f8ce4ac2bcdee7e8e360ef283962bae11314efb19f017cbeb92d44163'
    assert actual['status']=='saved_fd_arithmetic_verified_not_admitted'
    assert actual['denominators']==dict(points=8,nominal_anchors=8,evaluations=152,central_comparisons=72,
        reconstructed_full_contexts=304,module_attempts=608,module_completions=608,events=1824)
    assert len(actual['comparisons'])==72 and len(actual['by_direction_and_epsilon'])==9
    assert all(s['count']==24 for s in actual['by_epsilon']+actual['by_direction'])
    assert all(s['count']==8 for s in actual['by_direction_and_epsilon'])
    assert actual['AD_forward_predicts_already_inside_inherited156']==16
    assert actual['line_search_plus_FD_predict']==460
    assert actual['derivative_verdict']=='not_assigned' and actual['scientific_pass_threshold'] is None
    assert actual['P1_admission'] is actual['independent_confirmation'] is False
    assert not any(k in sys.modules for k in ('torch','gymnasium','pymunk'))


def test_external_report_hash_mismatch_fails_before_reading_outputs():
    with pytest.raises(ValueError,match='file SHA'):
        m.execute(expected='0'*64)
