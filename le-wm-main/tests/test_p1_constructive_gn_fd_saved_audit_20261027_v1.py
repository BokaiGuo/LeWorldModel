"""RAM-only saved trace arithmetic/fault fixtures; no producer/model import."""
from copy import deepcopy
import hashlib
import json
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_gn_fd_saved_audit_20261027_v1 as m


def fixture():
    points=[]; tasks={}; matrices={}
    j=np.sin(np.arange(3840).reshape(192,20)/31)/10
    for k in (0,1):
        context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
        tasks[k]=dict(root_context=context,action_mean=[.1,-.2],action_std=[.7,1.3])
        matrices[k]=j.copy()
        path=[]; previous=context
        for packet in (0,1):
            previous=m.arithmetic.reconstruct_context(tasks[k],previous,np.zeros((5,2)),np.zeros(192)); path.append(previous)
        points.append(dict(point_order=k,root_index=k,root_sha256=None,task_id=str(k),iteration=1,
            at_logits=np.zeros((2,10)).tolist(),at_terminal=np.zeros(192).tolist(),
            saved_context_sha256=[m.digest(c) for c in path],saved_Jv={d['id']:(j@d['vector']).tolist() for d in m.directions()}))
    events=[]; index=0
    for req in m.requests(points):
        events.append(dict(kind='evaluation_attempt',request=req)); previous=tasks[req['root_index']]['root_context']; hashes=[]
        raw=np.asarray(req['commands']).reshape(20)
        for packet in (0,1):
            for resource in ('action_encoder','predict'):
                e=dict(index=index,resource=resource,samples=1,completed=False,evaluation_order=req['order'],packet=packet)
                events.extend([dict(kind='module_attempt',entry=e),dict(kind='module_completed',entry=dict(e,completed=True))]); index+=1
            terminal=(j[:,:10]@raw[:10] if packet==0 else j@raw).astype(np.float32)
            previous=m.arithmetic.reconstruct_context(tasks[req['root_index']],previous,req['commands'][5*packet:5*packet+5],terminal)
            hashes.append(m.digest(previous))
            events.append(dict(kind='native_packet',evaluation_order=req['order'],packet=packet,context_sha256=hashes[-1],terminal=terminal.tolist()))
        events.append(dict(kind='evaluation_complete',evaluation=dict(req,terminal=previous['latent_history'][-1],context_sha256=hashes)))
    return points,tasks,matrices,events


def test_full_ram_ledger_and_independent_FD_denominators():
    points,tasks,matrices,events=fixture(); rows,counts=m.consume(events,points,tasks); a=m.recompute(points,matrices,rows)
    assert len(events)==456 and len(rows)==38 and len(a['comparisons'])==18
    assert counts=={k:dict(attempted=76,completed=76) for k in ('predict','action_encoder')}
    assert max(c['requested_direction_metrics']['relative_l2'] for c in a['comparisons'])<4e-5 # analytic mock only
    s=m.summarize(a['comparisons'])
    assert [r['epsilon'] for r in s['by_epsilon']]==list(m.EPS)
    assert all(r['count']==6 for r in s['by_epsilon'])
    assert all(r['count']==3 for r in s['by_root_and_epsilon'])
    assert all(r['count']==2 for r in s['by_direction_and_epsilon'])
    assert len(m.canonical(s))<30*1024


@pytest.mark.parametrize('damage',('drop','extra','swap','refund','packet','token','nominal','request','completion'))
def test_raw_stream_damage_fails_closed(damage):
    points,tasks,_,events=fixture()
    if damage=='drop': events.pop()
    elif damage=='extra': events.append(deepcopy(events[-1]))
    elif damage=='swap': events[1],events[2]=events[2],events[1]
    elif damage=='refund': events[1]['entry']['index']=1
    elif damage=='packet': events[5]['packet']=1
    elif damage=='token': tasks[0]['action_mean'][0]+=.1
    elif damage=='nominal': points[0]['at_terminal'][0]=1
    elif damage=='request': events[12]['request']['epsilon']=.5
    else: events[2]['entry']['completed']=False
    with pytest.raises(ValueError): m.consume(events,points,tasks)


@pytest.mark.parametrize('damage',('zero','task','point','native'))
def test_fixed_point_protocol(damage):
    points,_,_,_=fixture()
    if damage=='zero': points[0]['at_logits'][0][0]=.1
    elif damage=='task': points[1]['task_id']=points[0]['task_id']
    elif damage=='point': points[0]['iteration']=2
    else: points[0]['root_sha256']='a'*64
    with pytest.raises(ValueError): m.requests(points)


def test_requested_zero_base_addition_not_negative_zero_rewrite():
    points,_,_,_=fixture(); r=m.requests(points)[1]
    assert r['sign']==-1
    assert r['logits'][0][0]<0 and not np.signbit(r['logits'][0][1])


def test_negative_derivative_evidence_and_analysis_tampering():
    points,tasks,matrices,events=fixture(); rows,_=m.consume(events,points,tasks)
    original=m.recompute(points,matrices,rows)
    matrices[0]*=2
    for d in m.directions(): points[0]['saved_Jv'][d['id']]=(matrices[0]@d['vector']).tolist()
    wrong=m.recompute(points,matrices,rows)
    assert wrong['derivative_verdict']=='not_assigned' and wrong['P1_admission'] is False
    assert max(c['effective_direction_metrics']['relative_l2'] for c in wrong['comparisons'])>.4
    with pytest.raises(ValueError): m.arithmetic.numeric_equal(original,wrong)
    edited=deepcopy(original); edited['comparisons'][0]['fd'][0]+=.001
    with pytest.raises(ValueError): m.arithmetic.numeric_equal(edited,original)


def test_effective_float32_direction_and_null_ratios():
    points,tasks,matrices,events=fixture(); rows,_=m.consume(events,points,tasks)
    a=m.recompute(points,matrices,rows); c=a['comparisons'][6]
    assert c['effective_direction_error_l2']>0
    assert np.array_equal(c['effective_float32_Jv'],matrices[0]@np.asarray(c['effective_direction']))
    z=m.arithmetic.error_metrics(np.zeros(192),np.zeros(192))
    assert z['relative_l2'] is z['cosine'] is None
    summary=m.arithmetic.summary([dict(requested_direction_metrics=z,effective_direction_metrics=z)])
    assert summary['effective_relative_l2']['defined']==0 and summary['effective_relative_l2']['undefined']==1


@pytest.mark.parametrize('sha',('',None,'107b958...','a'*63,'G'*64))
def test_external_SHA_required_before_file_access(sha):
    with pytest.raises(ValueError,match='full external SHA256'): m.audit('not-opened',sha)


def test_gzip_duplicate_keys_nonfinite_and_trailing_rejected():
    for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
        with pytest.raises(ValueError): m.decode(raw)
    c=zlib.compressobj(wbits=31); packed=c.compress(b'{}\n')+c.flush()
    assert m.arithmetic.gunzip(packed)==b'{}\n'
    for raw in (packed[:-1],packed+b'x',packed+packed):
        with pytest.raises(ValueError): m.arithmetic.gunzip(raw)


def test_approved_task_export_plain_or_lossless_gzip():
    raw=m.canonical(dict(task='approved',task_id='opaque'))
    c=zlib.compressobj(wbits=31); packed=c.compress(raw)+c.flush()
    assert m.decode_artifact(raw,'task_0.json')==m.decode_artifact(packed,'task_0.json.gz')
    with pytest.raises(ValueError): m.decode_artifact(packed+b'x','task_0.json.gz')


def test_context_shift_is_independently_checkable():
    points,tasks,_,_=fixture(); task=tasks[0]; raw=np.arange(10,dtype=np.float32).reshape(5,2)/10
    c=m.arithmetic.reconstruct_context(task,task['root_context'],raw,np.ones(192,dtype=np.float32))
    expected=(raw.reshape(10)-np.tile(np.array([.1,-.2],np.float32),5))/np.tile(np.array([.7,1.3],np.float32),5)
    assert np.array_equal(c['action_history'][-1],expected)
    assert c['action_history'][0]==task['root_context']['action_history'][1]
    assert c['latent_history'][:2]==task['root_context']['latent_history'][1:]
    assert m.digest(c)==hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest()
