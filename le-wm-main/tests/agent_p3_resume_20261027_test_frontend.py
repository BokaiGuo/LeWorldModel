"""RAM metadata and interface tests; no model, CUDA, physics or real prepare."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_resume_20261027_frontend as m
from experiments.spectral_reachability import chat_p3_20261027_scheduler as rule


def h(x): return m.digest(str(x))


def fixture(weights=(2.,1.)):
    banks=[[dict(index=i,kind='SYNTHETIC_RAM',commands=np.full((10,2),i/64,np.float32).tolist()) for i in range(45)] for _ in range(2)]
    slots=[dict(source_slot=s,**{k:h((s,k)) for k in m.ID_KEYS[1:]}) for s in (0,1)]
    p=dict(slots=slots)
    sealed=dict(weights=list(weights),plans={})
    for a in m.ARMS:
        blocks=rule.allocation(a) if a in ('U','N') else rule.allocation(a,list(weights))
        jobs=rule.queue(blocks)
        sealed['plans'][a]=dict(blocks=blocks,jobs=jobs,candidate_sha256=[m.digest(banks[j['slot']][j['index']]) for j in jobs])
    queue=m.compile_queue(p,banks,sealed,rule)
    return dict(queue=queue,fixed_denominator=dict(roots=2,arms=4,branches=8,search_candidates=240),
                historical_encoder=dict(encoder='SYNTHETIC'),historical_anchor_receipts=[],bindings_sha256={}),p,banks,sealed


def full(weights=(2.,1.)):
    p,base,banks,sealed=fixture(weights); events=[]; shapes=[]; plans=[]
    result=dict(signals=[dict(signal=dict(w=x)) for x in weights],allocations={a:p['queue']['queues'][a]['blocks'] for a in m.ARMS})
    for a in m.ARMS:
        q=p['queue']['queues'][a]
        events.append(dict(kind='queue_frozen',arm=a,blocks=q['blocks'],jobs=[dict(slot=j['slot'],index=j['index']) for j in q['requests']]))
        for n,j in enumerate(q['requests']):
            events.append(dict(kind='candidate_request',scope=[a,j['slot']],paid=n+1,candidate=dict(banks[j['slot']][j['index']],status='attempted')))
        for s in (0,1): plans.append(dict(**base['slots'][s],arm=a,candidate_index=0,commands=np.zeros((10,2),np.float32).tolist()))
    for group in ('shared',)+m.ARMS:
        n=8 if group=='shared' else 128
        for resource in ('action_encoder','predictor','pred_proj'):
            shape=[3,192] if resource=='pred_proj' else [1,3,10] if resource=='action_encoder' else [1,3,192]
            shapes.extend(dict(resource=resource,scope=[group,0],shape=shape) for _ in range(n))
        events.append(dict(kind='resource_return',entry=dict(scope=[group,0],resource='predict',started_seconds=1.,ended_seconds=2.)))
    axes={r:dict(observed_calls=520,input_rows=1560 if r=='pred_proj' else 520,context_tokens=None if r=='pred_proj' else 1560)
          for r in ('action_encoder','predictor','pred_proj')}
    verified=dict(schema='chat_p3_20261027_cuda.consume',status='actual_rank_queues_verified_not_admitted',
        plans=plans,resources=dict(predict='SYNTHETIC actual reference'),scope_resources={},module_sample_axes=axes,
        paired_summary=dict(D_minus_control=dict(U=[0.,-1.],S_rank=[0.,-1.],N=[0.,-1.])),new_wall_seconds=3.,**m.FLAGS)
    report=dict(actual_module_shapes=shapes,backward_engine_calls=None)
    return p,verified,result,events,report


@pytest.mark.parametrize('weights',[(2.,1.),(1.,2.),(0.,1.),(1.,0.)])
def test_real_discrete_requests_and_minimum_and_same_total(weights):
    p,_,_,_=fixture(weights); q=p['queue']
    assert q['allocation_separated'] and q['budget']['total_predict_API']==520
    assert q['queues']['U']['blocks']==q['queues']['N']['blocks']==[10,10]
    assert q['queues']['D_rank']['blocks']==q['queues']['S_rank']['blocks'][::-1]
    for a in m.ARMS:
        jobs=q['queues'][a]['requests']
        assert len(jobs)==60 and sum(j['predict_API'] for j in jobs)==120
        assert [j['slot'] for j in jobs[:30]].count(0)==15
        for s in (0,1):
            assert [j['index'] for j in jobs if j['slot']==s]==list(range(3*q['queues'][a]['blocks'][s]))
            assert len({j['identity']['task_id'] for j in jobs if j['slot']==s})==1
        assert [x['mode'] for x in q['queues'][a]['replay_order']]==['whole_replay','segmented_replay']*2


@pytest.mark.parametrize('weights',[(0.,0.),(1.,1.)])
def test_equal_weights_collision_is_not_forced_apart(weights):
    p,_,_,_=fixture(weights)
    assert not p['queue']['allocation_separated'] and len(p['queue']['allocation_collision_pairs'])==6
    result=m.summarize(*full(weights))
    assert all(len(r['commands_collision_pairs'])==6 for r in result['collisions'])


def test_actual_axes_no_equal_flops_and_preserve_zero_negative_and_action_collision():
    result=m.summarize(*full())
    assert result['paired_model_only']['D_minus_control']['U']==[0.,-1.]
    assert result['queue']['allocation_separated'] and all(len(x['winner_collision_pairs'])==6 for x in result['collisions'])
    assert result['resources']['observed_samples_by_group']['D_rank']['predictor']['input_rows']==128
    assert result['resources']['observed_samples_by_group']['shared']['pred_proj']['input_rows']==24
    assert result['resources']['execution_wall_seconds']==3.
    assert result['resources']['backward_engine_calls'] is None
    assert result['new_neural_calls']==result['new_physics_calls']==0
    assert all(result[k] is False for k in m.FLAGS)
    assert result['status']=='completed_saved_execution_do_not_rerun_not_admitted'


@pytest.mark.parametrize('field',['weight','order','floor','hash','raw','identity'])
def test_queue_mutations_rejected(field):
    _,p,banks,sealed=fixture()
    if field=='weight':sealed['weights'][0]=float('nan')
    elif field=='order':sealed['plans']['D_rank']['jobs'].reverse()
    elif field=='floor':sealed['plans']['D_rank']['blocks']=[16,4]
    elif field=='hash':sealed['plans']['D_rank']['candidate_sha256'][0]=h('wrong')
    elif field=='raw':banks[0][0]['commands'][0][0]=.1
    else:p['slots'][0]['source_slot']=True
    with pytest.raises(ValueError): m.compile_queue(p,banks,sealed,rule)


@pytest.mark.parametrize('field',['missing_plan','identity','order','cost_after_allocation','paid','sample','signal','admission','time'])
def test_full_consume_mutations_failclosed(field):
    p,v,r,e,report=full()
    if field=='missing_plan':v['plans'].pop()
    elif field=='identity':v['plans'][0]['task_id']=h('wrong')
    elif field=='order':next(x for x in e if x['kind']=='queue_frozen')['jobs'].reverse()
    elif field=='cost_after_allocation':next(x for x in e if x['kind']=='candidate_request')['candidate']['cost']=-100.
    elif field=='paid':next(x for x in e if x['kind']=='candidate_request')['paid']=True
    elif field=='sample':report['actual_module_shapes'].pop()
    elif field=='signal':r['signals'][0]['signal']['w']=1.9
    elif field=='admission':v['P1_admission']=True
    else:next(x for x in e if x['kind']=='resource_return')['entry']['ended_seconds']=0.
    with pytest.raises(ValueError):m.summarize(p,v,r,e,report)


def test_prepare_verify_external_sha_and_whole_reconstruction(monkeypatch):
    prepared=dict(schema=m.NAME+'.prepare',value=1)
    monkeypatch.setattr(m,'material',lambda:deepcopy(prepared)); monkeypatch.setattr(m,'sha',lambda _:h('actual RAM prepare'))
    monkeypatch.setattr(m,'read',lambda _:deepcopy(prepared))
    assert m.verify('/RAM',h('actual RAM prepare'))==prepared
    with pytest.raises(ValueError):m.verify('/RAM',h('invented future hash'))
    monkeypatch.setattr(m,'read',lambda _:dict(schema=m.NAME+'.prepare',value=2))
    with pytest.raises(ValueError):m.verify('/RAM',h('actual RAM prepare'))


def test_queue_api_emits_literal_full_requests_without_execution(monkeypatch):
    p,*_=fixture();p['model_hash']=h('frozen model');p['execution_adapter']='existing sealed executor'
    monkeypatch.setattr(m,'verify',lambda *_:p)
    q=m.queue('/RAM',h('RAM prepare'),'D_rank')
    assert len(q['requests'])==60 and not q['neural_execution_started'] and not q['P3_admission']
    with pytest.raises(ValueError):m.queue('/RAM',h('RAM prepare'),'unknown')


def test_output_scope_and_no_runtime_loaded():
    with pytest.raises(ValueError):m.write_new('/tmp/not_authorized.json',{})
    assert not any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel'))
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                n.func.attr in ('run','execute','fit','solve','lstsq','pinv','load_compatible_pretrained')]
