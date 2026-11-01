from copy import deepcopy
import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_intersection_physical_prepare_20261029_v1 as m


def fixture():
    norm=m.physical.domain.normalization(); goal=[0.]*192; q=np.eye(192).tolist()
    ctx=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.zeros((2,10),np.float32).tolist())
    slots=[dict(task_id=str(i),task_file='task',task_sha256='a'*64,packet=dict(root_context=deepcopy(ctx),goal=goal),q=q,warm_raw10=np.zeros((10,2),np.float32).tolist()) for i in range(2)]
    request=dict(root_context=ctx,goal=goal,q=q,normalization=norm)
    decoded=m.physical.domain.raw15_inputs(request,[0.]*30); contexts=[]; prev=ctx
    for token in decoded['normalized_packets']:
        nxt=deepcopy(prev); nxt['action_history']=(np.asarray(prev['action_history'],np.float32).tolist()+[token])[-2:]; contexts.append(nxt); prev=nxt
    r=dict(schema=m.producer.NAME+'.consume',status='all6_joint_box_saved_verified_not_scientific_admission',report_sha256='b'*64,
        fixed_denominator=6,plans=[dict(logits=[0.]*30,contexts=deepcopy(contexts)) for _ in range(6)],
        accounts=[dict(root=root,arm=arm,used={'predict_API':1}) for root,_,arm in m.ORDER],
        missing_plan_indices=[],partial_arm_indices=[],new_neural_calls=0,new_physics_calls=0,training=0,
        physical_release=False,P1_admission=False,P2_mechanism=False,equal_FLOPs=False,actual_cost_equality=False)
    p=dict(schema=m.producer.NAME+'.prepare',root_arm_order=[[root,arm] for root,_,arm in m.ORDER],slots=slots)
    return p,r,[dict(root_sha256='c'*64),dict(root_sha256='d'*64)]


def test_six_real_contract_shapes():
    p,r,roots=fixture(); out=m.assemble(p,r,roots,'b'*64,'e'*64)
    assert len(out['plans'])==6 and len(out['accounts'])==6
    assert all(len(x['raw15'])==15 and len(x['raw_float32_bits'])==240 for x in out['plans'])
    assert out['native_physical_run_implemented'] is False and out['physical_release'] is False


@pytest.mark.parametrize('kind',['order','context','radius','missing','flags','account'])
def test_fail_closed(kind):
    p,r,roots=fixture()
    if kind=='order': p['root_arm_order'][0][1]='grasp'
    elif kind=='context': r['plans'][0]['contexts'][0]['action_history'][1][0]=1.
    elif kind=='radius': r['plans'][0]['logits'][0]=100.
    elif kind=='missing': r['plans'][0]=None
    elif kind=='flags': r['physical_release']=True
    elif kind=='account': r['accounts'][0]['arm']='grasp'
    with pytest.raises((ValueError,AssertionError)): m.assemble(p,r,roots,'b'*64,'e'*64)


def test_missing_slot_is_not_replaced():
    p,r,roots=fixture(); r['plans'][2]=None; r['missing_plan_indices']=[2]
    out=m.assemble(p,r,roots,'b'*64,'e'*64)
    assert out['plans'][2] is None and len(out['plans'])==6
