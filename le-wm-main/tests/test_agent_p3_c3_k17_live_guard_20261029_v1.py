import copy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_live_guard_20261029_v1 as m

def fixture():
    state=[1.,2.,3.,4.,5.,6.,7.]; goal=[7.,6.,5.,4.,3.,2.,1.]
    root=m.root_hash(state,goal)
    task=dict(binding=dict(task_id='task',root_sha256=root),action_mean=[0.,0.],action_std=[1.,1.],
      root_context=dict(action_history=[[0.]*10,[0.]*10],latent_history=[[0.],[0.],[0.]]))
    def frame(step,k):
        condition='on' if k==1 else 'off'
        inv=dict(goal_state=goal,goal_pose=[0.],goal_rgb_sha256='goal',renderer={'render':1},snapshot={'state':state})
        receipt=dict(phase='warmup',step=step,overlay_index=k,condition=condition,file_sha256=f'f{step}{condition}',
                     pixel_sha256=f'p{step}{condition}',state=state,snapshot=inv['snapshot'])
        return dict(pre=copy.deepcopy(inv),post=copy.deepcopy(inv),receipt=receipt)
    frames=[frame(s,k) for s,k in m.ORDER]
    trace=[frame(s,k) for s,k in m.REF_ORDER]
    for x in trace:
        if x['receipt']['overlay_index']==2: x['receipt']['overlay_index']=2
    bound=copy.deepcopy(frames[-1]['pre'])
    gf=copy.deepcopy(frames[-1]); gf['receipt'].update(phase='plan',step=0,overlay_index=0,condition='off')
    raw=np.zeros((10,2),dtype='<f4')
    ops=[dict(step=i+1,status='returned',actual_dtype='<f4',requested_f32_hex=raw[i].tobytes().hex(),
      actual_f32_hex=raw[i].tobytes().hex(),terminated=False,truncated=False) for i in range(10)]
    reference=dict(task_sha256=m.digest(task),task_id='task',root_sha256=root,reset_identity={'seed':1},
      warm_trace=trace,warm_raw=raw.tolist(),root_state=state,goal_state=goal,bound_goal=bound,goal_frame=gf)
    live=dict(reset_identity={'seed':1},warm_frames=frames,warm_returns=ops,root_state=state,
      goal_state=goal,bound_goal=bound,goal_frame=gf)
    identity=dict(slot=3,root=0,mode='fixedH1',K=17)
    return identity,task,reference,live

def proof():
    i,t,r,l=fixture(); return m.history_guard(identity=i,task=t,reference=r,live=l)

def test_guard_and_missing_historical_on():
    result=proof(); assert result['old_unpaired_on_steps']==[1,2,3,4,6,7,8,9]
    assert result['accepted'] and not result['physical_release']

@pytest.mark.parametrize('what',['identity','reset','frames','render','state','goal','snapshot','command','truncation','dtype','history','reference'])
def test_fail_closed(what):
    i,t,r,l=fixture()
    if what=='identity': i['slot']=9
    if what=='reset': l['reset_identity']['seed']=2
    if what=='frames': l['warm_frames'].pop()
    if what=='render': l['warm_frames'][3]['receipt']['pixel_sha256']='changed'
    if what=='state': l['root_state'][0]=10
    if what=='goal': l['bound_goal']['goal_rgb_sha256']='different'
    if what=='snapshot': l['goal_frame']['receipt']['snapshot']={}
    if what=='command': l['warm_returns'][0]['actual_f32_hex']='ff'*8
    if what=='truncation': l['warm_returns'][0]['truncated']=True
    if what=='dtype': l['warm_returns'][0]['actual_dtype']='<f8'
    if what=='history': t['root_context']['action_history'][0][0]=1.
    if what=='reference': r['warm_trace'].pop()
    with pytest.raises(ValueError): m.history_guard(identity=i,task=t,reference=r,live=l)

def test_action_replay_tokens_and_domain():
    a=m.action_guard([[0.,-0.]]*5,[0.,0.],[1.,1.],[0.,-0.]*5)
    assert a['raw_f32_hex'].startswith('0000000000000080') and a['actual_replay_tokens_exact']
    with pytest.raises(ValueError): m.action_guard([[2.,0.]]*5,[0.,0.],[1.,1.],[0.]*10)
    with pytest.raises(ValueError): m.action_guard([[0.,0.]]*5,[0.,0.],[1.,1.],[1.]*10)
    with pytest.raises((ValueError,TypeError)): m.action_guard([[0.,0.]]*5,[0.,0.],[1.,1.],None)

def test_real_journal_guard_before_dispatch(tmp_path):
    i,t,r,l=fixture(); j=m.DurableFixtureJournal(tmp_path/'journal.jsonl'); g=m.GuardedDispatch(j); seen=[]
    g.arm(identity=i,task=t,reference=r,live=l,raw5=[[0.,0.]]*5,replay_tokens=[0.]*10)
    def call(x):
        events=[json.loads(z) for z in j.path.read_bytes().splitlines()]
        assert [e['kind'] for e in events]==['guard_result','commit_intent']; seen.append(x.tobytes()); return {'actual_dtype':'<f4','actual_f32_hex':x.tobytes().hex()}
    assert g.first(current_goal=l['bound_goal'],current_frame=l['goal_frame'],command=[0.,0.],dispatch=call)['actual_dtype']=='<f4'
    with pytest.raises(ValueError): g.first(current_goal=l['bound_goal'],current_frame=l['goal_frame'],command=[0.,0.],dispatch=call)
    j.close(); events=[json.loads(z) for z in j.path.read_bytes().splitlines()]
    assert len(seen)==1 and [e['kind'] for e in events]==['guard_result','commit_intent','commit_return']
    assert events[1]['previous']==events[0]['sha256'] and events[2]['previous']==events[1]['sha256']

def test_rejected_guard_persisted_no_dispatch(tmp_path):
    i,t,r,l=fixture(); l['reset_identity']['seed']=9
    j=m.DurableFixtureJournal(tmp_path/'journal.jsonl'); g=m.GuardedDispatch(j)
    with pytest.raises(ValueError): g.arm(identity=i,task=t,reference=r,live=l,raw5=[[0.,0.]]*5,replay_tokens=[0.]*10)
    assert json.loads(j.path.read_bytes())['payload']['accepted'] is False
    with pytest.raises(ValueError): g.first(current_goal=l['bound_goal'],current_frame=l['goal_frame'],command=[0.,0.],dispatch=lambda x:pytest.fail('dispatch'))
    j.close()

def test_journal_failure_and_stale_root_block_dispatch(tmp_path):
    i,t,r,l=fixture()
    class Broken:
        def emit(self,*args): raise OSError('fsync failure')
    g=m.GuardedDispatch(Broken())
    with pytest.raises(OSError): g.arm(identity=i,task=t,reference=r,live=l,raw5=[[0.,0.]]*5,replay_tokens=[0.]*10)
    assert not g.ready
    j=m.DurableFixtureJournal(tmp_path/'j'); g=m.GuardedDispatch(j)
    g.arm(identity=i,task=t,reference=r,live=l,raw5=[[0.,0.]]*5,replay_tokens=[0.]*10)
    l['goal_frame']['receipt']['state']=[0.]*7
    with pytest.raises(ValueError): g.first(current_goal=l['bound_goal'],current_frame=l['goal_frame'],command=[0.,0.],dispatch=lambda x:pytest.fail('dispatch'))
    j.close()

def test_bad_actual_return_persisted(tmp_path):
    i,t,r,l=fixture(); j=m.DurableFixtureJournal(tmp_path/'j'); g=m.GuardedDispatch(j)
    g.arm(identity=i,task=t,reference=r,live=l,raw5=[[0.,0.]]*5,replay_tokens=[0.]*10)
    bad={'actual_dtype':'<f4','actual_f32_hex':'ff'*8}
    with pytest.raises(ValueError): g.first(current_goal=l['bound_goal'],current_frame=l['goal_frame'],command=[0.,0.],dispatch=lambda x:bad)
    j.close(); e=[json.loads(x) for x in j.path.read_bytes().splitlines()]
    assert e[-1]['kind']=='commit_failure' and e[-1]['payload']['returned_value']==bad

def test_inverse_is_diagnostic_original_raw_is_not_replaced():
    raw=np.full((5,2),np.float32(.1234567),dtype=np.float32)
    mean=np.array([.0087,.0068],dtype=np.float32); std=np.array([.2019,.2002],dtype=np.float32)
    token=((raw-mean)/std).reshape(10)
    result=m.action_guard(raw,mean,std,token)
    assert result['raw_f32_hex']==raw.tobytes().hex()
    assert result['raw_inverse_bitexact'] == (result['raw_f32_hex']==result['inverse_f32_hex'])
