from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_feedback_native_packet_20261029_v1 as m


class Store:
    def __init__(self):self.events=[]
    def emit(self,kind,**kw):self.events.append((kind,kw))


def fixture(stop=None,fail=None):
    store=Store();calls=[];goal=np.arange(7,dtype=np.float64);rgb=np.zeros((224,224,3),np.uint8)
    def dispatch(env,command,phase,step,row,meter,store,goal):
        calls.append(step)
        if step==fail:raise RuntimeError('original failure')
        return {},{'goal_state':goal,'goal':rgb},step==stop,False,0.,command.copy()
    def observe(env,obs,phase,step,row,meter,store,trace):return ('own_frame',step)
    ops=SimpleNamespace(dispatch=dispatch,allmode=SimpleNamespace(observe=observe),
        paired=SimpleNamespace(check_return=lambda *a:False),moving=SimpleNamespace(exact=np.array_equal))
    return store,calls,goal,rgb,ops


@pytest.mark.parametrize('stop',[6,7,8,9,10,None])
def test_exact_partial_native_packet(stop):
    store,calls,goal,rgb,ops=fixture(stop);a=np.full((5,2),.2,np.float32)
    r=m.packet(None,a,start=5,row={},goal=goal,goal_rgb=rgb,trace=None,meter=None,store=store,ops=ops)
    n=5 if stop is None else stop-5
    assert calls==list(range(6,6+n)) and r['end']==5+n and len(r['returned_raw'])==n
    assert r['complete']==(n==5) and r['stopped']==(stop is not None)
    assert len(store.events)==n


def test_exception_never_dispatches_later():
    store,calls,goal,rgb,ops=fixture(fail=8)
    with pytest.raises(RuntimeError,match='original failure'):
        m.packet(None,np.zeros((5,2),np.float32),start=5,row={},goal=goal,goal_rgb=rgb,
                 trace=None,meter=None,store=store,ops=ops)
    assert calls==[6,7,8] and len(store.events)==2


def test_reject_float64_and_bad_boundary():
    store,calls,goal,rgb,ops=fixture()
    with pytest.raises(ValueError):m.packet(None,np.zeros((5,2)),start=5,row={},goal=goal,goal_rgb=rgb,trace=None,meter=None,store=store,ops=ops)
    with pytest.raises(ValueError):m.packet(None,np.zeros((5,2),np.float32),start=True,row={},goal=goal,goal_rgb=rgb,trace=None,meter=None,store=store,ops=ops)
    assert calls==[]


def test_original_root_separate_from_new_goal():
    state=np.arange(7,dtype=np.float64);old=state+1;new=state+2
    rgb=np.zeros((224,224,3),np.uint8);env=SimpleNamespace(unwrapped=SimpleNamespace(_get_obs=lambda:state))
    original=m.native.source.root_hash(state,old);pixel=m.native.moving.image_sha(rgb)
    box={'goal':old.tolist(),'pixel':'original'};store=Store()
    def invariant(body):return dict(snapshot={'state':state.tolist()},goal_pose=[0.,0.,0.],renderer={'off':True},
                                    goal_state=box['goal'],goal_rgb_sha256=box['pixel'])
    def bind(body,goal,image,meter,store):box.update(goal=goal.tolist(),pixel=m.native.moving.image_sha(image))
    ops=SimpleNamespace(source=m.native.source,moving=m.native.moving,invariant=invariant,bind_goal=bind)
    m.root_goal_binding(env,old,original,new,rgb,new,pixel,None,store,ops=ops)
    assert [x[0] for x in store.events]==['original_root_accepted','new_goal_accepted']
    assert box['goal']==new.tolist()
    with pytest.raises(ValueError,match='original root'):
        m.root_goal_binding(env,new,original,new,rgb,new,pixel,None,Store(),ops=ops)
