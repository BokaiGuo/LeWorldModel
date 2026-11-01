from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v2 as m


def state(i):return np.full(7,i,dtype=np.float64)
def pixel(i):return np.full((224,224,3),i,dtype=np.uint8)
def imsha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def root_hash(s,g):return hashlib.sha256(np.asarray([*s,*g],dtype=np.float64).tobytes()).hexdigest()


class Env:
    def __init__(self,stop=None,success=False):
        self.unwrapped=self;self.block=SimpleNamespace(shapes=[]);self.i=0;self.stop=stop;self.success=success
        self.goal_state=state(40);self._goal=pixel(40);self.goal_pose=[0,0,0];self.renders=0
    def _get_obs(self):return state(self.i)
    def reset(self,**kw):return {'state':state(self.i)},{'goal_state':state(40)}
    def step(self,a):
        self.i+=1;self.latest_action=a.copy()
        return {'state':state(self.i)},0.,self.i==self.stop,False,{'goal_state':self.goal_state,'goal':self._goal}
    def eval_state(self,goal,s):return self.success,0. if self.success else 1.


class Store:
    def __init__(self):self.events=[]
    def emit(self,k,**kw):self.events.append((k,kw))


class Meter:
    region=None
    def call(self,k,fn):return fn()


def case(stop=None,success=False):
    env=Env(stop,success);store=Store();row=dict(episode_id='fixture',warmup_returned=0,control_returned=0,executed_raw=[])
    warm=np.zeros((10,2),np.float32);prefix=np.ones((5,2),np.float32)*.1
    def capture(env,obs,row,store,phase,step):
        env.renders+=1;rgb=pixel(env.i)
        return rgb,dict(step=step,state=state(env.i).tolist(),pixel_sha256=imsha(rgb),snapshot={'i':env.i})
    def attach(body,goal,rgb):body.goal_state=goal.copy();body._goal=rgb.copy()
    moving=SimpleNamespace(exact=lambda x,y:np.asarray(x).dtype==np.asarray(y).dtype and np.array_equal(x,y),
                           image_sha=imsha,attach_goal=attach)
    source=SimpleNamespace(vector=lambda x:np.asarray(x,dtype=np.float64),sample_state=lambda *a:state(0),root_hash=root_hash)
    def seal(body):return dict(snapshot={'i':body.i},goal_pose=body.goal_pose,goal=body.goal_state.tolist(),goal_rgb_sha256=imsha(body._goal))
    ops=SimpleNamespace(source=source,kernel=SimpleNamespace(moving=moving,raw_history=lambda x:x),capture=capture,
                        state_seal=seal,array_sha=imsha,native_transition=lambda *a:None)
    ref=dict(slot=dict(reset_seed=1,root_seed=2),raw_history=warm,
             history=np.array([pixel(i) for i in (0,5,10)]),states=np.array([state(i) for i in (0,5,10)]),
             dynamics=[{'i':i} for i in (0,5,10)],goal=state(41))
    kw=dict(ref=ref,source_plan={},original_root_sha=root_hash(state(10),state(41)),new_goal=state(10),
        new_rgb=pixel(10),expected_goal=state(10),expected_goal_pixel=imsha(pixel(10)),prefix5=prefix,
        expected_cp5=dict(step=5,state=state(15).tolist(),pixel_sha256=imsha(pixel(15)),snapshot={'i':15}),
        row=row,meter=Meter(),store=store,ops=ops)
    return env,kw,store,row


def test_full_setup_single_capture_and_cp5():
    env,kw,store,row=case();result=m.setup(env,**kw)
    assert result['cp5_observation']['step']==5 and env.i==15 and env.renders==17
    assert row['warmup_returned']==10 and row['control_returned']==5 and row['initial_success'] is False
    kinds=[k for k,_ in store.events]
    assert kinds.index('original_root_accepted')<kinds.index('actual_root_guard')<kinds.index('authorized_dispatch')<kinds.index('actual_cp5_guard')


@pytest.mark.parametrize('stop',[1,7,11,13,15])
def test_warm_or_mid_prefix_stop_never_pads(stop):
    env,kw,store,row=case(stop=stop);assert m.setup(env,**kw) is None
    assert env.i==stop and env.renders==(1+stop+(1 if stop>10 else 0))
    assert row['status'] in ('native_stopped_warmup','native_stopped_prefix')
    assert not any(k=='actual_cp5_guard' for k,_ in store.events)


def test_cp5_initial_success_no_task_dispatch():
    env,kw,store,row=case(success=True);assert m.setup(env,**kw) is None
    assert env.i==15 and row['initial_success'] and row['initial_success_control']==5
    assert row['status']=='native_initial_success_control5'


def test_cp5_tamper_fails_before_task():
    env,kw,store,row=case();kw['expected_cp5']['pixel_sha256']='0'*64
    with pytest.raises(ValueError,match='cp5'):m.setup(env,**kw)
    assert env.i==15 and not any(k=='initial_native_evaluation' for k,_ in store.events)
