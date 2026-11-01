from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_feedback_single_capture_20261029_v1 as a


class Store:
    def __init__(self):self.events=[]
    def emit(self,k,**v):self.events.append((k,v))


class Env:
    def __init__(self,stop,fail):self.unwrapped=self;self.n=0;self.stop=stop;self.fail=fail
    def step(self,raw):
        self.n+=1
        if self.n==self.fail:raise RuntimeError('native failed')
        self.latest_action=raw.copy()
        return {},-.5,self.n==self.stop,False,dict(goal_state=np.zeros(7),goal=np.zeros((2,2)))


@pytest.mark.parametrize('stop',[1,2,3,4,5,None])
def test_single_render_and_stop(stop):
    frames=[];env=Env(stop,None);store=Store();row=dict(episode_id='x',control_returned=0,executed_raw=[])
    def capture(env,obs,row,store,phase,step):frames.append(step);return None,dict(step=step)
    ops=SimpleNamespace(capture=capture,native_transition=lambda *args:None)
    out=a.packet(env,np.zeros((5,2),np.float32),start=5,row=row,goal=np.zeros(7),goal_rgb=np.zeros((2,2)),meter=None,store=store,ops=ops)
    n=stop or 5
    assert len(frames)==row['control_returned']==len(out['returned_raw'])==n
    assert out['end']==5+n and out['stopped']==(stop is not None)
    assert len(store.events)==2*n


def test_failure_keeps_prior_and_pending_intent():
    env=Env(None,3);store=Store();row=dict(episode_id='x',control_returned=0,executed_raw=[])
    ops=SimpleNamespace(capture=lambda e,o,r,s,p,i:(None,dict(step=i)),native_transition=lambda *args:None)
    with pytest.raises(RuntimeError,match='native failed'):
        a.packet(env,np.zeros((5,2),np.float32),start=5,row=row,goal=np.zeros(7),goal_rgb=np.zeros((2,2)),meter=None,store=store,ops=ops)
    assert row['control_returned']==2 and [k for k,_ in store.events]==['authorized_dispatch','control_return']*2+['authorized_dispatch']
