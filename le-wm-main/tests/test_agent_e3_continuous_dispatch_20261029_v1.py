import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_continuous_dispatch_20261029_v1 as m
from test_agent_e3_gd_host_cuda_20261029_v1 import body


class Store:
    def __init__(self):self.events=[]
    def emit(self,k,**v):self.events.append((k,v))


def fixture(stop=None,fail=None):
    b=body();prior=np.full((5,2),.1,np.float32);mu=np.asarray(b['action_mean'],np.float32);std=np.asarray(b['action_std'],np.float32)
    b['root_context']['action_history'][-1]=((prior-mu)/std).reshape(10).tolist()
    setup=dict(prior_raw5=prior,goal=np.zeros(7),goal_rgb=np.zeros((224,224,3),np.uint8))
    calls=[];env=object();store=Store();row=dict(episode_id='cell0')
    def pkt(instance,action,*,start,**kw):
        assert instance is env;calls.append(('packet',start))
        if fail==start:raise RuntimeError('actual packet failed')
        n=2 if stop==start else 5;rgb=str(start).encode()
        return dict(returned_raw=action[:n].copy(),end=start+n,stopped=stop==start,complete=n==5,
                    observation=dict(step=start+n,file_sha256=hashlib.sha256(rgb).hexdigest()))
    def enc(instance,ob,boundary):
        assert instance is env;calls.append(('encode',boundary));rgb=str(boundary-5).encode()
        return dict(latent=np.full(192,boundary,np.float32),rgb_bytes=rgb,input_sha=hashlib.sha256(rgb).hexdigest(),episode_id='cell0',boundary=boundary)
    def plan(task):
        calls.append(('plan',task['feedback_boundary']))
        assert task['root_context']['latent_history'][-1]==[float(task['feedback_boundary'])]*192
        return dict(first5=np.full((5,2),.2,np.float32).tolist())
    return b,setup,env,store,row,calls,pkt,enc,plan


def test_same_env_observed_feedback():
    b,s,env,store,row,calls,pkt,enc,plan=fixture()
    r=m.dispatch(env,b,s,np.full((5,2),.3,np.float32),enc,plan,store,None,row,packet_fn=pkt)
    assert r['status']=='completed20'
    assert calls==[('packet',5),('encode',10),('plan',10),('packet',10),('encode',15),('plan',15),('packet',15)]


@pytest.mark.parametrize('start',[5,10,15])
def test_midpacket_native_stop(start):
    b,s,env,store,row,calls,pkt,enc,plan=fixture(stop=start)
    r=m.dispatch(env,b,s,np.full((5,2),.3,np.float32),enc,plan,store,None,row,packet_fn=pkt)
    assert r['status']=='native_stopped' and calls[-1]==('packet',start)
    assert r['slots'][(start-5)//5]['end']==start+2


def test_failure_preserves_unattempted_slots():
    b,s,env,store,row,calls,pkt,enc,plan=fixture(fail=10)
    with pytest.raises(RuntimeError,match='actual packet failed'):
        m.dispatch(env,b,s,np.full((5,2),.3,np.float32),enc,plan,store,None,row,packet_fn=pkt)
    assert row['status']=='failed' and row['slots'][2]['status']=='not_reached'
    assert store.events[-1][0]=='e3_dispatch_failure'
