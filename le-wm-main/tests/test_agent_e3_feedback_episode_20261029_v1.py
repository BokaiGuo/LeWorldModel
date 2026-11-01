import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_feedback_episode_20261029_v1 as e
from test_agent_e3_gd_host_cuda_20261029_v1 import body


class Session:
    episode_id='cell0'
    def __init__(self,stop=None):self.calls=[];self.stop=stop
    def packet(self,start,raw):
        self.calls.append(('packet',start));return dict(raw=raw,stopped=start==self.stop,end=start+5,rgb_bytes=str(start).encode())
    def encode(self,boundary,rgb):
        self.calls.append(('encode',boundary));return dict(latent=np.full(192,boundary,np.float32),
            input_sha=hashlib.sha256(rgb).hexdigest(),episode_id=self.episode_id,boundary=boundary)


def fixture():
    b=body();prior=np.full((5,2),.1,np.float32);mean=np.asarray(b['action_mean'],np.float32);std=np.asarray(b['action_std'],np.float32)
    b['root_context']['action_history'][-1]=((prior-mean)/std).reshape(10).tolist()
    return b,prior


def test_sequential_actual_session():
    b,prior=fixture();s=Session();events=[];contexts=[]
    def plan(t):contexts.append(t);return {'first5':np.full((5,2),.2,np.float32).tolist()}
    out=e.episode(b,np.full((5,2),.3,np.float32),prior,s,plan,events.append)
    assert out['closed_loop_complete'] and not out['production_receipt']
    assert s.calls==[('packet',5),('encode',10),('packet',10),('encode',15),('packet',15)]
    assert [x['feedback_boundary'] for x in contexts]==[10,15]
    assert contexts[0]['root_context']['latent_history'][-1]==[10.0]*192
    assert contexts[1]['root_context']['latent_history'][-1]==[15.0]*192


def test_stop_does_not_encode_or_plan():
    b,prior=fixture();s=Session(stop=5)
    out=e.episode(b,np.zeros((5,2),np.float32),prior,s,lambda t:pytest.fail('unreachable'),lambda x:None)
    assert not out['closed_loop_complete'] and s.calls==[('packet',5)]
