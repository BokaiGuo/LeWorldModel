from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as node
from test_agent_e3_gd_host_cuda_20261029_v1 import body, FrozenRAM, algebra_forward, Tensor, grad
import numpy as np


class Backend:
    def __init__(self,b):self.b=b
    def forward(self,c,a,tag):return algebra_forward(c,a,self.b)[1]
    def gradient(self,c,a,goal,tag):
        leaf=Tensor(a).requires_grad_(True);z,context=algebra_forward(c,a,self.b,leaf)
        loss=((z[0,-1]-Tensor(goal))**2).mean()
        return context,grad(loss,leaf)[0].a,np.float32(loss.item())


@pytest.mark.parametrize('k,domain',[(1,'Dxy'),(1,'Dx'),(3,'Dxy'),(3,'Dx')])
def test_identical_control5_arithmetic(k,domain):
    b=body(k,domain);old_events=[];new_events=[]
    old=node.host.solve(b,Backend(b),old_events.append)
    later=deepcopy(b);later['feedback_boundary']=10
    actual=node.solve(later,Backend(later),new_events.append)
    assert actual==old and new_events==old_events


def test_only_later_boundary():
    b=body();
    with pytest.raises(ValueError):node.validate_body(b)
    b['feedback_boundary']=15;node.validate_body(b)
    b['feedback_boundary']=True
    with pytest.raises(ValueError):node.validate_body(b)
