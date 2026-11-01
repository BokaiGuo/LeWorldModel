import numpy as np
import pytest
from experiments.spectral_reachability import p2_h15_gn_native_ad_20261029_v1 as m


@pytest.mark.parametrize('boundary', [False, True])
def test_projected_difference_uses_actual_displacement(boundary):
    z=np.zeros(30)
    if boundary:z[0]=m.gn.RADIUS
    direction=np.arange(1,31,dtype=float)
    a,b=m.probes(z,direction)
    assert np.linalg.norm(a)<=m.gn.RADIUS+1e-12
    assert np.linalg.norm(b)<=m.gn.RADIUS+1e-12
    j=np.zeros((192,30));j[:30]=np.eye(30)
    m.check_difference(a,b,j@a,j@b,j)
    with pytest.raises(ValueError,match='directional difference'):
        m.check_difference(a,b,j@a,j@b,j*3)


def test_invalid_and_degenerate_probes():
    with pytest.raises(ValueError):m.probes(np.zeros(30),np.zeros(30))
    with pytest.raises(ValueError):m.check_difference(np.zeros(30),np.zeros(30),np.zeros(192),np.zeros(192),np.zeros((192,30)))


def test_real_autograd_current_action_and_ledger():
    import torch
    from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
    class Encoder(torch.nn.Module):
        def forward(self, x):
            return x.mean(-1, keepdim=True).expand(-1,-1,192)
    class Predictor(torch.nn.Module):
        def forward(self, state, actions):
            return state + actions * .1
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_encoder=Encoder();self.predictor=Predictor()
            self.pred_proj=torch.nn.Identity();self.encoder=torch.nn.Identity();self.projector=torch.nn.Identity()
        def predict(self,state,encoded):
            return self.pred_proj(self.predictor(state,encoded).reshape(3,192)).reshape(1,3,192)
    class Sink:
        def event(self, event): pass
    class CPUFixture(m.NativeCurrentAD):
        def __init__(self):
            self.device=torch.device('cpu');self.model=Model().eval().requires_grad_(False);self.store=Sink()
            self.task={'root_context':{'latent_history':np.zeros((3,192)).tolist(),
                                      'action_history':np.zeros((2,10)).tolist()},
                       'action_mean':[0.,0.], 'action_std':[1.,1.]}
        def frozen(self):
            assert not self.model.training
        def arrays(self,kind,operation_index=None,**values):
            assert all(v.device.type=='cpu' and v.dtype==torch.float32 for v in values.values())
    caps=dict.fromkeys(core.AXES,10000)
    ledger=core.Ledger(caps);ledger.scope={'test':1}
    z=np.linspace(-.2,.2,30)
    backend=CPUFixture();out=backend.linearize(z,ledger)
    assert ledger.attempted['VJP_directions']==192
    assert ledger.attempted['predict_samples']==3
    assert ledger.attempted['raw_backward_callbacks']==192
    assert ledger.attempted==ledger.completed
    j=np.asarray(out['logit_jacobian'])
    def terminal(x):
        a=m.gn.raw(x).reshape(3,10)
        return np.full(192, .1*sum(a[i].mean() for i in range(3)))
    a,b=m.probes(z,np.arange(1,31))
    m.check_difference(a,b,terminal(a),terminal(b),j)
    assert np.linalg.norm(j)>0
