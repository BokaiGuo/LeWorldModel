import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p1_future_joint_candidate_host_20261029_v1 as m


class ToyModel:
    def action_encoder(self, actions): return actions
    def predict(self, z, actions):
        # The future action has an effect that depends on the prefix state.
        effect = actions[:,-1,0:1] * (1 + z[:,-1,0:1])
        out = z.clone()
        last = z[:,-1,:] + effect.expand(-1,192)
        return torch.cat((out[:,:-1,:], last[:,None,:]), 1)


def test_terminal_backpropagates_through_future_state_and_action_history():
    z = torch.zeros(1,3,192); history = torch.zeros(1,2,10)
    a = torch.zeros(10, requires_grad=True); b = torch.zeros(10, requires_grad=True)
    with torch.no_grad(): a[0] = .2; b[0] = .4
    mid, end = m.two_step(ToyModel(), z, history, a, b, torch.zeros(2), torch.ones(2), torch)
    ga, gb = torch.autograd.grad(end.mean(), (a,b))
    assert mid[0].item() == pytest.approx(.2)
    assert end[0].item() == pytest.approx(.68)
    assert ga[0].item() == pytest.approx(1.4)
    assert gb[0].item() == pytest.approx(1.2)


def test_projection_and_chain_are_bounded_and_deterministic():
    x = np.zeros(10); action = np.tanh(x).astype(np.float32)
    out = m.update(x, np.ones(10)*100, action)
    assert np.linalg.norm(out) <= m.host.F.CAP
    assert np.array_equal(out, m.update(x, np.ones(10)*100, action))
    with pytest.raises(Exception): m.update(x, np.full(10, np.nan), action)


def test_fixed_budget_and_starts():
    c = m.contract()
    assert c['candidate_denominator'] == 4*4*9 == 144
    assert c['predict_calls'] == 4*(2+4*9*2) == 296
    assert c['scalar_reverse_calls'] == 4*4*8 == 128
    assert c['P1_admission'] is False and c['freshness'] is False
    values = m.starts(np.zeros(10))
    assert len(values) == 4 and values[2][0] == .25 and values[3][0] == -.25
    with pytest.raises(Exception): m.starts(np.ones(10)*100)


def test_cuda_requires_explicit_authority_before_source_access():
    with pytest.raises(Exception, match='authorization'):
        m.run('does-not-exist','bad','bad','not-authorized')
