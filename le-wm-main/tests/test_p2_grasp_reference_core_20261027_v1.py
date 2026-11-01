"""Analytic tensor algebra only: no neural models, weights, GPU or physics."""
import pytest
import torch
from experiments.spectral_reachability import p2_grasp_reference_core_20261027_v1 as m


class Affine:
    action_dim = 1
    def edge(self, s, a):
        return 2*s + 3*a
    def begin(self, s):
        return s
    def advance(self, state, action):
        out = self.edge(state, action)
        return out, out


def test_stopgrad_is_state_input_only_not_residual_target_or_action():
    s = torch.tensor([[[2.]]], requires_grad=True)
    a = torch.tensor([[[.3], [.5]]], requires_grad=True)
    start, goal = torch.tensor([[1.]]), torch.tensor([[4.]])
    loss = m.soft_loss(Affine(), start, goal, s, a)
    ds, da = torch.autograd.grad(loss, (s, a))
    assert loss.item() == pytest.approx(6.52)
    assert ds.item() == pytest.approx(-1.8)
    assert da.flatten().tolist() == pytest.approx([-1.2, 18.])


def test_serial_rollout_keeps_state_chain_gradient_for_sync():
    a = torch.zeros(1, 2, 1, requires_grad=True)
    traj = m.serial(Affine(), torch.tensor([[1.]]), a)
    assert traj.flatten().tolist() == [1., 2., 4.]
    grad = torch.autograd.grad(traj[:, -1].sum(), a)[0]
    assert grad.flatten().tolist() == [6., 3.]


def test_fixture_schedule_has_independent_verification_and_exact_counts():
    cfg = m.Config(horizon=2, steps=4, sync_interval=2, sync_steps=1, state_noise=.01)
    out = m.plan(Affine(), torch.tensor([[1.]]), torch.tensor([[4.]]), cfg, analytic_fixture=True)
    assert out['receipt']['sync_after_iterations_1based'] == [2, 4]
    assert out['receipt']['transitions_completed'] == {'soft': 8, 'sync': 4, 'verification': 2}
    assert out['receipt']['backward_completed'] == 6
    assert len(out['receipt']['iterations']) == 4
    assert out['receipt']['official_equivalence'] is False
    assert out['verification'].shape == (1, 3, 1)


def test_cpu_runtime_not_implicitly_authorized():
    with pytest.raises(ValueError, match='CUDA'):
        m.plan(Affine(), torch.tensor([[1.]]), torch.tensor([[4.]]), m.Config(steps=1))


def test_seed_repeatability_without_global_rng_changes():
    start, goal = torch.tensor([[1.]]), torch.tensor([[4.]])
    state = torch.random.get_rng_state().clone()
    cfg = m.Config(horizon=2, steps=2, sync_interval=0)
    x = m.plan(Affine(), start, goal, cfg, analytic_fixture=True)
    y = m.plan(Affine(), start, goal, cfg, analytic_fixture=True)
    assert torch.equal(x['states'], y['states'])
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize('kwargs', [{'horizon':0}, {'steps':0}, {'lr_s':float('nan')},
    {'state_noise':-1}, {'sync_interval':-1}, {'sync_steps':0}, {'seed':True}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        m.Config(**kwargs)


def test_failure_retains_attempts_no_success_imputation():
    class Broken(Affine):
        def edge(self, s, a):
            raise RuntimeError('deliberate analytic fault')
    with pytest.raises(m.PlanningFailure) as error:
        m.plan(Broken(), torch.ones(1,1), torch.ones(1,1)*2, m.Config(steps=1), analytic_fixture=True)
    receipt = error.value.receipt
    assert receipt['transitions_requested']['soft'] == 1
    assert receipt['transitions_completed']['soft'] == 0
    assert receipt['events'][0]['status'] == 'failed'
    assert receipt['backward_completed'] == 0


def test_h1_supported_without_fake_virtual_state():
    out = m.plan(Affine(), torch.ones(1,1), torch.ones(1,1)*2,
                 m.Config(horizon=1, steps=1, sync_interval=0), analytic_fixture=True)
    assert out['states'].shape == (1,0,1)


def test_lewm_adapter_uses_h1_edge_and_generated_history_for_sync():
    class Model:
        def action_encoder(self, a): return a
        def predict(self, s, a): return 2*s+3*a
    model=Model()
    wm=m.LeWMH1(model, history_size=3, analytic_fixture=True)
    s=torch.ones(1,1);a=torch.zeros(1,2,1,requires_grad=True)
    assert wm.edge(s,a[:,0]).item()==2.
    assert m.serial(wm,s,a).flatten().tolist()==[1.,2.,4.]


def test_native_adapter_preserves_patch_shape_and_propagates_proprio_in_sync():
    class Predictor:
        def __init__(self):self.prop=[]
        def __call__(self, visual, action, proprio):
            self.prop.append(proprio.detach().clone())
            b,t=visual.shape[:2]
            y=visual.reshape(b,t,2,1)+action[...,:1,None]
            return y,None,proprio+1
    predictor=Predictor()
    wm=m.NativeH1(predictor, last_proprio=torch.zeros(1,1,2,1),
                  visual_shape=(1,1,2,1), action_dim=1, history_size=3, analytic_fixture=True)
    result=m.serial(wm,torch.zeros(1,2),torch.ones(1,2,1))
    assert result[0,:,0].tolist()==[0.,1.,2.]
    assert predictor.prop[-1][0,-1,0,0].item()==1.


def test_no_claim_of_preserving_recorded_h3_context():
    assert m.protocol()['input_context'] == 'last_state_only_H1; generated history during sync'
    assert m.protocol()['same_as_existing_real_H3_contract'] is False
    counts=m.budget(m.Config())
    assert counts['soft_transition_samples_per_root']==2250
    assert counts['sync_transition_samples_per_root']==450
    assert counts['verification_transition_samples_per_root']==3
    assert counts['backward_api_calls_per_root']==900
