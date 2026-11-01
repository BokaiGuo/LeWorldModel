"""Non-neural linear fixtures; no CPU neural fallback or efficacy evidence."""
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric
from experiments.spectral_reachability.reacher_d2p_goal_metric_compiler import weighted_terminal, compile_programs, flat_controls
from experiments.spectral_reachability.reacher_d2p_masked_programs import fit


def test_weighting_matches_vector_batch_and_preserves_action_gradient():
    metric = GoalEncoderMetric(torch.diag(torch.tensor([.01,1.],dtype=torch.float64)))
    matrix = torch.tensor([[1.,2.,3.],[2.,-1.,.5]],dtype=torch.float64)
    vector = weighted_terminal(lambda x:matrix@x,metric)
    batch = weighted_terminal(lambda x:x@matrix.T,metric)
    action = torch.tensor([.2,-.1,.3],dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(batch(action[None])[0],vector(action))
    jacobian = torch.autograd.functional.jacobian(vector,action)
    torch.testing.assert_close(jacobian,metric.sqrt_matrix@matrix)


def test_fixed_metric_is_actually_used_by_action_optimization():
    # One actuator cannot satisfy both targets. Changing only the score after
    # Euclidean optimization leaves u=0; consistent search changes the action.
    terminal = lambda u:torch.stack((u[0],u[0]))
    target = torch.tensor([1.,-1.],dtype=torch.float32)
    initial = torch.zeros(1,dtype=torch.float32)
    euclidean,_ = fit(terminal,initial,target)
    metric = GoalEncoderMetric(torch.diag(torch.tensor([.01,100.],dtype=torch.float64)))
    weighted,_ = fit(weighted_terminal(terminal,metric),initial,metric.apply(target.double()))
    assert abs(float(euclidean[0])) < 1e-6
    assert float(weighted[0]) > .99
    assert weighted.dtype == initial.dtype


def test_neural_compiler_rejects_cpu_before_touching_model():
    metric = GoalEncoderMetric(torch.eye(192,dtype=torch.float64))
    with pytest.raises(ValueError,match="CPU fallback forbidden"):
        compile_programs(object(),torch.zeros(1,3,192),torch.zeros(1,3,10),torch.zeros(192),
            metric=metric,task="full_drive",mean=None,std=None,seed=1)


def test_partial_or_nonfinite_terminal_outputs_rejected():
    metric = GoalEncoderMetric(torch.eye(3,dtype=torch.float64))
    for terminal in (lambda x:x[:2],lambda x:x*float("nan")):
        with pytest.raises(ValueError): weighted_terminal(terminal,metric)(torch.zeros(3))


def test_full_flat_control_adapter_preserves_existing_three_result_contract():
    matrix = torch.tensor([[1.,.3,-.2],[.2,-.4,1.]],dtype=torch.float64)
    target = torch.tensor([.8,-.6],dtype=torch.float64)
    metric = GoalEncoderMetric(torch.diag(torch.tensor([.01,1.],dtype=torch.float64)))
    single = weighted_terminal(lambda x:matrix@x,metric)
    batch = weighted_terminal(lambda x:x@matrix.T,metric)
    weighted_target = metric.apply(target)
    methods = flat_controls(single,batch,torch.zeros(3,dtype=torch.float64),weighted_target,seed=4)
    assert set(methods) == {"flat_gn","flat_cem_gn_incumbent"}
    gn, _ = methods["flat_gn"]
    cem, receipt = methods["flat_cem_gn_incumbent"]
    assert float((single(cem)-weighted_target).square().mean()) <= float((single(gn)-weighted_target).square().mean())
    assert receipt["physical_labels_used"] is False
    assert bool((cem.abs()<=1).all())
