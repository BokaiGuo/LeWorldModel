"""CPU linear algebra fixtures only; live encoder audit strictly host CUDA."""
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric, RIDGE


def test_full_goal_metric_matches_independent_inverse_and_preserves_input():
    generator=torch.Generator().manual_seed(41)
    j=torch.randn((7,12),generator=generator,dtype=torch.float64)
    gram=j@j.T
    saved=gram.clone()
    metric=GoalEncoderMetric(gram)
    expected=RIDGE*torch.linalg.inv(gram+RIDGE*torch.eye(7,dtype=torch.float64))
    torch.testing.assert_close(metric.sqrt_matrix.T@metric.sqrt_matrix,expected,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(gram,saved,rtol=0,atol=0)


def test_zero_encoder_response_keeps_identity_in_unobserved_directions():
    metric=GoalEncoderMetric(torch.zeros((4,4),dtype=torch.float64))
    x=torch.arange(4,dtype=torch.float64)
    torch.testing.assert_close(metric.apply(x),x)


def test_same_goal_metric_applies_identically_to_different_action_masks():
    metric=GoalEncoderMetric(torch.eye(4,dtype=torch.float64))
    x=torch.randn(4,dtype=torch.float64)
    full=torch.stack((x,x,x),dim=1)
    torch.testing.assert_close(metric.apply(full)[:,0],metric.apply(x))


@pytest.mark.parametrize("bad",[torch.eye(3),-torch.eye(3,dtype=torch.float64),
    torch.full((3,3),float("nan"),dtype=torch.float64),torch.ones((3,2),dtype=torch.float64)])
def test_invalid_goal_gram_fails_closed(bad):
    with pytest.raises(ValueError): GoalEncoderMetric(bad)
