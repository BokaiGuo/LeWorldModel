import pytest
import torch

from experiments.spectral_reachability.run_reacher_pc_full_cem_feedback_development import first_packet_from_plan


def test_first_packet_preserves_complete_plan_and_counts():
    plan = torch.arange(30).reshape(1, 3, 10)
    packet, receipt = first_packet_from_plan(plan, {"forward_rollout_evaluations": 192})
    assert torch.equal(packet, plan[:, :1])
    assert receipt["full_pre_feedback_plan_params"] == plan.tolist()
    assert receipt["predict_samples_horizon_weighted"] == 576
    packet.fill_(0)
    assert plan[0, 0, 1] == 1


def test_rejects_wrong_horizon():
    with pytest.raises(ValueError):
        first_packet_from_plan(torch.zeros(1, 2, 10), {"forward_rollout_evaluations": 1})
