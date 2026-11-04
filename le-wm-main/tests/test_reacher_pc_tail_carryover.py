import pytest
import torch

from experiments.spectral_reachability.reacher_pc_tail_carryover import cem_with_carried_tail


def test_carried_tail_selected_without_extra_evaluations():
    tail = torch.full((1, 2, 3), 0.75)
    batches = []

    def score(batch):
        batches.append(batch.clone())
        return (batch - tail).square().sum(dim=(1, 2))

    result, receipt = cem_with_carried_tail(score, tail, samples=8, steps=2, elite=3, seed=3)
    assert torch.equal(result, tail)
    assert len(batches) == 2 and all(torch.equal(b[1], tail[0]) for b in batches)
    assert receipt["selected_old_tail"]
    assert receipt["forward_rollout_evaluations"] == 16
    assert receipt["model_predict_sample_evaluations"] == 32


def test_invalid_tail_rejected():
    with pytest.raises(ValueError):
        cem_with_carried_tail(lambda x: x, torch.tensor([float("nan")]),
                              samples=8, steps=2, elite=3, seed=0)
