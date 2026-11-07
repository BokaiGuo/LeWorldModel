import pytest
import torch

from experiments.pc_rgb_first_decision_temporal_score_20261107_codex import arm_indices, score_sequences


def test_temporal_excludes_initial_and_preserves_packet_order():
    predicted = torch.tensor([[[[0.], [5.], [4.], [3.], [2.], [1.]]]])
    costs = score_sequences(predicted, torch.tensor([[0.]]))
    torch.testing.assert_close(costs, torch.tensor([[[25., 16., 9., 4., 1.]]]))


def test_arm_indices_preserve_retained_order():
    assert arm_indices([2, 0]) == list(range(12)) + [16, 17, 12, 13]
    with pytest.raises(ValueError):
        arm_indices([1, 1])
