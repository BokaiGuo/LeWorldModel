import torch

from experiments.theory_spectral_transfer.action_interventions import (
    intervene_future_action_blocks,
    predictor_action_prefix_length,
    predictor_action_source_indices,
)


def test_first_prediction_consumes_last_observed_action_block():
    assert predictor_action_source_indices(3, 4) == [2, 3, 4, 5]
    assert predictor_action_prefix_length(3) == 2


def test_future_intervention_changes_h1_source_but_preserves_earlier_history():
    blocks = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
    modified = intervene_future_action_blocks(blocks, "future_zero", preserve_prefix=2)
    assert torch.equal(modified[:2], blocks[:2])
    assert torch.equal(modified[2:], torch.zeros_like(blocks[2:]))
    # With a three-frame visual context, block 2 is the action for h=1.
    assert modified[2].item() != blocks[2].item()


def test_source_aligned_export_has_one_block_per_horizon():
    indices = predictor_action_source_indices(3, 8)
    assert len(indices) == 8
    assert indices[-1] == 9
