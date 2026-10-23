"""Regression contract for V5 FiLM-conditioned candidate ranking."""

from __future__ import annotations

import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v5_recurrent as v5


def test_film_can_reverse_two_candidate_ranking_between_histories() -> None:
    """The actual V5 modules can make action ranking depend on GRU history.

    This fixes every parameter to a transparent constructive case.  The two
    candidates are evaluated in one batch for each history, and their distance
    to the origin is the ranking cost.  An additive history term alone would
    cancel from the candidate-effect difference; the FiLM multiplication makes
    candidate 1 best after the positive history and candidate 0 best after the
    negative history.
    """
    model = v5.RecurrentModulatedEffect(latent_dim=1, candidate_dim=2, hidden=2).eval()
    with torch.no_grad():
        encoder = model.encoder[0]
        encoder.weight.zero_(); encoder.bias.zero_(); encoder.weight[0, 0] = 1.0
        for parameter in model.gru.parameters():
            parameter.zero_()
        # GRU gates are reset, update, new.  Make the final state approximately
        # tanh(encoded history), with negligible carry-over from the zero state.
        model.gru.weight_ih_l0[4, 0] = 1.0
        model.gru.bias_ih_l0[2:4].fill_(-20.0)
        model.memory_effect.weight.zero_(); model.memory_effect.bias.zero_()
        action_encoder = model.action_encoder[0]
        action_encoder.weight.copy_(torch.eye(2)); action_encoder.bias.zero_()
        model.modulation.weight.zero_(); model.modulation.bias.zero_()
        model.modulation.weight[0, 0] = 1.0
        model.modulation.weight[1, 0] = -1.0
        model.action_effect.weight.zero_(); model.action_effect.bias.zero_()
        model.action_effect.weight[0] = torch.tensor([1.0, -1.0])

        positive_history = torch.tensor([[[1.0, 0.0, 0.0]]]).repeat(2, 1, 1)
        negative_history = torch.tensor([[[-1.0, 0.0, 0.0]]]).repeat(2, 1, 1)
        candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positive_cost = model(positive_history, candidates).square().sum(dim=1)
        negative_cost = model(negative_history, candidates).square().sum(dim=1)

    assert int(torch.argmin(positive_cost)) == 1
    assert int(torch.argmin(negative_cost)) == 0
