from __future__ import annotations

import numpy as np
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v7_task_contrast as v7


def test_history_ranker_can_reverse_candidate_order_between_histories() -> None:
    model = v7.HistoryTaskContrastRanker(history_dim=1, action_dim=2, hidden=2).eval()
    with torch.no_grad():
        # This is a constructive interaction check: the same action pair has
        # opposite scalar ordering under two different history states.
        model.history_encoder[0].weight.zero_(); model.history_encoder[0].bias.zero_()
        model.history_encoder[0].weight[0, 0] = 1.0
        for parameter in model.history_gru.parameters(): parameter.zero_()
        model.history_gru.weight_ih_l0[4, 0] = 1.0
        model.history_gru.bias_ih_l0[2:4].fill_(-20.0)
        for parameter in model.candidate_encoder.parameters(): parameter.zero_()
        # Candidate encoding now intentionally uses a manually unrolled
        # GRUCell so CUDA does not dispatch to the incompatible fused cuDNN
        # RNN path.  Configure that executed cell, rather than the removed
        # nn.GRU compatibility surface.
        model.candidate_encoder.cell.weight_ih[4, 0] = 1.0
        model.candidate_encoder.cell.bias_ih[2:4].fill_(-20.0)
        model.film.weight.zero_(); model.film.bias.zero_(); model.film.weight[0, 0] = 10.0
        for layer in model.head:
            if isinstance(layer, torch.nn.Linear): layer.weight.zero_(); layer.bias.zero_()
        model.head[0].weight[0, 0] = 1.0
        model.head[2].weight[0, 0] = 1.0
        candidates = torch.zeros(2, 20, 2); candidates[0, :, 0] = 1.0; candidates[1, :, 0] = -1.0
        positive = torch.tensor([[[1.0]]]).repeat(2, 8, 1)
        negative = torch.tensor([[[-1.0]]]).repeat(2, 8, 1)
        positive_scores, negative_scores = model(positive, candidates), model(negative, candidates)
    assert int(torch.argmin(positive_scores)) != int(torch.argmin(negative_scores))


def test_pairwise_rank_loss_prefers_true_lower_cost_order() -> None:
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    groups = [np.asarray([0, 1]), np.asarray([2, 3])]
    correct = v7._rank_loss(torch.tensor([0.0, 2.0, -1.0, 1.0]), target, groups)
    inverted = v7._rank_loss(torch.tensor([2.0, 0.0, 1.0, -1.0]), target, groups)
    assert float(correct) < float(inverted)


def test_pairwise_rank_loss_ignores_exact_task_ties() -> None:
    loss = v7._rank_loss(torch.tensor([4.0, -3.0]), torch.tensor([1.0, 1.0]), [np.asarray([0, 1])])
    assert float(loss) == 0.0


def test_margin_gate_routes_whole_root_and_never_individual_candidates() -> None:
    rows = [
        {"instance_id": "low", "candidate_id": 0}, {"instance_id": "low", "candidate_id": 1},
        {"instance_id": "high", "candidate_id": 0}, {"instance_id": "high", "candidate_id": 1},
    ]
    recurrent = np.asarray([1.00, 1.01, 0.10, 0.40])
    action = np.asarray([0.20, 0.30, 0.20, 0.30])
    routed = v7._margin_gated_cost(rows, recurrent, action, threshold=0.05)
    assert np.array_equal(routed[:2], action[:2])
    assert np.array_equal(routed[2:], recurrent[2:])


def test_calibration_refuses_refit_after_any_outcome_or_frozen_output(tmp_path) -> None:
    context = tmp_path / "temporal_ood_v6_context.json"
    context.write_text("{}")
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}")
    import pytest
    with pytest.raises(FileExistsError, match="sealed"):
        v7._ensure_fresh_calibration_output(context, frozen)
    frozen.unlink()
    (tmp_path / "temporal_ood_v6_outcome_labels.json").write_text("{}")
    with pytest.raises(FileExistsError, match="held outcome"):
        v7._ensure_fresh_calibration_output(context, frozen)
