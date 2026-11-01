import numpy as np
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_rgb_context_support import source_anchors, nearest_context_indices, ranking_metrics


def test_index_is_episode_disjoint_causal_and_reproducible():
    lengths, offsets = np.array([50] * 10), np.arange(10) * 50
    anchors = source_anchors(lengths, offsets, excluded={1, 4}, count=8, seed=7)
    assert anchors == source_anchors(lengths, offsets, excluded={1, 4}, count=8, seed=7)
    episodes = {row["source_episode"] for row in anchors}
    assert len(episodes) == 8 and not (episodes & {1, 4})
    for row in anchors:
        episode, root = row["source_episode"], row["root_row"]
        assert root - 15 >= offsets[episode] and root + 5 < offsets[episode] + lengths[episode]


def test_index_cannot_silently_replace_or_admit_excluded_episodes():
    with pytest.raises(ValueError):
        source_anchors([20, 21, 30], [0, 20, 41], excluded={2}, count=2, seed=1)


def test_temporal_retrieval_not_just_last_rgb_and_stable_ties():
    query = torch.tensor([[0.], [1.], [2.]])
    index = torch.tensor([[[10.], [1.], [2.]], [[0.], [1.], [2.]], [[0.], [1.], [2.]]])
    selected, distances = nearest_context_indices(query, index, count=2)
    assert selected.tolist() == [1, 2]
    assert distances[0] > distances[1]


@pytest.mark.parametrize("count", [0, 4])
def test_invalid_retrieval_count_rejected(count):
    with pytest.raises(ValueError):
        nearest_context_indices(torch.zeros(3, 2), torch.zeros(3, 3, 2), count=count)


def test_ranking_ties_do_not_count_as_success_and_precision_can_be_absent():
    result = ranking_metrics([0, -1, -2], [0, -2, -1])
    assert result["pairwise_strict_order_agreements"] == 2
    assert result["pairwise_strict_order_reversals"] == 1
    assert result["non_greedy_precision"] is None
    assert ranking_metrics([0, 0], [0, 0])["pairwise_agreement_fraction"] == 0


def test_cost_and_zero_relative_error_not_conflated():
    result = ranking_metrics([10, 11], [0, 1])
    assert result["mean_abs_cost_error"] == 10
    assert result["mean_abs_zero_relative_margin_error"] == 0
    assert result["non_greedy_precision"] == 1


def test_audit_aggregation_calls_real_runtime_scalar_bootstrap_api():
    from experiments.spectral_reachability.audit_reacher_d2p_rgb_conditioned_packets import aggregate_banks
    from experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress import _bootstrap_mean
    metrics = ("mean_abs_cost_error", "mean_abs_zero_relative_margin_error", "pairwise_agreement_fraction")
    rows = [{"banks": {name: {"metrics": {metric: value for metric in metrics}} for name, value in (("random", .2), ("retrieved", .1))}} for _ in range(3)]
    result = aggregate_banks(rows, seed=1, bootstrap_mean=_bootstrap_mean)
    assert result["mean_abs_cost_error"]["retrieved_minus_random_ci95"] == pytest.approx([-.1, -.1])
