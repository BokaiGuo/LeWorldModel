import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import (
    _candidate_sequence_bank,
    _expand_macro_sequence,
    _normalized_proposals_to_raw_steps,
    _pairwise_inversion,
    _select_deployment_proposals,
    _wrap_angle,
)


def test_pairwise_inversion_is_zero_for_same_order_and_one_for_reverse_order():
    assert _pairwise_inversion(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0])) == 0.0
    assert _pairwise_inversion(np.asarray([0.0, 1.0, 2.0]), np.asarray([2.0, 1.0, 0.0])) == 1.0


def test_reacher_angle_wrap_is_bounded():
    values = _wrap_angle(np.asarray([-4.0, 0.0, 4.0]))
    assert np.all(values >= -np.pi)
    assert np.all(values < np.pi)


def test_compact_sequence_bank_matches_online_support_contract():
    mean = np.zeros(10, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    raw, normalized, labels = _candidate_sequence_bank(
        mean, std, horizon=5, action_block=5, bank_family="compact",
    )
    assert raw.shape == (49, 5, 2)
    assert normalized.shape == (49, 5, 10)
    assert len(labels) == 49
    assert {row["family"] for row in labels} == {
        "constant", "pulse", "pulse_recovery",
    }


def test_macro_sequence_expansion_preserves_duration_and_phase():
    sequence = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    expanded = _expand_macro_sequence(sequence, action_block=3, replay_steps=8)
    assert expanded.shape == (8, 2)
    np.testing.assert_allclose(expanded[:3], [[1.0, 0.0]] * 3)
    np.testing.assert_allclose(expanded[3:], [[0.0, 1.0]] * 5)


def test_normalized_proposal_conversion_preserves_within_block_actions():
    proposal = np.arange(20, dtype=np.float32).reshape(1, 2, 10)
    raw = _normalized_proposals_to_raw_steps(
        proposal, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
        action_block=5,
    )
    assert raw.shape == (1, 10, 2)
    np.testing.assert_allclose(raw.reshape(-1), np.arange(20, dtype=np.float32))


def test_deployment_proposals_exclude_fixed_bank_and_span_cost_quantiles():
    candidates = torch.arange(6 * 2, dtype=torch.float32).reshape(1, 6, 1, 2)
    trace = {
        "candidates": candidates,
        "costs": torch.tensor([[5.0, 99.0, 1.0, 4.0, 2.0, 3.0]]),
        "bank_mask": torch.tensor([[False, True, False, False, False, False]]),
    }
    selected, costs, indices = _select_deployment_proposals(trace, count=3)
    assert selected.shape == (3, 1, 2)
    assert indices.tolist() == [2, 5, 0]
    np.testing.assert_allclose(costs, [1.0, 3.0, 5.0])
