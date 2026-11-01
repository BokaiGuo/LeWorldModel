import numpy as np
import pytest
import torch

from experiments.spectral_reachability.preflight_tworoom_d2p_p0 import (
    _bounded_live_actions,
    _normalize_raw_packets,
    _root_anchored_action_sequence,
    _validated_raw_live_actions,
    _validate_tworoom_action_protocol,
)
from experiments.spectral_reachability.preflight_tworoom_d2p_p1_reconfiguration import (
    _bounded_packets,
)


class _ActionEncoder:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim


class _Model:
    def __init__(self, input_dim: int):
        self.action_encoder = _ActionEncoder(input_dim)


def test_tworoom_packets_preserve_each_executed_time_major_action() -> None:
    raw = torch.tensor([[[0.10, -0.20], [0.30, -0.40], [0.50, -0.60], [0.70, -0.80], [0.90, -1.00]]])
    mean = np.asarray([0.10, -0.20] * 5, dtype=np.float32)
    std = np.asarray([0.20, 0.40] * 5, dtype=np.float32)
    normalized = _normalize_raw_packets(raw, mean, std, action_block=5, action_dim=2)
    assert normalized.shape == (1, 10)
    np.testing.assert_allclose(
        normalized.numpy().reshape(5, 2),
        ((raw.numpy()[0] - mean[:2]) / std[:2]),
    )


def test_tworoom_packets_reject_wrong_step_layout() -> None:
    with pytest.raises(ValueError, match="raw packets"):
        _normalize_raw_packets(torch.zeros((1, 4, 2)), np.zeros(10), np.ones(10), action_block=5, action_dim=2)


def test_tworoom_observed_root_shifts_action_context_to_first_future_packet() -> None:
    history = torch.tensor([[1.0], [2.0], [3.0]])
    future = torch.tensor([[4.0], [5.0]], requires_grad=True)
    sequence = _root_anchored_action_sequence(history, future)
    torch.testing.assert_close(sequence, torch.tensor([[2.0], [3.0], [4.0], [5.0]]))
    sequence[-1].sum().backward()
    torch.testing.assert_close(future.grad, torch.tensor([[0.0], [1.0]]))


def test_tworoom_future_actions_stay_in_exact_environment_box() -> None:
    low = torch.tensor([-1.0, -0.5])
    high = torch.tensor([0.75, 1.0])
    actions = _bounded_live_actions(torch.tensor([[[50.0, -50.0], [0.0, 0.0]]]), low, high)
    assert torch.all(actions <= high + 1.0e-6)
    assert torch.all(actions >= low - 1.0e-6)
    torch.testing.assert_close(actions[0, 1], (low + high) / 2.0)


def test_tworoom_sealed_physical_packets_are_not_tanh_remapped() -> None:
    low, high = torch.full((2,), -1.0), torch.full((2,), 1.0)
    packet = torch.tensor([[[0.75, -0.50], [0.20, -0.10]]])
    torch.testing.assert_close(_validated_raw_live_actions(packet, low, high), packet)
    assert not torch.equal(_bounded_live_actions(packet, low, high), packet)


def test_tworoom_p1_packet_bounds_preserve_every_packet_step() -> None:
    low, high = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
    params = torch.zeros((2, 1, 5, 2))
    params[1, 0, 3, 0] = 50.0
    packets = _bounded_packets(params, low, high)
    assert packets.shape == params.shape
    assert float(packets[1, 0, 3, 0]) > 0.99
    torch.testing.assert_close(packets[0], torch.zeros_like(packets[0]))


def test_tworoom_p1_prefix_recurrence_uses_shifted_packet_history() -> None:
    from experiments.spectral_reachability import preflight_tworoom_d2p_p1_reconfiguration as p1

    class AdditiveModel:
        def action_encoder(self, values: torch.Tensor) -> torch.Tensor:
            return values

        def predict(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return embeddings + actions[..., :1]

    model = AdditiveModel()
    emb = torch.tensor([[[10.0], [20.0], [30.0]]])
    actions = torch.tensor([[[1.0], [2.0], [3.0]]])
    # _advance receives unconstrained compiler coordinates; with [-100, 100]
    # these produce physical packet values 4 and 5 respectively.
    params = torch.atanh(torch.tensor([[[[0.04]], [[0.05]]]]))
    low, high = torch.tensor([-100.0]), torch.tensor([100.0])
    states, action_context = p1._advance(
        model, emb, actions, params, mean=np.zeros(1, dtype=np.float32), std=np.ones(1, dtype=np.float32),
        action_block=1, action_dim=1, low=low, high=high,
    )
    # First prediction sees [a1, a2, prefix0], then the second sees
    # [a2, prefix0, prefix1]; this is the x_t -> x_{t+1} causal shift.
    torch.testing.assert_close(states, torch.tensor([[[30.0], [34.0], [39.0]]]))
    torch.testing.assert_close(action_context, torch.tensor([[[3.0], [4.0], [5.0]]]))


def test_tworoom_checkpoint_and_training_stats_must_agree() -> None:
    mean = np.asarray([0.1, -0.2] * 5, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    assert _validate_tworoom_action_protocol(_Model(10), mean, std, action_dim=2, action_block=5) == 5
    with pytest.raises(ValueError, match="invalid LeWM action protocol"):
        _validate_tworoom_action_protocol(_Model(10), mean, std, action_dim=2, action_block=1)
    position_specific = mean.copy()
    position_specific[2] = 0.3
    with pytest.raises(ValueError, match="position-specific"):
        _validate_tworoom_action_protocol(_Model(10), position_specific, std, action_dim=2, action_block=5)


def test_tworoom_p1_source_supported_packet_bank_preserves_raw_packets() -> None:
    from experiments.spectral_reachability import tworoom_d2p_packet_support as support

    pool = np.asarray([
        [[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        [[1.0, -1.0], [0.5, -0.5], [0.0, 0.0], [-0.5, 0.5], [-1.0, 1.0]],
        [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3], [0.5, 0.4], [0.6, 0.5]],
        [[-0.2, 0.1], [-0.3, 0.2], [-0.4, 0.3], [-0.5, 0.4], [-0.6, 0.5]],
        [[0.7, 0.7], [0.6, 0.6], [0.5, 0.5], [0.4, 0.4], [0.3, 0.3]],
    ], dtype=np.float32)
    low, high = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
    params, receipt = support.support_proposal_parameters(
        pool, proposals=4, horizon=1, seed=7, low=low, high=high,
    )
    packets = _bounded_packets(params, low, high).cpu().numpy()
    assert receipt["proposal_distribution"].startswith("source_locked")
    np.testing.assert_allclose(packets[0], 0.0, atol=1.0e-6)
    for packet in packets[1:, 0]:
        assert any(np.allclose(packet, source, atol=1.0e-6) for source in pool)


def test_tworoom_p1_same_bank_greedy_excludes_raw_zero_and_uses_stable_tiebreak() -> None:
    from experiments.spectral_reachability.run_tworoom_d2p_p1_physical_calibration import _same_bank_greedy_prefix

    pool = np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    compilation = {
        "proposal_source_indices": [[2], [0], [1]],
        "candidate_rows": [
            {"proposal_index": 0, "immediate_goal_cost": -100.0},  # OOD raw-zero is never eligible for this comparator.
            {"proposal_index": 1, "immediate_goal_cost": 2.0},
            {"proposal_index": 2, "immediate_goal_cost": 1.0},
            {"proposal_index": 3, "immediate_goal_cost": 1.0},
        ],
    }
    greedy = _same_bank_greedy_prefix(compilation, pool, prefix_horizon=1, action_block=1, action_dim=1)
    assert greedy["proposal_index"] == 2
    assert greedy["source_indices"] == [0]
    np.testing.assert_allclose(greedy["prefix_raw_packets"], [[[1.0]]])


def test_tworoom_p1_same_bank_greedy_rejects_unsealed_candidate_coverage() -> None:
    from experiments.spectral_reachability.run_tworoom_d2p_p1_physical_calibration import _same_bank_greedy_prefix

    with pytest.raises(ValueError, match="exactly cover"):
        _same_bank_greedy_prefix(
            {"proposal_source_indices": [[0]], "candidate_rows": [{"proposal_index": 1, "immediate_goal_cost": 1.0}]},
            np.asarray([[[0.0]]], dtype=np.float32), prefix_horizon=1, action_block=1, action_dim=1,
        )
