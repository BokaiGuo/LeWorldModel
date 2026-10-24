import numpy as np
import pytest
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import (
    _bounded_normalized_actions,
    _normalized_block,
    _normalized_action_bounds,
    _prepare_context,
    _raw_block,
)
from experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress import (
    _normalized_blocks_to_raw_actions,
    _to_raw_actions,
    _validate_reacher_action_protocol,
)


class _ActionEncoder:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim


class _Model:
    def __init__(self, input_dim: int):
        self.action_encoder = _ActionEncoder(input_dim)


def test_reacher_d2p_history_block_normalizes_and_repeats_raw_action():
    mean = np.zeros(10, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    got = _normalized_block(np.asarray([0.25, -0.5], dtype=np.float32), mean, std, 5)
    assert got.shape == (10,)
    assert np.allclose(got.reshape(5, 2), [[0.25, -0.5]] * 5)


def test_reacher_d2p_history_block_rejects_mismatched_statistics():
    try:
        _normalized_block(np.zeros(2, dtype=np.float32), np.zeros(9), np.ones(9), 5)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("mismatched action statistics must fail closed")


def test_reacher_d2p_history_prefix_is_normalized_exactly_once():
    mean = np.asarray([0.25, -0.5] * 5, dtype=np.float32)
    std = np.asarray([0.5, 0.25] * 5, dtype=np.float32)
    raw_history = _raw_block(np.zeros(2, dtype=np.float32), action_block=5)
    assert np.array_equal(raw_history, np.zeros(10, dtype=np.float32))
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    _, _, history = _prepare_context(
        preprocessor,
        [frame, frame, frame],
        frame,
        [raw_history, raw_history, raw_history],
        frame,
        torch.device("cpu"),
    )
    assert np.allclose(history.cpu().numpy()[0, 0], (raw_history - mean) / std)


def test_reacher_d2p_future_actions_map_to_environment_box_not_symmetric_normalized_box():
    mean = np.asarray([0.25, -0.5] * 5, dtype=np.float32)
    std = np.asarray([0.5, 0.25] * 5, dtype=np.float32)
    lower_np, upper_np = _normalized_action_bounds(mean, std, np.asarray([-1.0, -1.0]), np.asarray([1.0, 1.0]))
    lower, upper = torch.as_tensor(lower_np), torch.as_tensor(upper_np)
    params = torch.tensor([[[50.0] * 10, [-50.0] * 10, [0.0] * 10]])
    raw, normalized = _to_raw_actions(params, mean, std, lower, upper, np.asarray([-1.0, -1.0]), np.asarray([1.0, 1.0]))
    assert np.max(raw) <= 1.0 + 1.0e-6
    assert np.min(raw) >= -1.0 - 1.0e-6
    assert np.allclose(raw.reshape(3, 5, 2)[2], 0.0, atol=1.0e-6)
    assert np.allclose(normalized, _bounded_normalized_actions(params, lower, upper).numpy())


def test_reacher_d2p_action_token_round_trip_preserves_time_major_order():
    raw_steps = np.asarray([
        [-0.80, 0.75], [-0.40, 0.35], [0.00, -0.05], [0.45, -0.55], [0.90, -0.95],
    ], dtype=np.float32)
    mean = np.asarray([0.1, -0.2] * 5, dtype=np.float32)
    std = np.asarray([0.5, 0.25] * 5, dtype=np.float32)
    normalizer = ActionNormalizer(mean, std)
    normalized_token = normalizer.transform(raw_steps.reshape(-1)).reshape(1, -1)
    recovered = _normalized_blocks_to_raw_actions(
        normalized_token, mean, std, action_dim=2,
    )
    assert np.allclose(recovered, raw_steps, atol=1.0e-7, rtol=0.0)


def test_reacher_d2p_action_protocol_binds_checkpoint_statistics_and_requested_block():
    mean = np.zeros(10, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    assert _validate_reacher_action_protocol(
        _Model(input_dim=10), mean, std, action_dim=2, action_block=5,
    ) == 5
    with pytest.raises(ValueError, match="invalid LeWM action protocol"):
        _validate_reacher_action_protocol(
            _Model(input_dim=10), mean, std, action_dim=2, action_block=1,
        )
    with pytest.raises(ValueError, match="statistics"):
        _validate_reacher_action_protocol(
            _Model(input_dim=10), np.zeros(8), np.ones(8), action_dim=2, action_block=5,
        )
