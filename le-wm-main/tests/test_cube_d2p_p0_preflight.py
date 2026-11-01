import numpy as np
import pytest

from experiments.spectral_reachability.preflight_cube_d2p_p0 import (
    _jsonable,
    _validate_cube_action_protocol,
)


class _ActionEncoder:
    input_dim = 25


class _Model:
    action_encoder = _ActionEncoder()


def test_cube_checkpoint_and_tiled_statistics_must_agree() -> None:
    mean = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5] * 5, dtype=np.float32)
    std = np.ones(25, dtype=np.float32)
    assert _validate_cube_action_protocol(_Model(), mean, std) == 5
    mean[5] = 9.0
    with pytest.raises(ValueError, match="position-specific"):
        _validate_cube_action_protocol(_Model(), mean, std)


def test_cube_replay_provenance_is_json_safe_without_losing_array_values() -> None:
    value = {"scalar": np.int64(7), "array": np.asarray([1.0, 2.0]), "nested": (np.float32(3.0),)}
    assert _jsonable(value) == {"scalar": 7, "array": [1.0, 2.0], "nested": [3.0]}
