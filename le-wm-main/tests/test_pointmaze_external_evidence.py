import numpy as np

from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _balanced_action_risk,
    _intervention_action_blocks,
    _root_key,
)


def test_pointmaze_balanced_action_risk_is_zero_for_exact_effects():
    actual = np.asarray([[[0.2, 0.0], [-0.2, 0.0], [0.0, 0.0]]])
    risk, identifiability, details = _balanced_action_risk(actual.copy(), actual)
    assert risk == 0.0
    assert identifiability == 1.0
    assert details["directional_pair_count"] == 1


def test_external_root_key_includes_window_not_only_episode():
    assert _root_key({"episode_id": 7, "window_start": 0}) != _root_key(
        {"episode_id": 7, "window_start": 5}
    )


def test_intervention_populates_all_action_blocks_in_the_model_clock():
    action = np.asarray([0.3, -0.4])
    blocks = _intervention_action_blocks(action)
    assert blocks.shape == (4, 10)
    assert np.allclose(blocks[0], np.tile(action, 5))
    assert np.allclose(blocks, blocks[0][None, :])
