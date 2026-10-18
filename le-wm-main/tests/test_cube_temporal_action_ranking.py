import numpy as np
import torch

from experiments.audit_cube_rgb_observability import (
    extract_color_geometry_temporal_features,
)
from experiments.audit_cube_macro_action_ranking import (
    _features as macro_features,
    _macro_bank,
)
from experiments.audit_cube_temporal_action_ranking import _candidate_bank, _oracle_mask


def test_post_contact_lateral_banks_exclude_non_lateral_actions():
    legacy = _candidate_bank("legacy")
    lateral4 = _candidate_bank("lateral4")
    lateral8 = _candidate_bank("lateral8")
    assert legacy.shape == (8, 5)
    assert lateral4.shape == (5, 5)
    assert lateral8.shape == (9, 5)
    assert np.allclose(lateral4[:, 2:], 0.0)
    assert np.allclose(lateral8[:, 2:], 0.0)
    assert np.allclose(lateral4[0], 0.0)
    assert np.allclose(lateral8[0], 0.0)
    assert np.all(np.linalg.norm(lateral8[:, :2], axis=1) <= 1.0 + 1.0e-6)


def test_cube_shared_candidate_bank_is_fixed_and_in_bounds():
    bank = _candidate_bank()
    assert bank.shape == (8, 5)
    assert np.all(np.abs(bank) <= 1.0)
    assert np.unique(bank, axis=0).shape[0] == bank.shape[0]


def test_oracle_mask_treats_equal_effects_as_equivalent():
    mask = _oracle_mask(np.asarray([1.0, 1.0 + 1.0e-7, 1.1]), 1.0e-6)
    assert mask.tolist() == [True, True, False]


def test_rgb_temporal_geometry_feature_has_fixed_action_conditioned_shape():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:4, 2:4, 0] = 255
    previous = image.copy()
    goal = image.copy()
    action = np.zeros((1, 5), dtype=np.float32)
    previous_action = np.zeros((1, 5), dtype=np.float32)
    features = extract_color_geometry_temporal_features(
        image[None], previous[None], goal[None], action, previous_action, "cpu",
    )["color_geometry_temporal_action"]
    assert features.shape == (1, 22)
    assert np.isfinite(features.numpy()).all()


def test_macro_features_include_rgb_action_frame_interactions():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:4, 2:4, 0] = 255
    goal = np.zeros((8, 8, 3), dtype=np.uint8)
    goal[4:6, 4:6, 0] = 255
    branch = {
        "current_image": image,
        "previous_image": image.copy(),
        "goal_image": goal,
        "previous_action": np.zeros(5, dtype=np.float32),
    }
    features = macro_features([branch], _macro_bank(), "cpu")
    assert features.shape == (5, 47)
    assert torch.isfinite(features).all()
    assert not torch.allclose(features[1, -6:], features[3, -6:])


def test_long_macro_bank_has_predeclared_eight_direction_support():
    bank = _macro_bank(length=60, direction_count=8)
    assert bank.shape == (9, 60, 5)
    assert np.all(np.abs(bank) <= 1.0)
    assert np.any(np.abs(bank[1:, :20, :2]) > 0.0)
    assert np.any(bank[1:, 4:19, 2] < 0.0)
    assert np.any(bank[1:, 19:24, 4] < 0.0)
