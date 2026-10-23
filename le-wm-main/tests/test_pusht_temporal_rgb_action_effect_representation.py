import numpy as np
import pytest

from experiments.audit_pusht_temporal_rgb_action_effect_representation import (
    TemporalTorqueExample,
    assert_root_disjoint,
    build_feature_matrix,
    raw_two_frame_effect_features,
    temporal_difference_history_features,
)


def _example(*, root_id: int, label: int) -> TemporalTorqueExample:
    frames = np.zeros((4, 10, 12, 3), dtype=np.uint8)
    frames[1, 2:5, 2:5, 0] = 25
    frames[2, 2:5, 3:6, 1] = 70
    frames[3, 2:5, 4:7, 2] = 130
    actions = np.asarray([[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0]], dtype=np.float32)
    return TemporalTorqueExample(
        root_id=root_id,
        rgb_frames=frames,
        declared_history_actions=actions,
        offline_mirrored_preference=label,
        offline_preference_margin=0.3,
    )


def test_feature_dimensions_are_fixed_and_include_only_declared_actions():
    item = _example(root_id=7, label=1)
    raw = raw_two_frame_effect_features(item.rgb_frames, item.declared_history_actions, image_size=5)
    temporal = temporal_difference_history_features(item.rgb_frames, item.declared_history_actions, image_size=5)

    # raw: previous/current/difference RGB plus the final declared action.
    assert raw.shape == (3 * 5 * 5 * 3 + 2,)
    # temporal: current RGB plus one RGB difference and action per history step.
    assert temporal.shape == (5 * 5 * 3 * (1 + 3) + 2 * 3,)
    assert np.isfinite(raw).all()
    assert np.isfinite(temporal).all()


def test_temporal_features_do_not_depend_on_offline_privileged_label():
    positive = _example(root_id=17, label=1)
    negative = TemporalTorqueExample(
        root_id=positive.root_id,
        rgb_frames=positive.rgb_frames.copy(),
        declared_history_actions=positive.declared_history_actions.copy(),
        offline_mirrored_preference=0,
        offline_preference_margin=-0.3,
    )
    for representation in ("raw_two_frame", "temporal_difference_history"):
        first = build_feature_matrix([positive], representation=representation, image_size=4)
        second = build_feature_matrix([negative], representation=representation, image_size=4)
        assert np.array_equal(first, second)


def test_root_disjointness_is_enforced_before_a_readout_can_fit():
    train = [_example(root_id=1, label=0), _example(root_id=2, label=1)]
    held = [_example(root_id=3, label=0)]
    assert_root_disjoint(train, held)
    with pytest.raises(ValueError, match="root-disjoint"):
        assert_root_disjoint(train, [_example(root_id=2, label=1)])


def test_temporal_feature_rejects_undeclared_or_misaligned_history():
    item = _example(root_id=31, label=1)
    with pytest.raises(ValueError, match="one action per RGB transition"):
        temporal_difference_history_features(item.rgb_frames, item.declared_history_actions[:2], image_size=4)
