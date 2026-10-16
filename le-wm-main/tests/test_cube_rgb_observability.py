import numpy as np
import torch

from experiments.audit_cube_rgb_observability import (
    _contact_metrics,
    _largest_effector_centroid,
    _largest_red_centroid,
    _target_red_centroid,
    _vector_metrics,
    root_split,
)
from experiments.run_cube_rgb_visual_servo import (
    GRIPPER_CLOSE_ACTION,
    GRIPPER_OPEN_ACTION,
    _adaptive_pixel_push_action,
)


def test_root_split_keeps_each_root_on_one_side():
    roots = np.repeat(np.arange(8), 3)
    calibration, test = root_split(roots, 0.5, 7)
    assert not np.any(calibration & test)
    assert set(np.unique(roots[calibration])).isdisjoint(set(np.unique(roots[test])))
    assert calibration.sum() + test.sum() == len(roots)


def test_vector_metrics_reports_component_and_frame_rmse():
    target = np.zeros((4, 2), dtype=np.float32)
    prediction = np.ones((4, 2), dtype=np.float32)
    metrics = _vector_metrics(prediction, target, np.asarray([True, False, True, False]))
    assert metrics["test_frame_rmse"] == 1.0
    assert metrics["test_component_rmse"] == [1.0, 1.0]


def test_contact_metrics_reports_balanced_accuracy_and_brier():
    target = np.asarray([0, 1, 0, 1], dtype=np.float32)
    score = np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32)
    metrics = _contact_metrics(score, target, np.asarray([True, True, True, True]))
    assert metrics["test_balanced_accuracy"] == 1.0
    assert metrics["test_brier"] < 0.03


def test_largest_red_centroid_is_normalized_and_ignores_small_components():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[8:12, 20:24] = [160, 70, 90]
    image[2:3, 2:3] = [180, 50, 60]
    feature = _largest_red_centroid(image)
    assert feature[2] == 1.0
    assert 0.4 < feature[0] < 0.7
    assert -0.3 < feature[1] < 0.2


def test_target_red_centroid_prefers_largest_physical_cube_over_marker():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:45, 10:35] = [170, 60, 70]  # physical cube
    image[60:70, 65:75] = [110, 70, 85]  # smaller translucent target marker
    feature = _target_red_centroid(image)
    assert feature[2] == 1.0
    assert -0.8 < feature[0] < -0.3
    assert -0.45 < feature[1] < -0.25


def test_target_red_centroid_ignores_goal_y_reference():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[15:40, 12:37] = [175, 60, 70]  # physical cube
    image[62:72, 67:77] = [115, 72, 86]  # target marker
    without_reference = _target_red_centroid(image)
    with_reference = _target_red_centroid(image, reference_y=0.3)
    np.testing.assert_allclose(with_reference, without_reference)


def test_target_red_centroid_selects_physical_component_for_all_cube_task_layouts():
    # The five task layouts move the cube/marker pair to different image
    # positions.  The selection invariant is component identity, not image y.
    for task_offset in range(5):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        x0 = 15 + task_offset * 22
        y0 = 12 + (task_offset % 3) * 18
        image[y0:y0 + 18, x0:x0 + 18] = [180, 60, 70]
        image[88 - task_offset:96 - task_offset, 120 - task_offset:128 - task_offset] = [112, 70, 84]
        feature = _target_red_centroid(image, reference_y=-0.5 + 0.25 * task_offset)
        expected_x = 2.0 * (x0 + 8.5) / 159.0 - 1.0
        expected_y = 2.0 * (y0 + 8.5) / 119.0 - 1.0
        np.testing.assert_allclose(feature[:2], [expected_x, expected_y], atol=0.03)


def test_largest_effector_centroid_aggregates_purple_working_region():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[8:13, 8:13] = [220, 220, 220]  # brightness seam/artifact: ignored
    image[35:45, 20:30] = [120, 55, 130]
    image[45:55, 60:70] = [130, 60, 140]
    feature = _largest_effector_centroid(image)
    assert feature[2] == 1.0
    assert -0.25 < feature[0] < 0.2
    assert -0.2 < feature[1] < 0.2


def test_largest_effector_centroid_accepts_lower_operating_height():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[60:70, 20:30] = [120, 55, 130]
    image[68:74, 60:70] = [130, 60, 140]
    feature = _largest_effector_centroid(image)
    assert feature[2] == 1.0
    assert -0.25 < feature[0] < 0.2
    assert 0.2 < feature[1] < 0.5


def test_cube_gripper_action_semantics_are_explicit():
    assert GRIPPER_CLOSE_ACTION == 1.0
    assert GRIPPER_OPEN_ACTION == -1.0


def test_adaptive_pixel_push_preserves_or_reverses_the_calibrated_proposal():
    error = np.asarray([-0.4, 0.2], dtype=np.float32)
    mapping = np.eye(2, dtype=np.float32)
    forward = _adaptive_pixel_push_action(error, mapping, 0.5, 1.0)
    reverse = _adaptive_pixel_push_action(error, mapping, 0.5, -1.0)
    np.testing.assert_allclose(reverse, -forward)
