import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_rgb_contact_feedback_20261101_codex as subject,
)


def _image(pusher_xy=(180, 110), object_xy=(150, 90)):
    image = np.zeros((224, 224, 3), np.uint8)
    px, py = pusher_xy
    ox, oy = object_xy
    image[py - 2 : py + 3, px - 2 : px + 3] = subject.rgb.BLUE
    image[oy - 5 : oy + 6, ox - 8 : ox + 9] = subject.rgb.BLOCK
    return image


def test_rgb_contact_action_points_toward_nearest_object_pixel():
    action, audit = subject.rgb_contact_action(_image())
    assert action.dtype == np.float32
    assert action.shape == (2,)
    assert action[0] < 0 and action[1] < 0
    assert np.linalg.norm(action) <= subject.MAX_RAW_MAGNITUDE + 1e-7
    assert audit["centroid_to_nearest_object_gap_pixels"] > subject.TARGET_GAP_PIXELS
    assert audit["simulator_state_used"] is False


def test_rgb_contact_action_stops_inside_target_gap():
    action, audit = subject.rgb_contact_action(_image(pusher_xy=(160, 90)))
    assert np.array_equal(action, np.zeros(2, np.float32))
    assert audit["centroid_to_nearest_object_gap_pixels"] <= subject.TARGET_GAP_PIXELS


def test_rgb_contact_action_accepts_compact_two_component_moving_sprite():
    image = _image()
    image[108:113, 178:183] = 0
    image[108:113, 184:189] = subject.rgb.BLUE
    action, audit = subject.rgb_contact_action(image)
    assert np.isfinite(action).all()
    assert audit["simulator_state_used"] is False


def test_rgb_contact_action_rejects_widely_fragmented_pusher():
    image = _image()
    image[40:45, 40:45] = subject.rgb.BLUE
    try:
        subject.rgb_contact_action(image)
    except ValueError as exc:
        assert "moving pusher mask geometry" in str(exc)
    else:
        raise AssertionError("widely fragmented pusher must fail closed")


def test_contract_is_development_only_and_preserves_common_tail():
    value = subject.build()
    assert value["designed_after_dev302_open_loop_outcome"] is True
    assert value["development_only"] is True
    assert value["fair_total_cost"] is False
    assert value["independent_confirmation"] is False
    assert value["P1_admission"] is False
    d2p = np.asarray(value["sealed_d2p_actions"], np.float32)
    tail = np.asarray(value["common_tail"], np.float32)
    assert np.array_equal(tail, d2p[subject.PREFIX_STEPS :])
    assert value["rgb_policy"]["simulator_state_input"] is False
    assert value["rgb_policy"]["segmentation_failure"] == "fail_closed"
    assert value["rgb_policy"]["moving_pusher_components_max"] == 2
