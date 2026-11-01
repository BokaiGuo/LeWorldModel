import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_cp5_tail_response_analysis_20261102_exclusive as analysis


def test_mask_distance_and_connected_components():
    rgb = np.zeros((224, 224, 3), np.uint8)
    rgb[10, 10] = [100, 120, 140]
    rgb[13, 14] = [0, 80, 230]
    measured = analysis.geometry(rgb)
    assert measured["nearest_mask_pixel_distance"] == 5
    assert measured["pusher_centroid_to_object_pixel_distance"] == 5
    assert measured["object_components_4_connected"] == 1
    assert not measured["object_touches_image_edge"]


def test_empty_masks_fail():
    with pytest.raises(ValueError):
        analysis.geometry(np.zeros((224, 224, 3), np.uint8))


def test_projection_is_sum_of_first_three_not_per_step():
    engine = analysis.run.handoff.planner.parent.engine
    bank = np.zeros((1, 2, 5, 2), np.float32)
    bank[0, 0, :3, 0] = [-0.5, 0.7, 0.1]
    bank[0, 1, :, 0] = -1
    assert np.isclose(engine._projection(bank, [1, 0])[0], 0.3)
    assert engine._projection(bank, [1, 0])[0] >= engine.MIN_PROJECTION
    bank[0, 1, :, 0] = 1
    assert np.isclose(engine._projection(bank, [1, 0])[0], 0.3)


def test_real_saved_data():
    report = analysis.analyze()
    assert [arm["arm"] for arm in report["arms"]] == ["rgb_chunk", "zero_prefix"]
    assert all(len(arm["rows"]) == 11 and arm["projection_pass"] for arm in report["arms"])
    assert report["P1_admission"] is False
