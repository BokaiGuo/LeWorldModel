import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_rgb_task_score_20261030_v1 as rgb


def image():
    a = np.full((224, 224, 3), 255, np.uint8)
    a[40:50, 40:50] = rgb.BLUE
    a[100:120, 80:90] = rgb.BLOCK
    return a


def test_identity_and_pusher_shift_separate_object():
    goal = image()
    current = goal.copy()
    current[40:50, 40:50] = 255
    current[40:50, 50:60] = rgb.BLUE
    same = rgb.score(goal, goal)
    moved = rgb.score(current, goal)
    assert same["pusher_centroid_distance_pixels"] == 0
    assert same["object_shape_chamfer_pixels"] == 0
    assert moved["pusher_centroid_distance_pixels"] == 10
    assert moved["object_shape_chamfer_pixels"] == 0
    assert moved["native_success"] is None


def test_object_shift_separate_pusher():
    goal = image()
    current = goal.copy()
    current[100:120, 80:90] = 255
    current[100:120, 90:100] = rgb.BLOCK
    result = rgb.score(current, goal)
    assert result["pusher_centroid_distance_pixels"] == 0
    assert result["object_shape_chamfer_pixels"] > 0


def test_reject_missing_clipped_and_fragmented():
    goal = image()
    for variant in ("missing", "clipped", "fragmented"):
        current = goal.copy()
        current[40:50, 40:50] = 255
        if variant == "clipped":
            current[0:10, 40:50] = rgb.BLUE
        if variant == "fragmented":
            current[40:45, 40:45] = rgb.BLUE
            current[60:65, 60:65] = rgb.BLUE
        with pytest.raises(ValueError):
            rgb.score(current, goal)
    with pytest.raises(ValueError):
        rgb.score(goal.astype(float), goal)
