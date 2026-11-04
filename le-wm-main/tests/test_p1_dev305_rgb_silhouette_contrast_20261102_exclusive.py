import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev305_rgb_silhouette_contrast_20261102_exclusive as target


def image():
    pixels = np.full((224, 224, 3), 255, dtype=np.uint8)
    pixels[50:70, 50:70] = [120, 140, 160]
    pixels[100:110, 100:110] = [50, 100, 230]
    return pixels


def test_score_and_visible_gate():
    a = image()
    assert target.silhouette_iou(a, a) == 1.0
    assert target.screen(a, a)["eligible"]
    b = a.copy()
    b[50:70, 50:70] = 255
    b[50:70, 60:80] = [120, 140, 160]
    assert target.silhouette_iou(a, b) == pytest.approx(1 / 3)
    assert target.centroid_distance(a, b) == pytest.approx(10)


def test_goal_clipping_blocks_even_with_numeric_score():
    goal = image()
    goal[50:70, 50:70] = 255
    goal[210:224, 50:70] = [120, 140, 160]
    assert target.silhouette_iou(goal, goal) == 1
    assert not target.screen(goal, image())["eligible"]


def test_missing_mask_rejected():
    with pytest.raises(ValueError, match="missing object mask"):
        target.silhouette_iou(image(), np.full((224, 224, 3), 255, dtype=np.uint8))


def test_hash_mismatch_rejected(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        target.checked_json(path, "0" * 64)


def test_saved_development_bank():
    report = target.analyze()
    assert not report["goal_visibility"]["eligible"]
    assert not report["selector_eligible"]
    assert [r["n"] for r in report["rows"]] == [289, 289, 289]
    assert all(r["post_outcome_best_iou"] == 0 and r["iou_best_ties"] == 289
               for r in report["rows"])
