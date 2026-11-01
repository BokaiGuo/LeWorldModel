import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_rgb_pusher_ablation_20261030_main as ablation


def test_correction_uses_current_rgb_features_not_object_alignment(monkeypatch):
    monkeypatch.setattr(ablation.launch.rgb_candidate.rgb, "score", lambda _a, _b: {
        "object_shape_chamfer_pixels": 100.0, "pusher_centroid_distance_pixels": 10.0,
        "current_pusher": {"centroid_xy": [0.0, 0.0]}, "goal_pusher": {"centroid_xy": [3.0, 4.0]}})
    result = ablation.decide(None, None, np.zeros((5, 2), dtype="<f4"))
    assert result["decision"] == "rgb_pusher_correction"
    np.testing.assert_allclose(result["packet"][0], [0.12, 0.16])
    np.testing.assert_array_equal(result["packet"][1:], np.zeros((4, 2)))


def test_segmentation_failure_remains_rejected(monkeypatch):
    def rejected(_a, _b):
        raise ValueError("pusher mask fragmented")
    monkeypatch.setattr(ablation.launch.rgb_candidate.rgb, "score", rejected)
    result = ablation.decide(None, None, np.zeros((5, 2), dtype="<f4"))
    assert result["decision"] == "rgb_rejected" and result["packet"] is None
    with pytest.raises(ValueError):
        ablation.decide(None, None, np.ones((5, 2), dtype=np.float64))
