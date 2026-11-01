import hashlib

import numpy as np

from experiments.spectral_reachability import codex_p1_rgb_shape_guard_20261030_main as guard


def test_far_shape_retains_model_and_near_shape_retains_five_pulses(monkeypatch):
    model = np.full((5, 2), 0.1, dtype="<f4")
    pulse = np.full((5, 2), 0.2, dtype="<f4")
    shape = {"value": 9.0}

    def proposed(*_):
        return {"decision": "rgb_pusher_correction", "packet": pulse.copy(),
                "features": {"object_shape_chamfer_pixels": shape["value"]}, "packet_sha256": "old"}

    monkeypatch.setattr(guard.five, "decide", proposed)
    result = guard.decide(None, None, model)
    assert result["decision"] == "model_packet"
    np.testing.assert_array_equal(result["packet"], model)
    assert result["packet_sha256"] == hashlib.sha256(model.tobytes()).hexdigest()
    shape["value"] = guard.SHAPE_LIMIT_PIXELS
    result = guard.decide(None, None, model)
    assert result["decision"] == "rgb_pusher_correction"
    np.testing.assert_array_equal(result["packet"], pulse)


def test_segmentation_rejection_is_preserved(monkeypatch):
    monkeypatch.setattr(guard.five, "decide",
                        lambda *_: {"decision": "rgb_rejected", "packet": None, "features": None})
    assert guard.decide(None, None, None)["decision"] == "rgb_rejected"
