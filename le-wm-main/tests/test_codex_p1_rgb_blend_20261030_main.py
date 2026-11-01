import numpy as np

from experiments.spectral_reachability import codex_p1_rgb_blend_20261030_main as blend


def test_blend_uses_model_and_all_five_pulses(monkeypatch):
    correction = np.zeros((5, 2), dtype="<f4")
    correction[0] = [0.2, -0.2]
    monkeypatch.setattr(blend.old, "decide", lambda *args: {
        "decision": "rgb_pusher_correction", "packet": correction, "packet_sha256": "", "features": {}})
    model = np.full((5, 2), 0.4, dtype="<f4")
    result = blend.decide(None, None, model)
    np.testing.assert_array_equal(result["packet"], np.float32(0.5) * model +
                                  np.float32(0.5) * np.repeat(correction[:1], 5, axis=0))
    assert np.all(np.abs(result["packet"]) <= 1)


def test_rejection_and_model_packet_are_unchanged(monkeypatch):
    for decision, packet in (("rgb_rejected", None), ("model_packet", np.zeros((5, 2), dtype="<f4"))):
        item = {"decision": decision, "packet": packet}
        monkeypatch.setattr(blend.old, "decide", lambda *args: item)
        assert blend.decide(None, None, None) is item
