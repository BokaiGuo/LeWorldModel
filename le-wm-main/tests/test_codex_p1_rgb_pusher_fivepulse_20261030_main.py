import numpy as np

from experiments.spectral_reachability import codex_p1_rgb_pusher_fivepulse_20261030_main as trial


def test_five_pulses_and_failure_are_preserved(monkeypatch):
    def corrected(*_args):
        packet = np.zeros((5, 2), dtype="<f4")
        packet[0] = [0.12, 0.16]
        return {"decision": "rgb_pusher_correction", "packet": packet, "packet_sha256": "old"}
    monkeypatch.setattr(trial.old, "decide", corrected)
    result = trial.decide(None, None, None)
    np.testing.assert_allclose(result["packet"], np.tile([0.12, 0.16], (5, 1)))
    assert result["packet_sha256"] != "old"
    monkeypatch.setattr(trial.old, "decide", lambda *_: {"decision": "rgb_rejected", "packet": None})
    assert trial.decide(None, None, None)["packet"] is None
