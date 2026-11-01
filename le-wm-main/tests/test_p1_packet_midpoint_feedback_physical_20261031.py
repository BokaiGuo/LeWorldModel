import hashlib

import pytest

from experiments.spectral_reachability import p1_packet_midpoint_feedback_physical_20261031 as m


def test_guard_rejects_midpoint_before_second_packet(monkeypatch):
    base = m.physical.engine.h10.base
    pixels = b"pixels"
    good = {"pixel_sha256": hashlib.sha256(pixels).hexdigest(), "state": [1], "snapshot": {"x": 1}}
    entered_second = []

    def capture(env, obs, phase, step, row, meter, store, reference=None):
        row["trajectory"].append(dict(good, state=[2]))
        return pixels

    monkeypatch.setattr(base, "capture", capture)

    def fake_execute(anchor, item, data, path, factory):
        row = {"trajectory": [], "responses": []}
        base.capture(None, None, "probe", 5, row, None, None)
        entered_second.append(True)
        return {"row": row}

    with pytest.raises(ValueError):
        m.guarded(fake_execute, {}, {"midpoint": good}, None, None, None)
    assert not entered_second
    assert base.capture is capture
