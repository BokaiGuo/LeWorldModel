import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_response_midpoint_physical_20261031 as physical


def test_prepared_real_roster_matches_fixed_plan():
    path = physical.ROOT / "results/p1_response_midpoint_physical_20261031_prepare_RUN01.json"
    assert physical.sha(path) == "3d70d2feb2a05ff8241c63a4c00cf3a5d4f5a73d6e5450f5a4f5f2bf37541cf4"
    prepared = json.loads(path.read_text())
    assert prepared == physical.build()
    assert len(prepared["anchors"]) == len(prepared["roster"]) == 13
    assert sorted(row["index"] for row in prepared["excluded"]) == [1, 3, 5]
    assert prepared["P1_admission"] is False


def test_guard_prevents_second_packet_on_midpoint_mismatch(monkeypatch):
    base = physical.prior.engine.h10.base
    engine = physical.prior.engine
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    receipt = {"pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
               "state": [0] * 7, "snapshot": {"identity": 1}}
    phases = []

    def original_capture(env, obs, phase, step, row, meter, store, reference=None):
        row["trajectory"].append(dict(receipt))
        return pixels

    def execute(anchor, item, data, path, factory):
        row = {"trajectory": []}
        base.capture(None, None, "probe", 5, row, None, None)
        phases.append("second_packet_dispatched")
        return {"row": {"responses": [None] * 5}}

    monkeypatch.setattr(base, "capture", original_capture)
    monkeypatch.setattr(engine, "execute", execute)
    item = {"midpoint": dict(receipt, state=[1] * 7)}
    with pytest.raises(Exception, match="midpoint replay mismatch"):
        physical.guarded_execute({}, item, None, Path("/tmp/unused"), None)
    assert phases == []
    assert base.capture is original_capture
    item["midpoint"] = receipt
    physical.guarded_execute({}, item, None, Path("/tmp/unused"), None)
    assert phases == ["second_packet_dispatched"]
    assert base.capture is original_capture
