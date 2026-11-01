import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p2_cp15_capture_builder_20261102_codexexclusive as build
from experiments.spectral_reachability import p2_shared_pair_saved_20261102_codexexclusive as saved
from experiments.spectral_reachability import p2_shared_rgb_pair_wiring_20261102_codexexclusive as wire


def test_real_source_capture_and_exact_receipts(tmp_path):
    out = tmp_path / "capture"
    digest = build.create(out)
    result = build.verify(out, digest)
    assert result["status"] == "verified_saved_known_root_source"
    bundle, frames, goal, actions = wire.load_capture(out / "capture.json", digest)
    assert [r["step"] for r in bundle["frames"]] == [5, 10, 15]
    assert actions.shape == (15, 2) and len(frames) == 3
    assert wire.pixel_sha(goal) == bundle["source"]["goal_pixel_sha256"]
    assert bundle["source"]["fresh_root"] is False
    assert result["P2_admission"] is False
    with pytest.raises(FileExistsError):
        build.create(out)


def test_capture_tamper_fails(tmp_path):
    out = tmp_path / "capture"
    digest = build.create(out)
    p = out / "frame_05.png"
    p.write_bytes(p.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        build.verify(out, digest)


def test_saved_consumer_rejects_partial_and_wrong_sha(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    report = {"schema": wire.NAME + ".report.v1", "status": "partial_or_failed"}
    p = output / "report.json"
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        saved.verify_report(p, hashlib.sha256(p.read_bytes()).hexdigest(), "a" * 64)
    with pytest.raises(ValueError):
        saved.verify_report(p, "0" * 64, "a" * 64)
