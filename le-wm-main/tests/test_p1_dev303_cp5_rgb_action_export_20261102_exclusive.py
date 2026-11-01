from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_cp5_rgb_action_export_20261102_exclusive as subject,
)


@pytest.fixture
def evidence():
    report = subject.decode(subject.read_regular(subject.RUN_DIR / "report.json"))
    prep = subject.decode(subject.read_regular(subject.PREP))
    events = subject.verify_journal(subject.read_regular(subject.RUN_DIR / "journal.jsonl"), report)
    return report, prep, events


def test_real_sealed_export_has_both_actual_histories_and_no_admission():
    manifest, arrays = subject.build()
    assert [r["arm"] for r in manifest["roles"]] == list(subject.ARMS)
    assert manifest["P1_admission"] is manifest["independent_confirmation"] is False
    assert manifest["model_or_cuda_executed"] is False
    assert arrays["rgb_chunk_rgb_history"].shape == (3, 224, 224, 3)
    assert arrays["rgb_chunk_raw_action_history"].shape == (10, 2)
    assert not arrays["rgb_chunk_raw_action_history"][:5].any()
    assert arrays["rgb_chunk_raw_action_history"][5:].any()
    assert not arrays["zero_prefix_raw_action_history"].any()


def test_refuses_journal_tamper(evidence):
    report, _, _ = evidence
    raw = subject.read_regular(subject.RUN_DIR / "journal.jsonl")
    with pytest.raises(ValueError, match="journal SHA"):
        subject.verify_journal(raw + b"{}\n", report)
    altered = dict(report, journal_sha256=hashlib.sha256(raw + b"{}\n").hexdigest())
    with pytest.raises(ValueError, match="journal hash chain"):
        subject.verify_journal(raw + b"{}\n", altered)


def test_rejects_actual_action_mismatch(evidence):
    report, prep, events = evidence
    changed = deepcopy(events)
    item = subject.event(changed, "rgb_chunk", "action_observed", "control", 3)
    item["action"][0] = 0.0
    with pytest.raises(ValueError, match="actual action chain"):
        subject.verify_arm(subject.RUN_DIR, "rgb_chunk", report["rows"][0], changed,
                           prep["warmup_rgb_sha256"])


def test_rejects_stale_controller_image(evidence):
    report, prep, events = evidence
    changed = deepcopy(events)
    item = subject.event(changed, "rgb_chunk", "controller_decision", step=3)
    item["proposal"]["rgb_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="RGB decision binding"):
        subject.verify_arm(subject.RUN_DIR, "rgb_chunk", report["rows"][0], changed,
                           prep["warmup_rgb_sha256"])


def test_rejects_wrong_observation_hash(evidence):
    report, prep, events = evidence
    changed = deepcopy(events)
    item = subject.event(changed, "rgb_chunk", "observation", "control", 5)
    item["pixel_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="observation journal/report mismatch"):
        subject.verify_arm(subject.RUN_DIR, "rgb_chunk", report["rows"][0], changed,
                           prep["warmup_rgb_sha256"])


def test_exclusive_export_and_archive_contents(tmp_path):
    output = tmp_path / "export"
    manifest_sha = subject.export(output)
    raw = (output / "manifest.json").read_bytes()
    assert subject.sha(raw) == manifest_sha
    manifest = json.loads(raw)
    archive = output / "rgb_action_history.npz"
    assert manifest["archive_sha256"] == subject.sha(archive.read_bytes())
    with np.load(archive, allow_pickle=False) as data:
        assert set(data.files) == {"goal_rgb", "rgb_chunk_rgb_history",
                                   "rgb_chunk_raw_action_history", "zero_prefix_rgb_history",
                                   "zero_prefix_raw_action_history"}
    with pytest.raises(ValueError, match="exclusive"):
        subject.export(output)


def test_reject_duplicate_json_keys():
    with pytest.raises(ValueError, match="duplicate"):
        subject.decode(b'{"x":1,"x":2}')
