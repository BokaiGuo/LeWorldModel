"""No-neural checks of the sealed opposite-side target's evidence interface."""
import json

import pytest

from experiments.spectral_reachability import p1_reverse_target_cuda_20261101_main as target


def test_sealed_target_has_no_witness_action_payload():
    entries, sources = target.collect()
    assert len(entries) == 1
    assert set(entries[0]) == {
        "root", "root_state", "goal_state", "pixels", "pixel_sha256",
        "pixels_uint8_base64", "task_eligibility", "privileged_task_generation",
    }
    assert all(entries[0]["task_eligibility"].values())
    assert str(target.DIRECTORY / "report.json") in sources


def test_changed_report_hash_is_rejected(monkeypatch):
    monkeypatch.setattr(target, "REPORT_SHA", "0" * 64)
    with pytest.raises(Exception, match="fixed eligible witness"):
        target.collect()


def test_missing_trajectory_step_cannot_pass_eligibility():
    report = json.loads((target.DIRECTORY / "report.json").read_bytes())
    states = [row["state"] for row in report["rows"]]
    assert all(target.witness.eligibility(states).values())
    assert target.witness.eligibility(states[:-1]) == {"complete20": False}
