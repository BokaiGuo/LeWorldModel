"""Saved development chain verification, not a scientific admission test."""
import pytest

from experiments.spectral_reachability import p1_dev303_rgb_chunk_saved_20261102_exclusive as m

PREP = m.run.ROOT / "results/p1_dev303_rgb_chunk_diagnostic_20261102_exclusive_PREP_DEV01.json"
OUTPUT = m.run.ROOT / "results/p1_dev303_rgb_chunk_diagnostic_20261102_exclusive_RUN_DEV01"
PREP_SHA = "668232ba7891bd447fd5ebfb222627c29c587b0664642cbd83957a85670abeb5"
REPORT_SHA = "694faea5854ee2d5b558fb9b10530aa5ecb20865cf5d2ce27473ec5c2c417625"


def test_saved_chain_replays_and_stays_development_only():
    if not PREP.exists() or not OUTPUT.exists():
        pytest.skip("sealed development artifacts unavailable")
    result = m.verify(PREP, PREP_SHA, OUTPUT, REPORT_SHA)
    assert result["status"] == "saved_feedback_chain_verified"
    assert result["P1_admission"] is False
    assert [row["control_actions"] for row in result["arms"]] == [5, 5]


def test_external_report_sha_cannot_be_substituted():
    if not PREP.exists() or not OUTPUT.exists():
        pytest.skip("sealed development artifacts unavailable")
    with pytest.raises(ValueError, match="report SHA"):
        m.verify(PREP, PREP_SHA, OUTPUT, "0" * 64)


def test_decision_replay_detects_controller_drift(monkeypatch):
    if not PREP.exists() or not OUTPUT.exists():
        pytest.skip("sealed development artifacts unavailable")
    original = m.run.control.RGBChunkApproach.observe

    def altered(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result["action_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(m.run.control.RGBChunkApproach, "observe", altered)
    with pytest.raises(ValueError, match="RGB decision replay"):
        m.verify(PREP, PREP_SHA, OUTPUT, REPORT_SHA)
