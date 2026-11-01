import pytest

from experiments.spectral_reachability import p1_exploratory_rgb_v4_verify_20261101_hypatia as mod


def test_saved_v4_bindings():
    report = mod.build()
    assert len(report["slots"]) == 2 and all(x["steps"] == 10 for x in report["slots"])
    assert report["freshness"] == "unknown"
    assert report["planner_observation_authorized"] is False


def test_journal_sha_tamper_rejected():
    with pytest.raises(ValueError, match="journal SHA"):
        mod.read_journal(mod.v4.OUTPUT / "journal.jsonl", "0" * 64)
