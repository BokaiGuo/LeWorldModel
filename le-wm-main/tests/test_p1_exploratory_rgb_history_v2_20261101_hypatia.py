import json

from experiments.spectral_reachability import p1_exploratory_rgb_history_v2_20261101_hypatia as mod


def test_journal_writes_complete_bytes_and_chains(tmp_path):
    journal = mod.Journal(tmp_path / "events.jsonl")
    journal.emit("started", slot=0)
    journal.emit("completed", slot=0)
    journal.stream.close()
    lines = [json.loads(line) for line in (tmp_path / "events.jsonl").read_bytes().splitlines()]
    assert lines[0]["previous_sha256"] is None
    assert lines[1]["previous_sha256"] == lines[0]["sha256"]
    assert lines[1]["index"] == 1


def test_v2_prepare_still_preserves_fixed_design():
    prepared = mod.prepare()
    assert prepared["slots"] == [0, 1]
    assert prepared["capture_after_steps"] == [0, 5, 10]
    assert prepared["freshness"] == "unknown"
