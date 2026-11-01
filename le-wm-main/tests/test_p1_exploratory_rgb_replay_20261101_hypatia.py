from experiments.spectral_reachability import p1_exploratory_rgb_replay_20261101_hypatia as mod


def test_replay_fixed_contract():
    contract = mod.prepare()
    assert contract["slots"] == [0, 1]
    assert contract["with_target"] is False
    assert contract["capture_after_steps"] == [0, 5, 10]
    assert contract["freshness"] == "unknown"
    assert contract["selection_by_outcome"] is False


def test_journal_is_durable_and_not_overwritten(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = mod.Journal(path)
    journal.emit("started")
    journal.emit("finished")
    journal.stream.close()
    assert len(path.read_bytes().splitlines()) == 2
