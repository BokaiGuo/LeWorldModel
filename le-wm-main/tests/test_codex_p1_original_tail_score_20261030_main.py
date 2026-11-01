from experiments.spectral_reachability import codex_p1_original_tail_score_20261030_main as mod


def test_fixed_roster_and_stopped_status():
    entries = mod.inputs()
    assert len(entries) == 24
    assert sum(common["status"] == "native_stopped" for _, _, common, _ in entries) == 2
    assert all(len(raw) == 80 for _, _, _, raw in entries)


def test_full_roster_action_identity():
    entries = mod.inputs()
    assert [row["index"] for row, _, _, _ in entries] == list(range(24))
    assert all(entry["identity"]["logical_pair"] == row["logical_pair"]
               for row, entry, _, _ in entries)
