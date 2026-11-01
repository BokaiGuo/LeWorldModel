import json

from experiments.spectral_reachability import codex_p1_adam_trust_compare_20261030_main as comparison


def test_rebuild_all_pairs_and_stop_censor():
    result = comparison.build()
    saved = json.loads((comparison.ROOT / f"results/{comparison.NAME}_RUN01.json").read_text())
    assert result == saved
    assert len(result["rows"]) == 24
    assert sum(row["open_last_success"] for row in result["rows"]) == 5
    assert sum(row["feedback_last_success"] for row in result["rows"]) == 2
    for row in result["rows"]:
        if row["open_status"] == "native_stopped" or row["feedback_status"] == "native_stopped":
            assert row["feedback_minus_open"] is None
