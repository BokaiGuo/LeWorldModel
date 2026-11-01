import json

from experiments.spectral_reachability import codex_p1_adam_feedback_compare_20261030_main as compare


def test_rebuild_and_censor():
    actual = compare.build()
    saved = json.loads((compare.base.ROOT / f"results/{compare.NAME}_RUN03.json").read_text())
    assert actual["rows"] == saved["rows"]
    assert len(actual["rows"]) == 24
    assert sum(r["open_last_success"] for r in actual["rows"]) == 5
    assert sum(r["feedback_last_success"] for r in actual["rows"]) == 1
    for r in actual["rows"]:
        if r["open_status"] == "native_stopped" or r["feedback_status"] == "native_stopped":
            assert r["feedback_minus_open"] is None
