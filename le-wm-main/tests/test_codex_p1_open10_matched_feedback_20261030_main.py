import hashlib

from experiments.spectral_reachability import codex_p1_open10_matched_feedback_20261030_main as run


def test_complete_fixed_actions_match_feedback_plan():
    run.install()
    prepared = run.build()
    assert len(prepared["roster"]) == 24
    for item in prepared["roster"]:
        raw = bytes.fromhex(item["raw10_float32_hex"])
        assert len(raw) == 80
        assert hashlib.sha256(raw).hexdigest() == item["raw10_sha256"]
    assert not prepared["fairness_proven"]
    assert not prepared["P1_admission"]
