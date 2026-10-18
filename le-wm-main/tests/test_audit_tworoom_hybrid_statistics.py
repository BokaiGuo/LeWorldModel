import json

from experiments.audit_tworoom_hybrid_statistics import audit


def test_paired_statistics_preserve_ties_and_compute_sign_test(tmp_path):
    payload = {
        "frozen_lewm": {"episode_successes": [0, 1, 0, 1]},
        "rgb_reactive": {"episode_successes": [1, 1, 0, 0]},
        "hybrid_rgb_fallback": {"episode_successes": [1, 1, 1, 1]},
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = audit([path])
    hybrid = result["comparisons"]["hybrid_minus_frozen"]
    assert hybrid["wins"] == 2
    assert hybrid["losses"] == 0
    assert hybrid["ties"] == 2
    assert hybrid["exact_two_sided_sign_flip_p"] == 0.5
    assert len(hybrid["paired_bootstrap_ci95"]) == 2
