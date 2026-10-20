import json

from experiments.gdd_wm.summarize_tworoom_repair_consequence import summarize


def _write(path, frozen, hybrid):
    path.write_text(
        json.dumps({
            "frozen_lewm": {
                "seeds": [1, 2, 3],
                "episode_successes": frozen,
            },
            "hybrid_rgb_fallback": {
                "seeds": [1, 2, 3],
                "episode_successes": hybrid,
            },
        }),
        encoding="utf-8",
    )


def test_repair_summary_pairs_the_declared_baseline_arm(tmp_path):
    baseline = tmp_path / "baseline.json"
    intervention = tmp_path / "intervention.json"
    _write(baseline, [1, 0, 0], [1, 0, 1])
    _write(intervention, [0, 0, 1], [1, 1, 1])

    report = summarize(
        baseline,
        intervention,
        intervention_key="hybrid_rgb_fallback",
        baseline_key="hybrid_rgb_fallback",
    )

    assert report["protocol"]["baseline_key"] == "hybrid_rgb_fallback"
    assert report["success"]["baseline"] == 2
    assert report["success"]["intervention"] == 3
    assert report["success"]["intervention_only"] == 1
    assert report["success"]["baseline_only"] == 0
