import json

from experiments.summarize_tworoom_failure_decomposition import summarize


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summary_keeps_reachability_and_online_panels_separate(tmp_path):
    oracle = _write(tmp_path / "oracle.json", {
        "metrics": {"root_count": 4, "oracle_available_roots": 3,
                    "oracle_available_rate": 0.75, "zero_success_roots": 1},
    })
    sequence = _write(tmp_path / "sequence.json", {"summary": {
        "mean_spearman": 0.1, "top1_accuracy": 0.25, "top4_accuracy": 0.5,
        "mean_regret": 2.0, "mean_true_rank_of_predicted_best": 2.0,
    }})
    offline = _write(tmp_path / "offline.json", {"summary": {
        "held_out_raw": {"mean_regret": 4.0},
        "held_out_latent_action_ranker": {"mean_regret": 3.0},
        "held_out_delta_ranker_minus_raw_regret": -1.0,
    }})
    panel = _write(tmp_path / "panel.json", {
        "frozen_lewm": {"episode_successes": [0, 1]},
        "rgb_reactive": {"episode_successes": [1, 1]},
        "hybrid_rgb_fallback": {"episode_successes": [1, 1], "risk_gate": {}},
    })
    report = summarize(
        oracle=oracle, sequence_ranker=sequence, offline_ranker=offline,
        hybrid_panels=[panel],
    )
    assert report["planner_reachability_ceiling"]["zero_success_roots"] == 1
    assert report["offline_to_online_gap"]["offline_regret_delta_ranker_minus_raw"] == -1.0
    assert report["online_intervention"]["aggregate"]["hybrid_vs_frozen"] == {
        "left_wins": 1, "right_wins": 0, "ties": 1,
    }
