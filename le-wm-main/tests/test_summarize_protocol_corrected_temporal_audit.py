import json

from experiments.empirical_spectral_audit.summarize_protocol_corrected_temporal_audit import summarize


def _report(path, seed, markov, memory4, delay=True):
    payload = {
        "split": {"split_seed": seed, "fit_episode_ids": [1, 2], "outcome_episode_ids": [3]},
        "protocol": {"memory_comparison": {"paired_outcome_current_count": 4}},
        "memory_closure": {"1": {"paired_outcome_rmse": markov}, "4": {"paired_outcome_rmse": memory4}},
        "signal_information": {"reports": {
            "delay": {"lifted_beats_base": delay, "relative_rmse_reduction": 0.2},
            "interaction": {"lifted_beats_base": True, "relative_rmse_reduction": 0.3},
            "time_frequency": {"lifted_beats_base": False, "relative_rmse_reduction": -0.1},
        }},
    }
    path.write_text(json.dumps(payload))


def test_summary_counts_splits_not_windows(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    _report(first, 1, 1.0, 0.8)
    _report(second, 2, 1.0, 1.1)
    report = summarize([first, second])
    assert report["n_splits"] == 2
    assert report["wins"]["paired_memory4"] == 1
    assert report["protocol"]["row_pooling"] is False
