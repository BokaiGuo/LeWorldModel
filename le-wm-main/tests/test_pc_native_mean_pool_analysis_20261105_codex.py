import json

import pytest

from experiments.pc_native_mean_pool_analysis_20261105_codex import analyze


def make_run(root, mode, *, first_solve=0, count=50):
    stem = f"pool_{mode}"
    directory = root / stem
    directory.mkdir()
    successes = [mode == "mean"] + [True] * (count - 1)
    (directory / "eval_metrics.json").write_text(json.dumps({"episode_successes": successes}))
    record = {
        "solve": first_solve,
        "mean_actions": [[[0.0]]] * count,
        "sample_actions": [[[1.0]]] * count,
        "mean_costs": [1.0] * count,
        "sample_costs": [0.5] * count,
        "choose_sample": [mode == "joint"] * count,
    }
    (root / f"{stem}.jsonl").write_text(json.dumps(record) + "\n")


def test_paired_outcomes_and_candidates(tmp_path):
    for mode in ("mean", "sample", "joint"):
        make_run(tmp_path, mode)
    result = analyze(tmp_path, "pool_{mode}")
    assert result["successes"] == {"mean": 50, "sample": 49, "joint": 49}
    assert result["paired_outcomes"]["mean_vs_joint"]["left_only"] == 1
    assert result["first_solve_joint_chooses_sample"] == 50
    assert result["first_solve_comparison"]["sample"]["sample_actions"]["exact"]


def test_incomplete_run_fails_closed(tmp_path):
    for mode in ("mean", "sample", "joint"):
        make_run(tmp_path, mode, count=49 if mode == "joint" else 50)
    with pytest.raises(ValueError, match="incomplete"):
        analyze(tmp_path, "pool_{mode}")
