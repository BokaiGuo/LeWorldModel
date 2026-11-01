from experiments.spectral_reachability import p1_queue_goal_planner_adapter_dev02_20261101_codex as subject


def test_dev02_task_adapts_without_changing_task_identity():
    source = subject.ROOT / "results/p1_queue_goal_blind_cuda_dev02_20261101_codex_RUN_MAIN01/task.json.gz"
    adapted = subject.adapt(source, "8a97b245d379f38f1e17b8d9c2fcc1fb65dcb1b6bcd8f302b0ce2124d7c19990")
    original = subject.base.read_gzip(source)
    assert adapted["task_id"] == original["task_id"]
    assert adapted["task"] == original["task"]
    assert "development_seed" not in adapted
