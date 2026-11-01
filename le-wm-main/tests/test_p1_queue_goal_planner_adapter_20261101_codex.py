import gzip
import json

from experiments.spectral_reachability import p1_queue_goal_planner_adapter_20261101_codex as subject


SOURCE = subject.ROOT / "results/p1_queue_goal_blind_cuda_20261101_codex_RUN_DEV01/task.json.gz"
SOURCE_SHA = "c67848964098d703fb0c16c75b8679219ef3960ca7d55d0563b168ee8c361a8d"


def test_adapter_removes_only_development_seed_and_preserves_task():
    original = json.loads(gzip.decompress(SOURCE.read_bytes()))
    adapted = subject.adapt(SOURCE, SOURCE_SHA)
    assert set(original) - set(adapted) == {"development_seed"}
    assert adapted == {key: value for key, value in original.items() if key != "development_seed"}
    assert adapted["task"] == original["task"]
    assert adapted["task_id"] == original["task_id"]
