from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_backtrack_observed_adapter_20261030_v1 as adapter


@pytest.fixture(scope="module")
def source_case():
    replay = adapter.saved.replay
    report = replay.one.saved.read_verified(replay.one.source.REPORT, replay.one.source.REPORT_SHA)
    prepared = replay.one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    task = prepared["requests"][0]["task"]
    path = next(p for p, h in prepared["bindings"].items() if h == task["binding"]["task_sha256"])
    export = adapter.export_load(path, task["binding"]["task_sha256"])
    physical = json.loads(adapter.saved.REPORT.read_text())
    prep = json.loads(Path(physical["execution_binding"]["prepared"]).read_text())
    plan = physical["rows"][0]["plan"]
    cp = physical["rows"][0]["checkpoints"][1]
    images = [None, dict(step=5, path=str(adapter.saved.RUN / cp["png"]),
                         file_sha256=cp["file_sha256"], pixel_sha256=cp["pixel_sha256"])]
    row = dict(**plan, images=images, warmup_raw_float32_hex=prep["task_rows"][0]["warmup_raw_float32_hex"],
               model_history=task["root_context"]["latent_history"],
               action_history=task["root_context"]["action_history"],
               action_mean=task["action_mean"], action_std=task["action_std"])
    policy = Path(prepared["policy"])
    hashes = {n: adapter.saved.sha(policy/n) for n in ("weights.pt", "config.json")}
    return task, export, row, report["model_before"], hashes


def test_actual_export_adapts_without_relabeling_contract(source_case):
    task, export, row, before, hashes = source_case
    result = adapter.adapt(task, export, row, before, hashes)
    assert result["task"] == export["task"]
    assert result["original_task_contract_sha256"] != result["compact_contract_binding_sha256"]
    assert len(result["prefix"]) == 5 and len(result["original_suffix10"]) == 10
    assert set(result["cp5"]) == {"step", "path", "file_sha256", "pixel_sha256"}
    assert "native_status" not in result


@pytest.mark.parametrize("mutation", ["task_goal", "export_history", "warm", "model", "cp5", "normalizer"])
def test_tampering_is_rejected(source_case, mutation):
    task, export, row, before, hashes = deepcopy(source_case)
    if mutation == "task_goal":
        task["goal"][0] += 1
    elif mutation == "export_history":
        export["task"]["root_raw_action_history"][0][0] += .1
    elif mutation == "warm":
        row["warmup_raw_float32_hex"] = "00" * 80
    elif mutation == "model":
        hashes["weights.pt"] = "0" * 64
    elif mutation == "cp5":
        row["images"][1]["pixel_sha256"] = "0" * 64
    else:
        task["action_std"][0] += 1
    with pytest.raises((ValueError, AssertionError, KeyError, TypeError)):
        adapter.adapt(task, export, row, before, hashes)


def test_bound_export_rejects_wrong_hash(source_case):
    task, _, _, _, _ = source_case
    replay = adapter.saved.replay
    report = json.loads(replay.one.source.REPORT.read_text())
    prepared = json.loads(Path(report["prepared"]).read_text())
    path = next(p for p, h in prepared["bindings"].items() if h == task["binding"]["task_sha256"])
    with pytest.raises((ValueError, AssertionError)):
        adapter.export_load(path, "0" * 64)
