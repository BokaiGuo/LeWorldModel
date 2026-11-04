import json
from pathlib import Path

import numpy as np

from experiments import pc_motion_cp5_physical_20261103_codex as physical
from experiments import pc_motion_cp5_replan_20261103_codex as cp5
from experiments import pc_motion_unprivileged_physical_20261103_codex as common


def test_sealed_context_and_replanned_actions():
    sha = cp5.prior.prior.sha
    source = json.loads(cp5.prior.DECISIONS.read_text())
    decision = json.loads(cp5.DECISION.read_text())
    execution = json.loads(cp5.RECEIPT.read_text())
    assert decision["source_sha256"] == sha(cp5.prior.DECISIONS)
    assert decision["context_sha256"] == sha(cp5.CONTEXT)
    assert decision["protocol_sha256"] == sha(cp5.PROTOCOL)
    assert decision["runner_sha256"] == sha(Path(cp5.__file__))
    assert execution["decision_sha256"] == sha(cp5.DECISION)
    assert execution["arrays_sha256"] == sha(cp5.EXECUTION)
    assert len(decision["rows"]) == len(execution["rows"]) == 12
    mean, std = np.asarray(source["action_mean"]), np.asarray(source["action_std"])
    with np.load(cp5.CONTEXT, allow_pickle=False) as contexts, np.load(cp5.EXECUTION, allow_pickle=False) as outputs:
        for row, executed in zip(decision["rows"], execution["rows"], strict=True):
            root, role = row["root_index"], row["role"]
            assert (root, role) == (executed["root_index"], executed["role"])
            assert role in ("selected", "zero", "random_0")
            key = f"root{root}_{role}"
            frames = contexts[key]
            assert len(frames) == 21
            assert cp5.prior.digest_image(frames[20]) == row["cp5_rgb_sha256"]
            raw = np.concatenate((cp5.prior.WARMUP[10:15], row["prefix"]))
            np.testing.assert_allclose(row["history_actions_raw"], raw)
            np.testing.assert_allclose(np.asarray(row["history_normalized"]).reshape(10, 2) * std + mean, raw, atol=1e-6)
            scores = [evaluation["score"] for evaluation in row["evaluations"]]
            assert len(scores) == 100
            assert row["best_index"] == int(np.argmin(scores))
            assert row["model_score"] == scores[row["best_index"]]
            best = np.asarray(row["evaluations"][row["best_index"]]["suffix_normalized"]).reshape(10, 2)
            np.testing.assert_allclose(row["suffix"], best * std + mean, atol=1e-6)
            np.testing.assert_allclose(executed["prefix"], row["prefix"])
            np.testing.assert_allclose(executed["suffix"], row["suffix"])
            assert [cp5.prior.digest_image(frame) for frame in outputs[key]] == executed["suffix_rgb_sha256"]


def test_physical_replay_and_common_prefix_alignment():
    sha = cp5.prior.prior.sha
    result = json.loads(physical.OUTPUT.read_text())
    prior = json.loads(common.OUTPUT.read_text())
    assert result["decision_sha256"] == sha(cp5.DECISION)
    assert result["execution_sha256"] == sha(cp5.RECEIPT)
    assert result["runner_sha256"] == sha(Path(physical.__file__))
    assert len(result["rows"]) == 12
    for row in result["rows"]:
        assert len(row["trace"]) == 31
        for tick, field in ((15, "initial_distance"), (20, "prefix_distance"), (30, "terminal_distance")):
            distance = np.linalg.norm(np.asarray(row["trace"][tick]["state"][:2]) - physical.GOAL)
            np.testing.assert_allclose(row[field], distance, atol=1e-12)
        match = next(x for x in prior["rows"] if x["root"] == row["root_index"]
                     and x["kind"] == "common_suffix" and x["role"] == row["role"]
                     and x["suffix_index"] == 0)
        np.testing.assert_allclose(row["trace"][20]["state"], match["trace"][20]["state"], atol=1e-12)
