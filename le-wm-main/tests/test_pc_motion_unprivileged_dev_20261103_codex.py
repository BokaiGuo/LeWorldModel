import json
from pathlib import Path

import numpy as np

from experiments import pc_motion_unprivileged_dev_20261103_codex as run
from experiments import pc_motion_unprivileged_physical_20261103_codex as physical
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base


def test_sealed_generation_real_history_and_execution():
    decision = json.loads(run.DECISIONS.read_text())
    execution = json.loads(run.RECEIPT.read_text())
    assert decision["runner_sha256"] == run.prior.sha(Path(run.__file__))
    assert decision["generator_sha256"] == run.prior.sha(Path(run.policy.__file__))
    assert decision["protocol_sha256"] == run.prior.sha(run.PROTOCOL)
    assert decision["history_sha256"] == run.prior.sha(run.HISTORY)
    assert execution["decision_sha256"] == run.prior.sha(run.DECISIONS)
    assert execution["arrays_sha256"] == run.prior.sha(run.EXECUTION)
    mean, std = np.asarray(decision["action_mean"]), np.asarray(decision["action_std"])
    with np.load(run.HISTORY, allow_pickle=False) as histories, np.load(run.EXECUTION, allow_pickle=False) as images:
        for root, row in enumerate(decision["rows"]):
            generated = run.policy.build_fixed_bank(row["candidate_seed"])
            assert len(generated) == len(row["candidates"]) == 11
            np.testing.assert_allclose(row["warmup"], run.WARMUP)
            np.testing.assert_allclose(np.asarray(row["history_action_packets_normalized"]).reshape(10, 2)*std+mean,
                                       run.WARMUP[5:], atol=1e-7)
            frames = histories[f"root{root}_history_rgb"]
            assert frames.shape == (16, 224, 224, 3)
            assert [run.digest_image(frame) for frame in frames] == row["rgb_sha256"]
            assert any(not np.array_equal(frames[i], frames[0]) for i in range(1, 16))
            for generated_item, candidate in zip(generated, row["candidates"], strict=True):
                program = candidate["program"]
                assert (program["name"], program["source"]) == (generated_item["name"], generated_item["source"])
                np.testing.assert_allclose(program["prefix"], generated_item["prefix"])
                assert len(candidate["evaluations"]) == 100
                assert program["terminal_score"] == min(x["score"] for x in candidate["evaluations"])
            result = run.policy.select_candidates([c["program"] for c in row["candidates"]], row["initial_score"])
            assert result == row["choice"]
            selected = row["candidates"][result["selected_index"]]["program"]
            np.testing.assert_allclose(execution["rows"][root]["actions"], selected["prefix"]+selected["suffix"])
            assert [run.digest_image(frame) for frame in images[f"root{root}_rgb"]] == execution["rows"][root]["rgb_sha256"]
            assert all(t[0] <= t[1] for t in decision["timings"][root])


def test_physical_alignment_and_common_suffix_pairing():
    result = json.loads(physical.OUTPUT.read_text())
    decision = json.loads(run.DECISIONS.read_text())
    assert result["decision_sha256"] == run.prior.sha(run.DECISIONS)
    assert result["protocol_sha256"] == run.prior.sha(physical.PROTOCOL)
    assert result["physical_chains"] == len(result["rows"]) == 116
    for row in result["rows"]:
        assert len(row["trace"]) == 31
        for tick, key in ((15, "initial_distance"), (20, "prefix_distance"), (30, "terminal_distance")):
            np.testing.assert_allclose(np.linalg.norm(np.asarray(row["trace"][tick]["state"][:2]) - physical.GOAL),
                                       row[key], atol=1e-12)
    for root in range(4):
        common = [r for r in result["rows"] if r["root"] == root and r["kind"] == "common_suffix"]
        for role in ("selected", "zero", "random_0"):
            assert [r["suffix_index"] for r in common if r["role"] == role] == list(range(6))
        index = decision["rows"][root]["choice"]["selected_index"]
        chosen = next(r for r in result["rows"] if r["root"] == root and r["kind"] == "model_chain" and r["candidate_index"] == index)
        paired = next(r for r in common if r["role"] == "selected" and r["suffix_index"] == 0)
        np.testing.assert_allclose(chosen["prefix_position"], paired["prefix_position"], atol=1e-12)
