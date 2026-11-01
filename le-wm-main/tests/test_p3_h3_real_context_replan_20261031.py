import hashlib
import json

import numpy as np

from experiments.spectral_reachability import p3_h3_real_context_replan_20261031 as plan
from experiments.spectral_reachability import p3_h3_real_context_replay_20261031 as replay


def test_sealed_real_context_and_action_binding():
    path = plan.ROOT / "results/p3_h3_real_context_replan_20261031_RUN01.json"
    assert plan.base.sha(path) == "7abd67ecf6d87885588062134b6114c8a614b7adc438e599a63e207a3385bdaf"
    result = json.loads(path.read_text())
    previous = json.loads((plan.ROOT / "results/p3_h3_real_context_replay_20261031_RUN01.json").read_text())
    entries, excluded, _ = replay.collect()
    assert len(entries) == len(result["rows"]) == len(previous["rows"]) == 52
    assert excluded == result["excluded"]
    assert result["model_before"] == result["model_after"] == previous["model_before"]
    assert result["host"]["device"] == "cuda:0"
    for entry, row, old in zip(entries, result["rows"], previous["rows"], strict=True):
        assert all(row[k] == old[k] for k in ("logical_pair", "variant", "decision_index",
                                              "candidate", "policy", "root_slot"))
        assert row["baseline_score"] == old["refreshed_goal_cost"]
        assert bytes.fromhex(row["prefix15_float32_hex"]) == entry["actions"][:15].astype("<f4").tobytes()
        assert bytes.fromhex(row["original_suffix_float32_hex"]) == entry["actions"][15:].astype("<f4").tobytes()
        assert row["context_pixel_sha256"] == [
            hashlib.sha256(frame.tobytes()).hexdigest() for frame in entry["frames"][:3]
        ]
        chosen = bytes.fromhex(row["chosen_suffix_float32_hex"])
        assert len(chosen) == 40
        assert np.isfinite(np.frombuffer(chosen, dtype="<f4")).all()
        assert (np.abs(np.frombuffer(chosen, dtype="<f4")) <= 1).all()
        assert hashlib.sha256(bytes.fromhex(row["prefix15_float32_hex"]) + chosen).hexdigest() == row["action_sha256"]
        assert row["chosen"] == ("optimized" if row["optimized_score"] < row["baseline_score"] else "original")
    assert not result["P3_admission"] and not result["equal_budget_certified"]
