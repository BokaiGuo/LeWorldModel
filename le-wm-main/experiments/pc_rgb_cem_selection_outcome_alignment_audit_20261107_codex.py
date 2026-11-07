"""Recompute recorded feature distances and compare physical rewards."""
import json

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_rgb_cem_selection_outcome_alignment_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_rgb_cem_selection_outcome_alignment_audit_20261107_codex.json"


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.RESULT.read_text())
    replay = json.loads(run.REPLAY.read_text())
    if (result["features_sha256"] != file_hash(run.REPRESENTATIONS)
            or result["replay_sha256"] != file_hash(run.REPLAY)
            or replay["frames_sha256"] != file_hash(run.FRAMES)):
        raise ValueError("source drift")
    rows = []
    with np.load(run.REPRESENTATIONS, allow_pickle=False) as features:
        for item in result["rows"]:
            root = item["root"]
            prediction = features[f"r{root}_predicted"]
            actual = features[f"r{root}_actual"]
            goal = features[f"r{root}_goal"]
            if prediction.shape != actual.shape or prediction.shape[0] != 2 or prediction.shape[1:] != goal.shape:
                raise ValueError("feature shape mismatch")
            predicted = np.square(prediction - goal).sum(axis=-1)
            true = np.square(actual - goal).sum(axis=-1)
            mse = np.square(prediction - actual).mean(axis=-1)
            np.testing.assert_allclose(predicted, item["predicted_cost"], atol=1e-3, rtol=0)
            np.testing.assert_allclose(true, item["real_encoded_cost"], atol=1e-5, rtol=0)
            np.testing.assert_allclose(mse, item["feature_mse"], atol=1e-8, rtol=0)
            reward = [next(r["final_reward"] for r in replay["rows"] if r["root"] == root and r["arm"] == arm)
                      for arm in ("old", "best")]
            np.testing.assert_allclose(reward, item["reward"], atol=0, rtol=0)
            prefs = run.compare_signs(predicted, true, reward)
            if prefs != item["preferences"]:
                raise ValueError("preference mismatch")
            rows.append({"root": root, "preferences": prefs})
    if len(rows) != 4 or len({r["root"] for r in rows}) != 4:
        raise ValueError("panel incomplete")
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "result_sha256": file_hash(run.RESULT)}, stream, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    audit()
