"""Replay every TwoRoom fixed-chain trajectory and audit saved feature arithmetic."""
import json

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_tworoom_rgb_action_effect_alignment_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result, physical, predicted = (json.loads(path.read_text()) for path in (run.RESULT, run.PHYSICAL, run.PRED))
    if (result["features_sha256"] != file_hash(run.FEATURES)
            or result["physical_sha256"] != file_hash(run.PHYSICAL)
            or result["prediction_sha256"] != file_hash(run.PRED)
            or physical["frames_sha256"] != file_hash(run.FRAMES)
            or predicted["features_sha256"] != file_hash(run.PRED_FEATURES)
            or predicted["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, max_distance_error = [], 0.
    try:
        with np.load(run.INPUT, allow_pickle=False) as inputs, np.load(run.FRAMES, allow_pickle=False) as frames, \
                np.load(run.FEATURES, allow_pickle=False) as feats, np.load(run.PRED_FEATURES, allow_pickle=False) as pred:
            for item in result["rows"]:
                seed = item["seed"]
                true, goal = feats[f"s{seed}_actual"], feats[f"s{seed}_goal"]
                forecast = pred[f"s{seed}_predicted"]
                if true.shape != forecast.shape or true.shape[0] != 25 or true.shape[1:] != goal.shape:
                    raise ValueError("feature shape mismatch")
                np.testing.assert_allclose(np.square(true-goal).sum(-1).reshape(5, 5),
                                           item["encoded"], atol=1e-4, rtol=0)
                np.testing.assert_allclose(np.square(forecast-goal).sum(-1).reshape(5, 5),
                                           item["prediction"], atol=1e-3, rtol=0)
                np.testing.assert_allclose(np.square(forecast-true).mean(), item["mean_feature_mse"], atol=1e-8)
                distance = []
                for i, a in enumerate(run.ACTIONS):
                    for j, b in enumerate(run.ACTIONS):
                        env.reset(seed=seed)
                        np.testing.assert_array_equal(env.unwrapped.render(), inputs[f"s{seed}_current"])
                        np.testing.assert_array_equal(run.image_goal(env.unwrapped), inputs[f"s{seed}_goal"])
                        saved = next(r for r in physical["rows"] if (r["seed"], r["prefix"], r["tail"]) == (seed, i, j))
                        for action, trace in zip([a] * 5 + [b] * 5, saved["trace"], strict=True):
                            _, _, terminated, truncated, info = env.unwrapped.step(action.copy())
                            error = abs(float(info["distance_to_target"]) - trace["distance"])
                            max_distance_error = max(error, max_distance_error)
                            if (bool(terminated), bool(truncated)) != (trace["terminated"], trace["truncated"]):
                                raise ValueError("physical flag mismatch")
                        np.testing.assert_array_equal(env.unwrapped.render(), frames[f"s{seed}_{i}_{j}"])
                        distance.append(info["distance_to_target"])
                np.testing.assert_allclose(np.asarray(distance).reshape(5, 5), item["distance"], atol=1e-6, rtol=0)
                matrices = [np.asarray(item[k]) for k in ("prediction", "encoded", "distance")]
                for key, left, right in (("predicted_vs_encoded", 0, 1),
                                         ("predicted_vs_distance", 0, 2),
                                         ("encoded_vs_distance", 1, 2)):
                    effect = lambda m: m[:, :1] - m[:, 1:]
                    if run.pair_agreement(effect(matrices[left]), effect(matrices[right])) != item[key]:
                        raise ValueError("effect agreement mismatch")
                    if run.ranking_agreement(matrices[left], matrices[right]) != item[key + "_ranking"]:
                        raise ValueError("ranking agreement mismatch")
                rows.append({"seed": seed, "replayed_chains": 25})
    finally:
        env.close()
    if max_distance_error > 1e-6 or len(rows) != 4:
        raise ValueError("physical replay mismatch")
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "max_distance_error": max_distance_error,
                   "result_sha256": file_hash(run.RESULT), "source_sha256": file_hash(__file__)}, stream, indent=2)
    print(f"replayed {len(rows) * 25} chains; maximum distance error {max_distance_error}")


if __name__ == "__main__":
    audit()
