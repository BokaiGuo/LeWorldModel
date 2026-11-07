"""Audit dataset RGB goals against the evaluator's reconstructed physical target."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_actual_prefix_action_effect_20261107_codex as source
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

run = source.run
PROTOCOL = run.ROOT.parent / "docs/PC_RGB_GOAL_RENDER_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
OUT = run.ROOT / "results/pc_tworoom_goal_render_alignment_20261107_codex.json"
FEATURES = run.ROOT / "results/pc_tworoom_goal_render_alignment_20261107_codex.npz"


def difference(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("image mismatch or nonfinite values")
    delta = np.abs(a - b)
    return {"mae": float(delta.mean()), "max": float(delta.max()),
            "unequal_fraction": float((delta > 0).mean())}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists() or FEATURES.exists():
        raise FileExistsError("result exists")
    prior = json.loads(source.OUT.read_text())
    original = json.loads(run.OUTPUT.read_text())
    if (prior["source_sha256"] != file_hash(source.source.OUTPUT)
            or prior["protocol_sha256"] != file_hash(source.PROTOCOL)
            or prior["runner_sha256"] != file_hash(source.__file__)
            or prior["feature_sha256"] != file_hash(source.FEATURES)):
        raise ValueError("prior artifact drift")
    model, scaler, transform, normalizer = run.common.model_setup()
    del scaler
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    previous = {x["episode"]: x for x in prior["rows"]}
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, arrays = [], {}
    try:
        with (np.load(run.INPUT, allow_pickle=False) as inputs,
              np.load(source.FEATURES, allow_pickle=False) as vectors,
              h5py.File(run.common.DATA, "r") as data, torch.inference_mode()):
            for i, episode in enumerate(inputs["episodes"]):
                episode, row = int(episode), int(inputs["rows"][i])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row])
                actor._set_goal_state(data["proprio"][row + 25])
                state_error = float(np.max(np.abs(actor.agent_position.numpy() - data["proprio"][row])))
                goal_error = float(np.max(np.abs(actor.target_position.numpy() - data["proprio"][row + 25])))
                if state_error != 0 or goal_error != 0:
                    raise ValueError("physical state restoration mismatch")
                current = actor.render().copy()
                goal = actor._target_img.cpu().numpy().transpose(1, 2, 0).copy()
                original_current, original_goal = inputs["current"][i], inputs["goal"][i]
                transformed = [transform(x) for x in (original_current, current, original_goal, goal)]
                embeddings = model.encode({"pixels": torch.stack(transformed).cuda()[:, None]})["emb"][:, 0]
                feature = embeddings.cpu().numpy()
                entry = {"episode": episode, "state_error": state_error, "goal_state_error": goal_error,
                         "current_rgb": difference(original_current, current), "goal_rgb": difference(original_goal, goal),
                         "current_preprocessed": difference(transformed[0].numpy(), transformed[1].numpy()),
                         "goal_preprocessed": difference(transformed[2].numpy(), transformed[3].numpy()),
                         "current_feature_mse": float(np.square(feature[0] - feature[1]).mean()),
                         "goal_feature_mse": float(np.square(feature[2] - feature[3]).mean()), "arms": {}}
                arrays[f"e{episode}_images"] = np.stack((original_current, current, original_goal, goal))
                arrays[f"e{episode}_features"] = feature
                if "ineligible_reason" not in previous[episode]:
                    for arm, old in previous[episode]["arms"].items():
                        key = f"e{episode}_{arm}"
                        pred, actual, old_goal = (vectors[key + suffix] for suffix in ("_predicted", "_actual", "_goal"))
                        goal_reencode_error = float(np.max(np.abs(old_goal - feature[2])))
                        if goal_reencode_error > 1e-3:
                            raise ValueError("original goal encoding drift")
                        old_pred = np.square(pred - old_goal).sum(-1)
                        old_actual = np.square(actual - old_goal).sum(-1)
                        np.testing.assert_allclose(old_pred, old["predicted_cost"], rtol=0, atol=1e-3)
                        np.testing.assert_allclose(old_actual, old["encoded_cost"], rtol=0, atol=1e-3)
                        entry["arms"][arm] = {
                            "goal_reencode_max_abs_error": goal_reencode_error,
                            "old_predicted": old_pred.tolist(), "old_encoded": old_actual.tolist(),
                            "rendered_predicted": np.square(pred - feature[3]).sum(-1).tolist(),
                            "rendered_encoded": np.square(actual - feature[3]).sum(-1).tolist(),
                            "physical_distance": [x[-1]["distance"] for x in old["trace"]]}
                rows.append(entry)
                print(f"episode={episode} goal_pixel_mae={entry['goal_rgb']['mae']:.4f}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(source.OUT), "input_sha256": file_hash(run.INPUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "feature_sha256": file_hash(FEATURES), "model_sha256": fingerprint, "rows": rows},
                  stream, allow_nan=False)
    print(f"sealed {len(rows)} goal alignment records", flush=True)


if __name__ == "__main__":
    main()
