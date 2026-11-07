"""Post-seal predicted/actual encoding comparison for paired RGB action chains."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments import pc_rgb_cem_incumbent_selection_20261107_codex as source

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_CEM_SELECTION_OUTCOME_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
FRAMES = ROOT / "results/pc_rgb_cem_selection_outcome_alignment_frames_20261107_codex.npz"
REPLAY = ROOT / "results/pc_rgb_cem_selection_outcome_alignment_replay_20261107_codex.json"
REPRESENTATIONS = ROOT / "results/pc_rgb_cem_selection_outcome_alignment_features_20261107_codex.npz"
RESULT = ROOT / "results/pc_rgb_cem_selection_outcome_alignment_20261107_codex.json"


def collect():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import PHYSICAL, IMAGES
    if FRAMES.exists() or REPLAY.exists():
        raise FileExistsError("replay output already exists")
    recorded = json.loads(source.EVAL.read_text())
    if recorded["scores_sha256"] != file_hash(source.SCORES) or recorded["physical_sha256"] != file_hash(PHYSICAL):
        raise ValueError("sealed scoring or physical source changed")
    physical = json.loads(PHYSICAL.read_text())
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor, frames, rows = env.unwrapped, {}, []
    try:
        with np.load(IMAGES, allow_pickle=False) as images:
            for row in recorded["rows"]:
                root, task = row["root"], physical["rows"][row["root"]]
                condition = next(c for c in task["conditions"] if c["side"] == 1 and c["appearance"] == "white")
                for arm in ("old", "best"):
                    env.reset(seed=881001 + root, options={"state": np.asarray(condition["requested_state"]),
                                                           "goal_state": np.asarray(task["goal"])})
                    actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                    np.testing.assert_array_equal(actor.render(), images[f"r{root}_white_1_plus_initial"])
                    trace = row["arms"][arm]
                    if len(trace) != 15 or any(s["terminated"] or s["truncated"] for s in trace):
                        raise ValueError("no complete fifteen-step chain")
                    for step in trace:
                        _, reward, terminated, truncated, _ = actor.step(np.asarray(step["action"], np.float32))
                        np.testing.assert_allclose(snapshot(actor), step["snapshot"], atol=1e-9, rtol=0)
                        if abs(float(reward) - step["reward"]) > 1e-9 or (bool(terminated), bool(truncated)) != (
                                step["terminated"], step["truncated"]):
                            raise ValueError("physical replay mismatch")
                    frames[f"r{root}_{arm}"] = actor.render().copy()
                    rows.append({"root": root, "arm": arm, "final_reward": trace[-1]["reward"], "steps": 15})
    finally:
        env.close()
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, **frames)
    with REPLAY.open("x") as stream:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES), "eval_sha256": file_hash(source.EVAL),
                   "physical_sha256": file_hash(PHYSICAL), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__)}, stream, indent=2, allow_nan=False)
    print(f"replayed {len(rows)} complete chains", flush=True)


def compare_signs(predicted, actual, reward, tolerance=1e-6):
    def sign(x):
        return int(x > tolerance) - int(x < -tolerance)
    return {"predicted_preference": sign(predicted[0] - predicted[1]),
            "real_encoded_preference": sign(actual[0] - actual[1]),
            "reward_preference": sign(reward[1] - reward[0])}


def analyze():
    from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import IMAGES
    if REPRESENTATIONS.exists() or RESULT.exists():
        raise FileExistsError("analysis output already exists")
    replay, scored = json.loads(REPLAY.read_text()), json.loads(source.SCORES.read_text())
    if (replay["frames_sha256"] != file_hash(FRAMES) or replay["eval_sha256"] != file_hash(source.EVAL)
            or replay["runner_sha256"] != file_hash(__file__) or replay["protocol_sha256"] != file_hash(PROTOCOL)
            or scored["runner_sha256"] != file_hash(source.__file__) or scored["images_sha256"] != file_hash(IMAGES)):
        raise ValueError("source drift")
    model, _, transform, action_hash = source.source.model_and_scaler()
    if model_fingerprint(model) != scored["model_sha256"] or action_hash != scored["normalizer_source_sha256"]:
        raise ValueError("model or action normalizer changed")
    features, rows = {}, []
    with np.load(IMAGES, allow_pickle=False) as images, np.load(FRAMES, allow_pickle=False) as frames, torch.inference_mode():
        for row in scored["rows"]:
            root = row["root"]
            actions, costs, best, old = source.choose(row["batches"])
            pixel = transform(images[f"r{root}_white_1_plus_initial"])[None, None, None].cuda()
            goal = transform(images[f"r{root}_white_goal"])[None, None, None].cuda()
            selected = torch.tensor(actions[[old, best]][None], dtype=torch.float32, device="cuda")
            context = {"pixels": pixel, "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
            pred_cost = model.get_cost(context, selected).cpu().numpy()[0]
            np.testing.assert_allclose(pred_cost, costs[[old, best]], atol=1e-3, rtol=0)
            predicted = context["predicted_emb"][0, :, -1].cpu().numpy()
            target = context["goal_emb"][0, -1].cpu().numpy()
            actual = model.encode({"pixels": torch.stack([transform(frames[f"r{root}_{arm}"])
                                                           for arm in ("old", "best")]).cuda()[:, None]})["emb"][:, 0].cpu().numpy()
            true_cost = np.square(actual - target).sum(axis=-1)
            mse = np.square(predicted - actual).mean(axis=-1)
            reward = [next(r["final_reward"] for r in replay["rows"] if r["root"] == root and r["arm"] == arm)
                      for arm in ("old", "best")]
            features[f"r{root}_predicted"] = predicted
            features[f"r{root}_actual"] = actual
            features[f"r{root}_goal"] = target
            rows.append({"root": root, "predicted_cost": pred_cost.tolist(), "real_encoded_cost": true_cost.tolist(),
                         "feature_mse": mse.tolist(), "reward": reward,
                         "preferences": compare_signs(pred_cost, true_cost, reward)})
    if model_fingerprint(model) != scored["model_sha256"]:
        raise ValueError("model weights changed")
    with REPRESENTATIONS.open("xb") as stream:
        np.savez_compressed(stream, **features)
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "replay_sha256": file_hash(REPLAY), "scores_sha256": file_hash(source.SCORES),
                   "features_sha256": file_hash(REPRESENTATIONS), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "model_sha256": scored["model_sha256"],
                   "scope": "post-selection exposed RGB development; no independent confirmation"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("collect", "analyze"):
        raise SystemExit("expected collect|analyze")
    {"collect": collect, "analyze": analyze}[sys.argv[1]]()
