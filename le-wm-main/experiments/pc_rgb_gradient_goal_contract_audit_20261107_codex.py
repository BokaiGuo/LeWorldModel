"""Evaluation-only audit of sealed RGB goal images against physical goal state."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from eval import img_transform
from experiments import pc_rgb_gradient_endpoint_decomposition_20261107_codex as source
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_rgb_single_frame_contract_20261107_codex import model_digest

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_GRADIENT_GOAL_CONTRACT_AUDIT_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_rgb_gradient_goal_contract_audit_20261107_codex.json"


def image_difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
        raise ValueError("RGB images must have equal HWC shapes")
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {"shape": list(left.shape), "mae": float(delta.mean()), "max": int(delta.max()),
            "different_pixels": int(np.count_nonzero(np.any(delta != 0, axis=-1)))}


def cost_summary(predicted, actual, target, distances):
    predicted, actual, target, distances = map(np.asarray, (predicted, actual, target, distances))
    if (predicted.shape != (5, 192) or actual.shape != predicted.shape
            or target.shape != (192,) or distances.shape != (5,)
            or not all(np.isfinite(x).all() for x in (predicted, actual, target, distances))):
        raise ValueError("invalid sealed features or distances")
    pred_cost = np.square(predicted - target).sum(-1)
    actual_cost = np.square(actual - target).sum(-1)
    pred_index, actual_index = int(pred_cost.argmin()), int(actual_cost.argmin())
    return {"predicted_cost": pred_cost.tolist(), "actual_cost": actual_cost.tolist(),
            "predicted_index": pred_index, "actual_oracle_index": actual_index,
            "selected_distance": float(distances[pred_index]),
            "actual_oracle_distance": float(distances[actual_index])}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads(source.OUT.read_text())
    plan = json.loads(source.PLAN.read_text())
    legacy = json.loads(source.capture.previous.legacy.PLAN.read_text())
    if (report["features_sha256"] != file_hash(source.FEATURES)
            or report["plan_sha256"] != file_hash(source.PLAN)
            or plan["public_sha256"] != file_hash(source.PUBLIC)):
        raise ValueError("sealed input drift")
    model = load_compatible_pretrained(str(source.CHECKPOINT), device="cuda")
    if model_digest(model) != report["model_digest"]:
        raise ValueError("checkpoint drift")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (h5py.File(source.capture.previous.legacy.prior.source.common.DATA, "r") as db,
              np.load(source.PUBLIC, allow_pickle=False) as public,
              np.load(source.FEATURES, allow_pickle=False) as features, torch.inference_mode()):
            source.capture.previous.planner.validate_public(public)
            for record, historic in zip(report["rows"], legacy["rows"], strict=True):
                episode = record["episode"]
                if episode != historic["episode"]:
                    raise ValueError("episode mismatch")
                index, = np.flatnonzero(public["episodes"] == episode)
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(db["proprio"][historic["row"]])
                actor._set_goal_state(db["proprio"][historic["row"] + 25])
                if not np.array_equal(actor.render(), public["current"][index]):
                    raise ValueError("initial RGB mismatch")
                goal_image = actor._target_img.cpu().numpy().transpose(1, 2, 0).copy()
                actor._set_state(db["proprio"][historic["row"] + 25])
                at_goal = actor.render().copy()
                position_distance = float(torch.linalg.vector_norm(actor.agent_position - actor.target_position))
                if position_distance != 0 or not np.array_equal(at_goal, goal_image):
                    raise ValueError("goal state does not match rendered goal")
                _, _, goal_success, _, info = actor.step(np.zeros(2, dtype=np.float32))
                if not goal_success or float(info["distance_to_target"]) != 0:
                    raise ValueError("goal state violates success predicate")
                key = f"e{episode}"
                predicted, actual = features[key + "_predicted"], features[key + "_actual"]
                old_goal = features[key + "_goal"]
                rerendered_goal = model.encode({"pixels": transform(goal_image)[None, None].cuda()})["emb"][0, 0].cpu().numpy()
                old = cost_summary(predicted, actual, old_goal, record["distance"])
                new = cost_summary(predicted, actual, rerendered_goal, record["distance"])
                if (not np.allclose(old["predicted_cost"], record["predicted_cost"], rtol=1e-5, atol=1e-3)
                        or not np.allclose(old["actual_cost"], record["actual_cost"], rtol=1e-5, atol=1e-3)
                        or old["predicted_index"] != record["predicted_index"]):
                    raise ValueError("original scoring mismatch")
                rows.append({"episode": episode, "initial_success": record["initial_success"],
                             "goal_image_difference": image_difference(public["goal"][index], goal_image),
                             "at_goal_distance": position_distance, "at_goal_zero_action_success": bool(goal_success),
                             "target_encoding_mse": float(np.square(old_goal-rerendered_goal).mean()),
                             "recorded_goal": old, "rendered_goal": new})
                print(f"episode={episode} rgb_mae={rows[-1]['goal_image_difference']['mae']:.3f} "
                      f"selection={old['predicted_index']}/{new['predicted_index']}", flush=True)
    finally:
        env.close()
    if model_digest(model) != report["model_digest"]:
        raise ValueError("model changed")
    with OUTPUT.open("x") as stream:
        json.dump({"scope": "post-hoc exposed-task evaluation; no planner access to physical state",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "decomposition_sha256": file_hash(source.OUT), "plan_sha256": file_hash(source.PLAN),
                   "model_digest": report["model_digest"], "rows": rows}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
