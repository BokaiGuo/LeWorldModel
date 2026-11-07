"""Physical replay and latent alignment of all first-round V2 geometric chains."""
import json

import numpy as np
import torch

from experiments import pc_tworoom_rgb_geometric_candidates_v2_20261107_codex as v2
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = v2.v1.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_GEOMETRY_POOL_PHYSICAL_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_geometry_pool_alignment_20261107_codex.json"
FEATURES = ROOT / "results/pc_tworoom_geometry_pool_alignment_features_20261107_codex.npz"


def success_pair_accuracy(cost, success):
    cost, success = np.asarray(cost, dtype=float), np.asarray(success, dtype=bool)
    if cost.ndim != 1 or cost.shape != success.shape or not np.isfinite(cost).all():
        raise ValueError("invalid pair input")
    positive, negative = cost[success], cost[~success]
    if not len(positive) or not len(negative):
        return None
    diff = positive[:, None] - negative[None, :]
    return float(((diff < 0).sum() + .5 * (diff == 0).sum()) / diff.size)


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists() or FEATURES.exists():
        raise FileExistsError("sealed result exists")
    manifest = json.loads(v2.MANIFEST.read_text())
    for key, path in (("wrapper_sha256", v2.__file__), ("shared_runner_sha256", v2.v1.__file__),
                      ("protocol_sha256", v2.PROTOCOL), ("result_sha256", v2.OUTPUT)):
        if manifest[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    result = json.loads(v2.OUTPUT.read_text())
    model, scaler, transform, normalizer = v2.v1.panel.source.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != result["model_sha256"] or normalizer != result["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, v2.OUTPUT,
                         v2.v1.panel.PUBLIC, v2.v1.panel.source.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five physical steps per model packet")
    boundary = RGBLeWM(model, audit)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    records, arrays = [], {}
    print("Replay all initial geometric candidates and compare predicted, encoded and physical outcomes.", flush=True)
    try:
        with np.load(v2.v1.panel.PUBLIC, allow_pickle=False) as public, \
             np.load(v2.v1.panel.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            if len(result["rows"]) != 40:
                raise ValueError("expected complete 40-task panel")
            for i, row in enumerate(result["rows"]):
                if row["seed"] != int(public["seeds"][i]):
                    raise ValueError("task mismatch")
                original = row["arms"]["model"]["decisions"][0]
                pool = np.asarray(original["pool"], np.float32)
                if pool.shape != (16, 25, 2) or np.max(np.abs(pool)) > 1:
                    raise ValueError("invalid pool")
                normal = scaler.transform(pool.reshape(-1, 2)).reshape(1, 16, 5, 10)
                context_input = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                                 "goal": transform(public["goal"][i])[None, None, None].cuda(),
                                 "action": torch.empty((1, 1, 0, 10), device="cuda")}
                _, context = boundary.score(context_input, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                predicted = context["predicted_emb"][0, :, 1:]
                goal = context["goal_emb"][0, -1]
                if predicted.shape[:2] != (16, 5):
                    raise ValueError("prediction clock mismatch")
                predicted_cost = (predicted - goal).square().sum(-1)
                np.testing.assert_allclose(predicted_cost.min(-1).values.cpu().numpy(), original["scores"],
                                           rtol=1e-4, atol=1e-3)
                images, chains = [], []
                for action_chain in pool:
                    env.reset(seed=row["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    trace = []
                    for t, action in enumerate(action_chain):
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]), "success": bool(term),
                                      "truncated": bool(trunc)})
                        if (t + 1) % 5 == 0:
                            images.append(actor.render().copy())
                    chains.append(trace)
                encoded_batches = []
                for begin in range(0, len(images), 16):
                    batch = torch.stack([transform(im) for im in images[begin:begin + 16]]).cuda()
                    encoded_batches.append(boundary.encode({"pixels": batch[:, None]})["emb"][:, 0])
                actual = torch.cat(encoded_batches).reshape(16, 5, -1)
                if actual.shape != predicted.shape:
                    raise ValueError("predicted and encoded shape mismatch")
                encoded_cost = (actual - goal).square().sum(-1)
                physical = np.array([[step["distance"] for step in chain] for chain in chains])
                success = np.array([any(step["success"] for step in chain) for chain in chains])
                pred_costs = predicted_cost.min(-1).values.cpu().numpy()
                encoded_costs = encoded_cost.min(-1).values.cpu().numpy()
                distances = physical.min(-1)
                key = f"task{i:02d}"
                arrays[key + "_predicted"] = predicted.cpu().numpy()
                arrays[key + "_actual"] = actual.cpu().numpy()
                arrays[key + "_goal"] = goal.cpu().numpy()
                records.append({"seed": row["seed"], "chosen": original["chosen"], "chains": chains,
                                "predicted_cost_by_packet": predicted_cost.cpu().tolist(),
                                "encoded_cost_by_packet": encoded_cost.cpu().tolist(),
                                "predicted_success_pair_accuracy": success_pair_accuracy(pred_costs, success),
                                "encoded_success_pair_accuracy": success_pair_accuracy(encoded_costs, success),
                                "physical_distance_success_pair_accuracy": success_pair_accuracy(distances, success),
                                "feature_mse": float((predicted - actual).square().mean().item())})
                print(f"task={i} successful_chains={success.sum()}/16", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with OUTPUT.open("x") as stream:
        json.dump({"track": RGB, "scope": "exposed initial candidate pools, evaluation replay only",
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_sha256": file_hash(v2.OUTPUT), "features_sha256": file_hash(FEATURES),
                   "rows": records}, stream, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
