"""Compare saved RGB plans, their model trajectories, and physical replay."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_rgb_conditional_temporal_cem_20261107_codex as src
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = src.old.ROOT
PREFIX = ROOT / "results/pc_conditional_selected_alignment_20261107_codex"
PRED, META, FRAMES, PHYS, FEATURES, RESULT = [Path(str(PREFIX) + suffix) for suffix in (
    "_pred.npz", "_pred.json", "_frames.npz", "_physical.json", "_features.npz", "_result.json")]
PROTOCOL = ROOT.parent / "docs/PC_CONDITIONAL_SELECTED_ALIGNMENT_PROTOCOL_20261107_CODEX.md"


def source_records():
    plan = json.loads(src.PLAN.read_text())
    truth = json.loads(src.PHYS.read_text())
    source = src.old.base.source
    inputs = source.block.prior.inputs.INPUT
    paths = {"pool": src.POOL, "old_plan": src.old.PLAN, "source_pool": source.POOL,
             "inputs": inputs, "protocol": src.PROTOCOL, "runner": src.__file__}
    if (any(plan["sha256"][k] != file_hash(v) for k, v in paths.items())
            or truth["plan_sha256"] != file_hash(src.PLAN)
            or truth["pool_sha256"] != file_hash(src.POOL)):
        raise ValueError("source artifact drift")
    if ([r["root"] for r in plan["rows"]] != list(range(20, 50))
            or [r["root"] for r in truth["rows"]] != list(range(20, 50))):
        raise ValueError("incomplete panel")
    return plan, truth, inputs


def setup(plan):
    common = src.old.base.source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    return model, scaler, transform


def predict():
    if PRED.exists() or META.exists():
        raise FileExistsError("prediction exists")
    plan, _, inputs_path = source_records()
    model, scaler, transform = setup(plan)
    common = src.old.base.source.block.prior.common
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, src.PLAN, src.POOL, inputs_path,
                         common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="initial plus five packets of five environment steps")
    arrays, rows = {}, []
    print("Seal predicted trajectories before using real successor frames.", flush=True)
    with (np.load(src.POOL, allow_pickle=False) as pool,
          np.load(inputs_path, allow_pickle=False) as inputs, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for task in plan["rows"]:
            root = task["root"]
            actions = pool[f"r{root}_selected"]
            if (actions.shape != (2, 25, 2) or not np.isfinite(actions).all()
                    or np.any(np.abs(actions) > 1)):
                raise ValueError("illegal selected actions")
            for i, method in enumerate(src.METHODS):
                if src.digest(actions[i]) != task["decisions"][method]["action_sha256"]:
                    raise ValueError("selected action mismatch")
            normalized = scaler.transform(actions.reshape(-1, 2)).reshape(1, 2, 5, 10)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            endpoint, result = wrapper.score(context, torch.as_tensor(normalized, dtype=torch.float32, device="cuda"))
            predicted = result["predicted_emb"][0].cpu().numpy()
            goal = result["goal_emb"][0, -1].cpu().numpy()
            if predicted.shape[:2] != (2, 6):
                raise ValueError("prediction time mismatch")
            costs = np.square(predicted[:, 1:] - goal).sum(-1)
            np.testing.assert_allclose(endpoint[0].cpu().numpy(), costs[:, -1], atol=0.001, rtol=0)
            original = np.asarray([task["decisions"][m]["cost"] for m in src.METHODS])
            error = float(np.max(np.abs(costs[:, -1] - original)))
            if error > 0.005:
                raise ValueError(f"stored score mismatch root={root}: {error}")
            arrays[f"r{root}_predicted"] = predicted
            arrays[f"r{root}_goal"] = goal
            rows.append({"root": root, "action_sha256": [src.digest(a) for a in actions],
                         "max_endpoint_recompute_error": error})
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with PRED.open("xb") as file:
        np.savez_compressed(file, **arrays)
    with META.open("x") as file:
        json.dump({"rows": rows, "pred_sha256": file_hash(PRED), "plan_sha256": file_hash(src.PLAN),
                   "pool_sha256": file_hash(src.POOL), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "model_sha256": plan["model_sha256"]}, file, indent=2)


def verify_prediction():
    metadata = json.loads(META.read_text())
    for key, path in (("pred_sha256", PRED), ("plan_sha256", src.PLAN), ("pool_sha256", src.POOL),
                      ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if metadata[key] != file_hash(path):
            raise ValueError("prediction binding drift")
    return metadata


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if FRAMES.exists() or PHYS.exists():
        raise FileExistsError("physical replay exists")
    plan, truth, inputs_path = source_records()
    metadata = verify_prediction()
    common = src.old.base.source.block.prior.common
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays, rows = {}, []
    try:
        with (np.load(src.POOL, allow_pickle=False) as pool,
              np.load(inputs_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as dataset):
            for task, old, receipt in zip(plan["rows"], truth["rows"], metadata["rows"], strict=True):
                root, row = task["root"], task["row"]
                all_frames, outcomes, initial_differences = [], {}, []
                for i, method in enumerate(src.METHODS):
                    chain = pool[f"r{root}_selected"][i]
                    if src.digest(chain) != receipt["action_sha256"][i]:
                        raise ValueError("action hash mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(dataset["proprio"][row])
                    actor._set_goal_state(dataset["proprio"][row + 25])
                    initial = actor.render().copy()
                    reference = inputs["current"][root]
                    difference = (float(np.abs(initial.astype(np.float32) - reference.astype(np.float32)).mean())
                                  if initial.shape == reference.shape else None)
                    frames, trace = [initial], []
                    for step, action in enumerate(chain, 1):
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if step % 5 == 0:
                            frames.append(actor.render().copy())
                    expected = old["outcomes"][method]["trace"]
                    if len(expected) != 25 or any(
                            abs(a["distance"] - b["distance"]) > 1e-5
                            or a["terminated"] != b["terminated"] or a["truncated"] != b["truncated"]
                            for a, b in zip(trace, expected, strict=True)):
                        raise ValueError(f"physical replay mismatch root={root} method={method}")
                    all_frames.append(frames)
                    outcomes[method] = trace
                    initial_differences.append({"raw_mean_abs": difference,
                                                "render_shape": list(initial.shape),
                                                "input_shape": list(reference.shape)})
                arrays[f"r{root}_frames"] = np.asarray(all_frames, dtype=np.uint8)
                rows.append({"root": root, "outcomes": outcomes, "initial_image_difference": initial_differences})
                print(f"root={root}: physical traces reproduced", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as file:
        np.savez_compressed(file, **arrays)
    with PHYS.open("x") as file:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES),
                   "prediction_meta_sha256": file_hash(META), "source_physical_sha256": file_hash(src.PHYS)},
                  file, allow_nan=False)


def analyze():
    if FEATURES.exists() or RESULT.exists():
        raise FileExistsError("analysis exists")
    plan, _, inputs_path = source_records()
    verify_prediction()
    physical_data = json.loads(PHYS.read_text())
    if (physical_data["frames_sha256"] != file_hash(FRAMES)
            or physical_data["prediction_meta_sha256"] != file_hash(META)
            or physical_data["source_physical_sha256"] != file_hash(src.PHYS)):
        raise ValueError("physical binding drift")
    model, _, transform = setup(plan)
    arrays, rows = {}, []
    with (np.load(PRED, allow_pickle=False) as predictions, np.load(FRAMES, allow_pickle=False) as frames,
          torch.inference_mode()):
        for task, record in zip(plan["rows"], physical_data["rows"], strict=True):
            root = task["root"]
            if record["root"] != root:
                raise ValueError("root mismatch")
            images = frames[f"r{root}_frames"]
            if images.shape[:2] != (2, 6):
                raise ValueError("physical clock mismatch")
            batch = torch.stack([transform(frame) for frame in images.reshape(-1, *images.shape[2:])])[:, None].cuda()
            actual = model.encode({"pixels": batch})["emb"][:, 0].reshape(2, 6, -1).cpu().numpy()
            pred, goal = predictions[f"r{root}_predicted"], predictions[f"r{root}_goal"]
            if actual.shape != pred.shape or not np.isfinite(actual).all():
                raise ValueError("feature clock mismatch")
            pred_cost = np.square(pred[:, 1:] - goal).sum(-1)
            actual_cost = np.square(actual[:, 1:] - goal).sum(-1)
            mse = np.square(pred - actual).mean(-1)
            distances = np.asarray([[record["outcomes"][m][t]["distance"] for t in (4, 9, 14, 19, 24)]
                                    for m in src.METHODS])
            terms = [[record["outcomes"][m][t]["terminated"] for t in (4, 9, 14, 19, 24)]
                     for m in src.METHODS]
            arrays[f"r{root}_predicted"] = pred
            arrays[f"r{root}_actual"] = actual
            arrays[f"r{root}_goal"] = goal
            rows.append({"root": root, "predicted_cost": pred_cost.tolist(), "actual_cost": actual_cost.tolist(),
                         "feature_mse_initial_and_packets": mse.tolist(), "distance": distances.tolist(),
                         "terminated": terms, "initial_image_difference": record["initial_image_difference"]})
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with FEATURES.open("xb") as file:
        np.savez_compressed(file, **arrays)
    predicted = np.asarray([r["predicted_cost"] for r in rows])
    actual = np.asarray([r["actual_cost"] for r in rows])
    distance = np.asarray([r["distance"] for r in rows])
    pred_delta, actual_delta, dist_delta = (x[:, 0, :] - x[:, 1, :] for x in (predicted, actual, distance))
    summary = {"roots": len(rows), "steps": [5, 10, 15, 20, 25],
               "predicted_vs_actual_cost_sign_agreement": (np.sign(pred_delta) == np.sign(actual_delta)).sum(0).tolist(),
               "actual_cost_vs_distance_sign_agreement": (np.sign(actual_delta) == np.sign(dist_delta)).sum(0).tolist(),
               "predicted_vs_distance_sign_agreement": (np.sign(pred_delta) == np.sign(dist_delta)).sum(0).tolist(),
               "mean_feature_mse_initial_and_packets": np.mean(
                   np.asarray([r["feature_mse_initial_and_packets"] for r in rows]), axis=(0, 1)).tolist()}
    with RESULT.open("x") as file:
        json.dump({"summary": summary, "rows": rows, "features_sha256": file_hash(FEATURES),
                   "pred_sha256": file_hash(PRED), "physical_sha256": file_hash(PHYS),
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "scope": "exposed selected-chain evaluation; real observations are not planner inputs"},
                  file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "physical", "analyze"):
        raise SystemExit("expected predict|physical|analyze")
    {"predict": predict, "physical": physical, "analyze": analyze}[sys.argv[1]]()
