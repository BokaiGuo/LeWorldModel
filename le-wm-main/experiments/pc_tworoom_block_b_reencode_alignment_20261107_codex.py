"""Evaluation-only alignment of sealed B actions, latent predictions and true RGB."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_BLOCK_B_REENCODE_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_block_b_reencode_alignment_20261107_codex"
PRED = Path(str(PREFIX) + "_pred.npz")
PRED_META = Path(str(PREFIX) + "_pred.json")
FRAMES = Path(str(PREFIX) + "_frames.npz")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
FEATURES = Path(str(PREFIX) + "_features.npz")
RESULT = Path(str(PREFIX) + "_result.json")


def source_records():
    plan = json.loads(prior.PLAN.read_text())
    physical = json.loads(prior.PHYSICAL.read_text())
    if (plan["pool_sha256"] != file_hash(prior.POOL)
            or physical["pool_sha256"] != file_hash(prior.POOL)
            or physical["plan_sha256"] != file_hash(prior.PLAN)):
        raise ValueError("source artifact mismatch")
    return plan, physical


def setup(plan):
    model, scaler, transform, normalizer = prior.source.block.prior.common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    return model, scaler, transform


def predict():
    if PRED.exists() or PRED_META.exists():
        raise FileExistsError("prediction already exists")
    plan, _ = source_records()
    model, scaler, transform = setup(plan)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.PLAN, prior.POOL,
                         prior.source.block.prior.inputs.INPUT,
                         prior.source.block.prior.common.CHECKPOINT / "weights.pt",
                         prior.source.block.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="five packets of five steps")
    arrays, errors = {}, []
    print("Reconstruct and seal 640 B predictions before true RGB processing.", flush=True)
    with (np.load(prior.POOL, allow_pickle=False) as pool,
          np.load(prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in range(10):
            if plan["rows"][root]["row"] != int(inputs["rows"][root]):
                raise ValueError("row mismatch")
            actions = pool[f"r{root}_b_actions"].reshape(64, 25, 2)
            norm = scaler.transform(actions.reshape(-1, 2)).reshape(64, 25, 2)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            wrapper = RGBLeWM(model, audit)
            costs, embeddings = [], []
            for offset in (0, 32):
                chunk = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                     dtype=torch.float32, device="cuda")
                value, out = wrapper.score(context, chunk)
                costs.append(value[0].cpu().numpy())
                embeddings.append(out["predicted_emb"][0, :, -1].cpu().numpy())
            costs = np.concatenate(costs)
            error = float(np.abs(costs - pool[f"r{root}_b_cost"].reshape(-1)).max())
            if error > 0.001:
                raise ValueError(f"source score mismatch root={root} error={error}")
            arrays[f"r{root}_predicted"] = np.concatenate(embeddings)
            arrays[f"r{root}_cost"] = costs
            errors.append(error)
            print(f"root={root} max_score_error={error:.8g}", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PRED_META.open("x") as f:
        json.dump({"pred_sha256": file_hash(PRED), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "plan_sha256": file_hash(prior.PLAN),
                   "pool_sha256": file_hash(prior.POOL), "max_score_errors": errors},
                  f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if FRAMES.exists() or PHYSICAL.exists():
        raise FileExistsError("physical output already exists")
    plan, old = source_records()
    meta = json.loads(PRED_META.read_text())
    for key, path in (("pred_sha256", PRED), ("runner_sha256", __file__),
                      ("protocol_sha256", PROTOCOL), ("plan_sha256", prior.PLAN),
                      ("pool_sha256", prior.POOL)):
        if meta[key] != file_hash(path):
            raise ValueError(f"prediction binding mismatch: {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    frames, rows = {}, []
    try:
        with (np.load(prior.POOL, allow_pickle=False) as pool,
              np.load(prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(prior.source.block.prior.common.DATA, "r") as data):
            for root in range(10):
                task = plan["rows"][root]["row"]
                root_frames, initial_diff = [], None
                for i, chain in enumerate(pool[f"r{root}_b_actions"].reshape(64, 25, 2)):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    if i == 0:
                        initial = actor.render().copy()
                        reference = inputs["current"][root]
                        if initial.shape != reference.shape:
                            raise ValueError("initial RGB shape mismatch")
                        initial_diff = float(np.abs(initial.astype(float) - reference.astype(float)).max())
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    if trace != old["rows"][root]["outcomes"][i]["trace"]:
                        raise ValueError(f"physical replay drift root={root} candidate={i}")
                    root_frames.append(actor.render().copy())
                frames[f"r{root}_frames"] = np.stack(root_frames)
                rows.append({"root": root, "row": task, "initial_rgb_max_abs_diff": initial_diff,
                             "replayed_traces": len(root_frames)})
                print(f"root={root} reproduced 64 physical trajectories", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, **frames)
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES),
                   "prior_physical_sha256": file_hash(prior.PHYSICAL),
                   "pred_meta_sha256": file_hash(PRED_META)},
                  f, indent=2, allow_nan=False)


def correlation(first, second):
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else None


def summarize_root(predicted, observed, goal, costs, outcomes):
    predicted, observed, goal, costs = map(np.asarray, (predicted, observed, goal, costs))
    if predicted.shape != observed.shape or predicted.shape[0] != 64 or len(outcomes) != 64:
        raise ValueError("feature shape mismatch")
    encoded_cost = np.square(observed - goal).sum(axis=-1)
    distance = np.asarray([x["final_distance"] for x in outcomes])
    model_choice, encoded_choice, physical_choice = map(int, (np.argmin(costs), np.argmin(encoded_cost), np.argmin(distance)))
    def selected(index):
        value = outcomes[index]
        return {"index": index, "ever": value["ever"], "prefix_success": value["prefix_success"],
                "final_distance": value["final_distance"]}
    return ({"feature_mse": float(np.square(predicted - observed).mean()),
             "cost_mae": float(np.abs(costs - encoded_cost).mean()),
             "predicted_vs_encoded": correlation(costs, encoded_cost),
             "predicted_vs_distance": correlation(costs, distance),
             "encoded_vs_distance": correlation(encoded_cost, distance),
             "model_selected": selected(model_choice), "encoded_oracle": selected(encoded_choice),
             "physical_oracle": selected(physical_choice),
             "pool_success": any(x["ever"] for x in outcomes)},
            encoded_cost)


def analyze():
    if FEATURES.exists() or RESULT.exists():
        raise FileExistsError("analysis output already exists")
    plan, old = source_records()
    meta, physics = json.loads(PRED_META.read_text()), json.loads(PHYSICAL.read_text())
    for actual, path in ((meta["pred_sha256"], PRED), (meta["runner_sha256"], __file__),
                         (meta["protocol_sha256"], PROTOCOL), (physics["frames_sha256"], FRAMES),
                         (physics["prior_physical_sha256"], prior.PHYSICAL),
                         (physics["pred_meta_sha256"], PRED_META)):
        if actual != file_hash(path):
            raise ValueError("alignment binding drift")
    model, _, transform = setup(plan)
    arrays, rows = {}, []
    with (np.load(prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          np.load(PRED, allow_pickle=False) as predictions,
          np.load(FRAMES, allow_pickle=False) as frames, torch.inference_mode()):
        for root in range(10):
            goal = model.encode({"pixels": transform(inputs["goal"][root])[None, None].cuda()})["emb"][0, 0]
            images = frames[f"r{root}_frames"]
            embeddings = []
            for offset in (0, 32):
                batch = torch.stack([transform(frame) for frame in images[offset:offset+32]])[:, None].cuda()
                embeddings.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
            actual = np.concatenate(embeddings)
            goal_np = goal.cpu().numpy()
            row, encoded_cost = summarize_root(predictions[f"r{root}_predicted"], actual, goal_np,
                                               predictions[f"r{root}_cost"], old["rows"][root]["outcomes"])
            row.update({"root": root, "row": plan["rows"][root]["row"],
                        "initial_rgb_max_abs_diff": physics["rows"][root]["initial_rgb_max_abs_diff"]})
            rows.append(row)
            arrays[f"r{root}_actual"] = actual
            arrays[f"r{root}_goal"] = goal_np
            arrays[f"r{root}_encoded_cost"] = encoded_cost
            print(f"root={root} encoded 64 real successors", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    with FEATURES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    summary = {"model_selected_success": sum(r["model_selected"]["ever"] for r in rows),
               "encoded_oracle_success": sum(r["encoded_oracle"]["ever"] for r in rows),
               "pool_success": sum(r["pool_success"] for r in rows),
               "mean_feature_mse": float(np.mean([r["feature_mse"] for r in rows])),
               "mean_predicted_vs_encoded": float(np.mean([r["predicted_vs_encoded"] for r in rows if r["predicted_vs_encoded"] is not None])),
               "mean_predicted_vs_distance": float(np.mean([r["predicted_vs_distance"] for r in rows if r["predicted_vs_distance"] is not None])),
               "mean_encoded_vs_distance": float(np.mean([r["encoded_vs_distance"] for r in rows if r["encoded_vs_distance"] is not None]))}
    with RESULT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "features_sha256": file_hash(FEATURES),
                   "prediction_sha256": file_hash(PRED), "physical_sha256": file_hash(PHYSICAL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "evaluation-only exposed development; encoded successor is oracle"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "physical", "analyze"):
        raise SystemExit("expected predict|physical|analyze")
    {"predict": predict, "physical": physical, "analyze": analyze}[sys.argv[1]]()
