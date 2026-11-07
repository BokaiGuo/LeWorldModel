"""Five-packet predicted/observed alignment for the sealed RGB B candidate pool."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_block_b_reencode_alignment_20261107_codex as base
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = base.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_BLOCK_B_TEMPORAL_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_block_b_temporal_alignment_20261107_codex"
PRED = Path(str(PREFIX) + "_pred.npz")
PRED_META = Path(str(PREFIX) + "_pred.json")
FRAMES = Path(str(PREFIX) + "_frames.npz")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
FEATURES = Path(str(PREFIX) + "_features.npz")
RESULT = Path(str(PREFIX) + "_result.json")


def predict():
    if PRED.exists() or PRED_META.exists():
        raise FileExistsError("prediction already exists")
    plan, _ = base.source_records()
    model, scaler, transform = base.setup(plan)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, base.prior.PLAN, base.prior.POOL,
                         base.prior.source.block.prior.inputs.INPUT,
                         base.prior.source.block.prior.common.CHECKPOINT / "weights.pt",
                         base.prior.source.block.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="initial plus five packets of five environment steps")
    arrays = {}
    print("Seal full model-time trajectories before reading new observed frames.", flush=True)
    with (np.load(base.prior.POOL, allow_pickle=False) as pool,
          np.load(base.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          np.load(base.PRED, allow_pickle=False) as old, torch.inference_mode()):
        for root in range(10):
            actions = pool[f"r{root}_b_actions"].reshape(64, 25, 2)
            norm = scaler.transform(actions.reshape(-1, 2)).reshape(64, 25, 2)
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            wrapper = RGBLeWM(model, audit)
            parts = []
            for offset in (0, 32):
                chunk = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                     dtype=torch.float32, device="cuda")
                _, output = wrapper.score(ctx, chunk)
                parts.append(output["predicted_emb"][0].cpu().numpy())
            full = np.concatenate(parts)
            if full.shape[:2] != (64, 6):
                raise ValueError(f"unexpected rollout clock: {full.shape}")
            np.testing.assert_allclose(full[:, -1], old[f"r{root}_predicted"], atol=1e-6, rtol=0)
            arrays[f"r{root}_predicted"] = full
            print(f"root={root} prediction clock verified", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model weights changed")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PRED_META.open("x") as f:
        json.dump({"pred_sha256": file_hash(PRED), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "source_plan_sha256": file_hash(base.prior.PLAN),
                   "source_pool_sha256": file_hash(base.prior.POOL), "old_prediction_sha256": file_hash(base.PRED)},
                  f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if FRAMES.exists() or PHYSICAL.exists():
        raise FileExistsError("physical output already exists")
    plan, old = base.source_records()
    meta = json.loads(PRED_META.read_text())
    for key, path in (("pred_sha256", PRED), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("source_pool_sha256", base.prior.POOL),
                      ("source_plan_sha256", base.prior.PLAN)):
        if meta[key] != file_hash(path):
            raise ValueError("prediction binding mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays, rows = {}, []
    try:
        with (np.load(base.prior.POOL, allow_pickle=False) as pool,
              np.load(base.FRAMES, allow_pickle=False) as prior_frames,
              h5py.File(base.prior.source.block.prior.common.DATA, "r") as data):
            for root in range(10):
                task = plan["rows"][root]["row"]
                frames = []
                for i, chain in enumerate(pool[f"r{root}_b_actions"].reshape(64, 25, 2)):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    trace, item = [], []
                    for step, action in enumerate(chain, start=1):
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if step % 5 == 0:
                            item.append(actor.render().copy())
                    if trace != old["rows"][root]["outcomes"][i]["trace"]:
                        raise ValueError(f"physical trajectory mismatch root={root} candidate={i}")
                    np.testing.assert_array_equal(item[-1], prior_frames[f"r{root}_frames"][i])
                    frames.append(item)
                arrays[f"r{root}_frames"] = np.asarray(frames, dtype=np.uint8)
                rows.append({"root": root, "row": task, "trajectory_count": len(frames)})
                print(f"root={root} aligned 64 x 5 frames", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES),
                   "pred_meta_sha256": file_hash(PRED_META), "old_physical_sha256": file_hash(base.prior.PHYSICAL),
                   "old_frames_sha256": file_hash(base.FRAMES)}, f, indent=2, allow_nan=False)


def temporal_summary(pred, actual, goal, outcomes):
    if pred.shape != actual.shape or pred.shape[:2] != (64, 5) or len(outcomes) != 64:
        raise ValueError("time/candidate alignment mismatch")
    pred_cost = np.square(pred - goal).sum(-1)
    actual_cost = np.square(actual - goal).sum(-1)
    dist = np.asarray([[x["trace"][i]["distance"] for i in (4, 9, 14, 19, 24)] for x in outcomes])
    term = np.asarray([[x["trace"][i]["terminated"] for i in (4, 9, 14, 19, 24)] for x in outcomes])
    ever = np.asarray([x["ever"] for x in outcomes])
    final = np.asarray([x["trace"][-1]["terminated"] for x in outcomes])
    points = []
    for t in range(5):
        points.append({"step": 5 * (t + 1), "feature_mse": float(np.square(pred[:, t] - actual[:, t]).mean()),
                       "pred_vs_encoded": base.correlation(pred_cost[:, t], actual_cost[:, t]),
                       "pred_vs_distance": base.correlation(pred_cost[:, t], dist[:, t]),
                       "encoded_vs_distance": base.correlation(actual_cost[:, t], dist[:, t])})
    indices = {"pred_final": int(np.argmin(pred_cost[:, -1])),
               "pred_min5": int(np.argmin(pred_cost.min(axis=1))),
               "encoded_final_oracle": int(np.argmin(actual_cost[:, -1])),
               "encoded_min5_oracle": int(np.argmin(actual_cost.min(axis=1)))}
    selections = {name: {"index": i, "ever": bool(ever[i]), "final": bool(final[i]),
                         "sampled_success": bool(term[i].any())} for name, i in indices.items()}
    return ({"points": points, "selections": selections, "ever": int(ever.sum()),
             "final": int(final.sum()), "ever_not_final": int((ever & ~final).sum()),
             "sampled": int(term.any(axis=1).sum()),
             "ever_not_sampled": int((ever & ~term.any(axis=1)).sum())},
            pred_cost, actual_cost)


def analyze():
    if FEATURES.exists() or RESULT.exists():
        raise FileExistsError("analysis already exists")
    plan, old = base.source_records()
    meta, physical_record = json.loads(PRED_META.read_text()), json.loads(PHYSICAL.read_text())
    for actual, path in ((meta["pred_sha256"], PRED), (meta["runner_sha256"], __file__),
                         (meta["protocol_sha256"], PROTOCOL), (physical_record["frames_sha256"], FRAMES),
                         (physical_record["pred_meta_sha256"], PRED_META),
                         (physical_record["old_physical_sha256"], base.prior.PHYSICAL)):
        if actual != file_hash(path):
            raise ValueError("analysis binding mismatch")
    model, _, transform = base.setup(plan)
    arrays, rows = {}, []
    with (np.load(base.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          np.load(PRED, allow_pickle=False) as predictions,
          np.load(FRAMES, allow_pickle=False) as frames,
          np.load(base.FEATURES, allow_pickle=False) as old_features, torch.inference_mode()):
        for root in range(10):
            goal = model.encode({"pixels": transform(inputs["goal"][root])[None, None].cuda()})["emb"][0, 0].cpu().numpy()
            np.testing.assert_allclose(goal, old_features[f"r{root}_goal"], atol=1e-6, rtol=0)
            images = frames[f"r{root}_frames"].reshape(320, *frames[f"r{root}_frames"].shape[2:])
            encoded = []
            for offset in range(0, 320, 32):
                batch = torch.stack([transform(frame) for frame in images[offset:offset+32]])[:, None].cuda()
                encoded.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
            actual = np.concatenate(encoded).reshape(64, 5, -1)
            np.testing.assert_allclose(actual[:, -1], old_features[f"r{root}_actual"], atol=1e-5, rtol=0)
            predicted = predictions[f"r{root}_predicted"][:, 1:]
            row, pred_cost, actual_cost = temporal_summary(predicted, actual, goal, old["rows"][root]["outcomes"])
            row.update({"root": root, "row": plan["rows"][root]["row"]})
            rows.append(row)
            arrays[f"r{root}_actual"] = actual
            arrays[f"r{root}_goal"] = goal
            arrays[f"r{root}_predicted_cost"] = pred_cost
            arrays[f"r{root}_encoded_cost"] = actual_cost
            print(f"root={root} encoded 320 time-aligned frames", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model weights changed")
    with FEATURES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    metrics = ("feature_mse", "pred_vs_encoded", "pred_vs_distance", "encoded_vs_distance")
    points = [{"step": 5 * (t + 1), **{key: float(np.mean([r["points"][t][key] for r in rows
                                                             if r["points"][t][key] is not None]))
                                        for key in metrics}} for t in range(5)]
    summary = {"points": points, "ever_count": sum(r["ever"] for r in rows),
               "final_count": sum(r["final"] for r in rows),
               "ever_not_final_count": sum(r["ever_not_final"] for r in rows),
               "ever_not_sampled_count": sum(r["ever_not_sampled"] for r in rows),
               "selected_ever": {key: sum(r["selections"][key]["ever"] for r in rows)
                                 for key in ("pred_final", "pred_min5", "encoded_final_oracle", "encoded_min5_oracle")}}
    with RESULT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "features_sha256": file_hash(FEATURES),
                   "pred_sha256": file_hash(PRED), "physical_sha256": file_hash(PHYSICAL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "post-hoc exposed development; true frames are oracle"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "physical", "analyze"):
        raise SystemExit("expected predict|physical|analyze")
    {"predict": predict, "physical": physical, "analyze": analyze}[sys.argv[1]]()
