"""Compare predictions, rendered successors and physical outcomes on sealed pools."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_conditional_selected_alignment_20261107_codex as align
from experiments import pc_conditional_failure_pool_20261107_codex as failure
from experiments import pc_rgb_conditional_temporal_cem_20261107_codex as src
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

BASE = src.old.ROOT / "results/pc_failure_pool_three_way_20261107_codex"
PRED, META, FRAMES, PHYS, FEAT, OUT = [Path(str(BASE) + suffix) for suffix in
    ("_pred.npz", "_pred.json", "_frames.npz", "_physical.json", "_features.npz", "_result.json")]
PROTOCOL = src.old.ROOT.parent / "docs/PC_FAILURE_POOL_THREE_WAY_PROTOCOL_20261107_CODEX.md"
ARMS = ((38, "temporal"), (39, "endpoint"), (39, "temporal"))


def get_pool(pool, root, method):
    method_index = src.METHODS.index(method)
    actions = np.concatenate((pool[f"r{root}_inc"], pool[f"r{root}_actions"][method_index]))
    scores = np.concatenate((pool[f"r{root}_inc_scores"][:, -1],
                             pool[f"r{root}_scores"][method_index, :, -1]))
    if actions.shape != (194, 25, 2) or scores.shape != (194,):
        raise ValueError("pool shape")
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1 or not np.isfinite(scores).all():
        raise ValueError("illegal pool")
    return actions, scores


def label_metrics(costs, success):
    costs, success = np.asarray(costs), np.asarray(success, dtype=bool)
    if costs.ndim != 1 or costs.shape != success.shape or not np.isfinite(costs).all():
        raise ValueError("invalid outcomes")
    order = np.argsort(costs, kind="stable")
    diff = costs[~success, None] - costs[success][None, :]
    ranks = np.flatnonzero(success[order])
    return {"selected_index": int(order[0]), "selected_success": bool(success[order[0]]),
            "success_count": int(success.sum()), "pair_count": int(diff.size),
            "success_better_pairs": int((diff > 0).sum()), "tied_pairs": int((diff == 0).sum()),
            "success_worse_pairs": int((diff < 0).sum()),
            "best_success_rank": int(ranks[0] + 1) if ranks.size else None}


def verify_failure():
    plan, _, inputs = align.source_records()
    record = json.loads(failure.OUTPUT.read_text())
    for key, path in (("plan_sha256", src.PLAN), ("pool_sha256", src.POOL),
                      ("physical_sha256", src.PHYS), ("protocol_sha256", failure.PROTOCOL),
                      ("runner_sha256", failure.__file__)):
        if record[key] != file_hash(path):
            raise ValueError("failure replay binding drift")
    return plan, record, inputs


def predict():
    if PRED.exists() or META.exists():
        raise FileExistsError("predictions exist")
    plan, _, inputs_path = verify_failure()
    model, scaler, transform = align.setup(plan)
    common = src.old.base.source.block.prior.common
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, failure.OUTPUT, src.PLAN, src.POOL,
                         inputs_path, common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="five packets of five environment steps")
    arrays, records = {}, []
    print("Score all fixed candidates before reading successor images.", flush=True)
    with (np.load(src.POOL, allow_pickle=False) as pool,
          np.load(inputs_path, allow_pickle=False) as inputs, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for root, method in ARMS:
            actions, scores = get_pool(pool, root, method)
            norm = scaler.transform(actions.reshape(-1, 2)).reshape(-1, 5, 10)
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            predicted, goals = [], []
            for start in range(0, len(norm), 32):
                batch = torch.as_tensor(norm[None, start:start+32], device="cuda", dtype=torch.float32)
                _, result = wrapper.score(ctx, batch)
                predicted.append(result["predicted_emb"][0, :, -1].cpu().numpy())
                goals.append(result["goal_emb"][0, -1].cpu().numpy())
            pred = np.concatenate(predicted)
            if any(not np.allclose(goals[0], g, atol=1e-6, rtol=0) for g in goals):
                raise ValueError("goal drift")
            current_costs = np.square(pred - goals[0]).sum(-1)
            error = float(np.max(np.abs(current_costs - scores)))
            if error > 0.005:
                raise ValueError(f"recomputed prediction error {root}/{method}: {error}")
            key = f"r{root}_{method}"
            arrays[key + "_pred"] = pred
            arrays[key + "_goal"] = goals[0]
            records.append({"root": root, "method": method, "max_score_error": error})
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with PRED.open("xb") as file:
        np.savez_compressed(file, **arrays)
    with META.open("x") as file:
        json.dump({"records": records, "pred_sha256": file_hash(PRED), "plan_sha256": file_hash(src.PLAN),
                   "pool_sha256": file_hash(src.POOL), "failure_sha256": file_hash(failure.OUTPUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__)}, file, indent=2)


def verify_prediction():
    meta = json.loads(META.read_text())
    for key, path in (("pred_sha256", PRED), ("plan_sha256", src.PLAN), ("pool_sha256", src.POOL),
                      ("failure_sha256", failure.OUTPUT), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__)):
        if meta[key] != file_hash(path):
            raise ValueError("prediction artifact drift")


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if FRAMES.exists() or PHYS.exists():
        raise FileExistsError("physical artifacts exist")
    plan, truth, _ = verify_failure()
    verify_prediction()
    truth_by_root = {r["root"]: r for r in truth["rows"]}
    tasks = {r["root"]: r for r in plan["rows"]}
    common = src.old.base.source.block.prior.common
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays, rows = {}, []
    try:
        with (np.load(src.POOL, allow_pickle=False) as pool, h5py.File(common.DATA, "r") as data):
            for root, method in ARMS:
                actions, _ = get_pool(pool, root, method)
                expected = truth_by_root[root]["arms"][method]["trajectories"]
                images, labels = [], []
                for i, chain in enumerate(actions):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    row = tasks[root]["row"]
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    if len(expected[i]) != 25 or any(
                            abs(a["distance"] - b["distance"]) > 1e-5
                            or a["terminated"] != b["terminated"] or a["truncated"] != b["truncated"]
                            for a, b in zip(trace, expected[i], strict=True)):
                        raise ValueError(f"physical drift {root}/{method}/{i}")
                    images.append(actor.render().copy())
                    labels.append({"ever": any(x["terminated"] for x in trace),
                                   "final": trace[-1]["terminated"], "distance": trace[-1]["distance"]})
                arrays[f"r{root}_{method}_frames"] = np.asarray(images, dtype=np.uint8)
                rows.append({"root": root, "method": method, "outcomes": labels})
                print(f"{root}/{method}: replayed 194 trajectories and rendered endpoints", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as file:
        np.savez_compressed(file, **arrays)
    with PHYS.open("x") as file:
        json.dump({"rows": rows, "frames_sha256": file_hash(FRAMES), "pred_meta_sha256": file_hash(META),
                   "failure_sha256": file_hash(failure.OUTPUT)}, file, allow_nan=False)


def analyze():
    if FEAT.exists() or OUT.exists():
        raise FileExistsError("analysis exists")
    plan, _, _ = verify_failure()
    verify_prediction()
    record = json.loads(PHYS.read_text())
    if (record["frames_sha256"] != file_hash(FRAMES) or record["pred_meta_sha256"] != file_hash(META)
            or record["failure_sha256"] != file_hash(failure.OUTPUT)):
        raise ValueError("physical artifacts drift")
    model, _, transform = align.setup(plan)
    arrays, rows = {}, []
    with (np.load(PRED, allow_pickle=False) as predictions, np.load(FRAMES, allow_pickle=False) as frames,
          np.load(src.POOL, allow_pickle=False) as pool, torch.inference_mode()):
        for (root, method), item in zip(ARMS, record["rows"], strict=True):
            if (root, method) != (item["root"], item["method"]) or len(item["outcomes"]) != 194:
                raise ValueError("physical pool mismatch")
            key = f"r{root}_{method}"
            images = frames[key + "_frames"]
            actual_parts = []
            for start in range(0, len(images), 32):
                batch = torch.stack([transform(frame) for frame in images[start:start+32]])[:, None].cuda()
                actual_parts.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
            actual, pred, goal = np.concatenate(actual_parts), predictions[key + "_pred"], predictions[key + "_goal"]
            if actual.shape != pred.shape:
                raise ValueError("feature shape mismatch")
            predicted_cost = np.square(pred - goal).sum(-1)
            actual_cost = np.square(actual - goal).sum(-1)
            _, saved = get_pool(pool, root, method)
            success = np.asarray([x["ever"] for x in item["outcomes"]])
            final = np.asarray([x["final"] for x in item["outcomes"]])
            arrays[key + "_pred"] = pred
            arrays[key + "_actual"] = actual
            arrays[key + "_goal"] = goal
            rows.append({"root": root, "method": method, "saved_prediction": label_metrics(saved, success),
                         "recomputed_prediction": label_metrics(predicted_cost, success),
                         "actual_encoding": label_metrics(actual_cost, success),
                         "actual_encoding_final_success": label_metrics(actual_cost, final),
                         "feature_mse": float(np.square(pred - actual).mean())})
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with FEAT.open("xb") as file:
        np.savez_compressed(file, **arrays)
    with OUT.open("x") as file:
        json.dump({"rows": rows, "features_sha256": file_hash(FEAT), "pred_sha256": file_hash(PRED),
                   "physical_sha256": file_hash(PHYS), "failure_sha256": file_hash(failure.OUTPUT),
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "scope": "post-hoc exposed failure pools; actual successors are evaluation-only"},
                  file, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "physical", "analyze"):
        raise SystemExit("expected predict|physical|analyze")
    {"predict": predict, "physical": physical, "analyze": analyze}[sys.argv[1]]()
