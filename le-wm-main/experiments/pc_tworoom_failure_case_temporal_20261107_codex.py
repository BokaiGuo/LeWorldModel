"""Case-selected, evaluation-only predicted/physical RGB temporal alignment."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_full_b_pool_diagnostic_20261107_codex as b
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOTS = (38, 39)
PREFIX = b.q.ROOT / "results/pc_tworoom_failure_case_temporal_20261107_codex"
PRED = Path(str(PREFIX) + "_pred.npz")
META = Path(str(PREFIX) + "_seal.json")
FRAMES = Path(str(PREFIX) + "_frames.npz")
FEATURES = Path(str(PREFIX) + "_features.npz")
RESULT = Path(str(PREFIX) + "_result.json")
PROTOCOL = b.q.ROOT.parent / "docs/PC_TWOROOM_FAILURE_CASE_TEMPORAL_PROTOCOL_20261107_CODEX.md"


def binding():
    return {"runner": Path(__file__), "protocol": PROTOCOL, "full_b_seal": b.SEAL,
            "full_b_physical": b.PHYS, "full_b_result": b.RESULT, "previous_plan": b.q.PLAN,
            "previous_pool": b.q.POOL, "previous_physical": b.q.PHYSICAL,
            "old_pool": b.q.old.POOL, "input": b.q.old.base.source.block.prior.inputs.INPUT,
            "data": b.q.old.base.source.block.prior.common.DATA}


def setup():
    plan = json.loads(b.q.PLAN.read_text())
    model, scaler, transform, normalizer = b.q.old.base.source.block.prior.common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    return plan, model, scaler, transform


def actions(root, bank, previous, plan):
    actions_b = bank[f"r{root}_b"].reshape(64, 25, 2)
    index = next(r for r in plan["rows"] if r["root"] == root)
    selected = previous[f"r{root}_actions"][0]
    if b.q.digest(selected) != index["decisions"]["lookahead"]["action_sha256"]:
        raise ValueError("selected action identity mismatch")
    return np.concatenate((actions_b, selected[None]), axis=0)


def predict():
    if PRED.exists() or META.exists():
        raise FileExistsError("predictions exist")
    b.verify()
    plan, model, scaler, transform = setup()
    common = b.q.old.base.source.block.prior.common
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, b.q.PLAN, b.q.old.POOL,
                         b.q.old.base.source.block.prior.inputs.INPUT, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224; TwoRoom action scaler",
                         clock="initial plus five packets of five environment steps")
    arrays, max_errors = {}, {}
    with np.load(b.q.old.POOL, allow_pickle=False) as bank, np.load(b.q.POOL, allow_pickle=False) as old, \
            np.load(b.SCORES, allow_pickle=False) as scores, \
            np.load(b.q.old.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inp, torch.inference_mode():
        for root in ROOTS:
            raw = actions(root, bank, old, plan)
            if not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1:
                raise ValueError("illegal action")
            normalized = scaler.transform(raw.reshape(-1, 2)).reshape(65, 5, 10)
            ctx = {"pixels": transform(inp["current"][root])[None, None, None].cuda(),
                   "goal": transform(inp["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            wrapper = RGBLeWM(model, audit)
            pred, cost = [], []
            for start in (0, 32, 64):
                batch = torch.as_tensor(normalized[start:start + 32][None], dtype=torch.float32, device="cuda")
                val, out = wrapper.score(ctx, batch)
                pred.append(out["predicted_emb"][0].cpu().numpy())
                cost.append(val[0].cpu().numpy())
            pred, cost = np.concatenate(pred), np.concatenate(cost)
            if pred.shape[:2] != (65, 6):
                raise ValueError("prediction clock mismatch")
            error = float(np.max(np.abs(cost[:64] - scores[f"r{root}_scores"])))
            if error > 0.001:
                raise ValueError("prediction score drift")
            max_errors[str(root)] = error
            arrays[f"r{root}_pred"] = pred
            arrays[f"r{root}_cost"] = cost
            print(f"sealed candidate predictions root={root}, max score drift={error}", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with META.open("x") as f:
        json.dump({"sha256": {k: file_hash(v) for k, v in binding().items()}, "pred_sha256": file_hash(PRED),
                   "model_sha256": plan["model_sha256"], "max_score_errors": max_errors}, f, indent=2)


def verify():
    b.verify()
    meta = json.loads(META.read_text())
    if meta["pred_sha256"] != file_hash(PRED) or any(meta["sha256"][k] != file_hash(v) for k, v in binding().items()):
        raise ValueError("source or prediction binding changed")
    return meta


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    verify()
    if FRAMES.exists():
        raise FileExistsError(FRAMES)
    plan = json.loads(b.q.PLAN.read_text())
    phys = json.loads(b.q.PHYSICAL.read_text())
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays = {}
    try:
        with np.load(b.q.old.POOL, allow_pickle=False) as bank, np.load(b.q.POOL, allow_pickle=False) as old, \
                np.load(b.PHYS, allow_pickle=False) as ref, \
                np.load(b.q.old.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inp, \
                h5py.File(b.q.old.base.source.block.prior.common.DATA, "r") as data:
            for root in ROOTS:
                task = next(r for r in plan["rows"] if r["root"] == root)
                past = next(r for r in phys["rows"] if r["root"] == root)["outcomes"]["lookahead"]["trace"]
                frames, distances, terms, truncs = [], [], [], []
                for i, chain in enumerate(actions(root, bank, old, plan)):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    if i == 0:
                        initial = actor.render().copy()
                        reference = inp["current"][root]
                        if initial.shape != reference.shape:
                            raise ValueError("initial image shape mismatch")
                        arrays[f"r{root}_initial_rgb_maxdiff"] = np.asarray(
                            np.abs(initial.astype(float) - reference.astype(float)).max())
                    ims, dist, term, trunc = [], [], [], []
                    for step, action in enumerate(chain, 1):
                        _, _, done, cut, info = actor.step(action.copy())
                        dist.append(float(info["distance_to_target"]))
                        term.append(bool(done))
                        trunc.append(bool(cut))
                        if step % 5 == 0:
                            ims.append(actor.render().copy())
                    if i < 64:
                        if not (np.array_equal(dist, ref[f"r{root}_distance"][i]) and
                                np.array_equal(term, ref[f"r{root}_term"][i]) and
                                np.array_equal(trunc, ref[f"r{root}_trunc"][i])):
                            raise ValueError(f"physical replay drift {root}/{i}")
                    elif any((dist[t] != past[t]["distance"] or term[t] != past[t]["terminated"] or
                              trunc[t] != past[t]["truncated"]) for t in range(25)):
                        raise ValueError("selected replay drift")
                    frames.append(ims)
                    distances.append(dist)
                    terms.append(term)
                    truncs.append(trunc)
                arrays[f"r{root}_frames"] = np.asarray(frames, dtype=np.uint8)
                arrays[f"r{root}_distance"] = np.asarray(distances)
                arrays[f"r{root}_term"] = np.asarray(terms)
                arrays[f"r{root}_trunc"] = np.asarray(truncs)
                print(f"root={root}: verified 65 full physical trajectories", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, **arrays)


def analyze():
    verify()
    if FEATURES.exists() or RESULT.exists():
        raise FileExistsError("analysis exists")
    plan, model, _, transform = setup()
    arrays, rows = {}, []
    with np.load(PRED, allow_pickle=False) as pred, np.load(FRAMES, allow_pickle=False) as frames, \
            np.load(b.q.old.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs, torch.inference_mode():
        for root in ROOTS:
            goal = model.encode({"pixels": transform(inputs["goal"][root])[None, None].cuda()})["emb"][0, 0].cpu().numpy()
            rgb = frames[f"r{root}_frames"]
            flat = rgb.reshape(325, *rgb.shape[2:])
            features = []
            for start in range(0, len(flat), 32):
                batch = torch.stack([transform(image) for image in flat[start:start + 32]])[:, None].cuda()
                features.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
            actual = np.concatenate(features).reshape(65, 5, -1)
            predicted = pred[f"r{root}_pred"][:, 1:]
            if predicted.shape != actual.shape:
                raise ValueError("latent clock mismatch")
            pred_cost = np.square(predicted - goal).sum(-1)
            actual_cost = np.square(actual - goal).sum(-1)
            cost_error = float(np.max(np.abs(pred_cost[:, -1] - pred[f"r{root}_cost"])))
            if cost_error > 0.01:
                raise ValueError("latent and score mismatch")
            term = frames[f"r{root}_term"]
            success = term[:64].any(axis=1)
            successful = np.flatnonzero(success)
            if not len(successful):
                raise ValueError("expected successful candidate absent")
            indices = {"selected": 64, "model_best_b": int(np.argmin(pred_cost[:64, -1])),
                       "best_success_b_oracle": int(successful[np.argmin(pred_cost[successful, -1])])}
            cases = {}
            for label, i in indices.items():
                cases[label] = {"index": i, "ever": bool(term[i].any()), "final": bool(term[i, -1]),
                                "first_success": int(np.flatnonzero(term[i])[0] + 1) if term[i].any() else None,
                                "pred_cost": pred_cost[i].tolist(), "actual_cost": actual_cost[i].tolist(),
                                "physical_distance": frames[f"r{root}_distance"][i, [4, 9, 14, 19, 24]].tolist(),
                                "latent_mse": np.square(predicted[i] - actual[i]).mean(-1).tolist()}
            arrays[f"r{root}_actual"] = actual
            arrays[f"r{root}_goal"] = goal
            arrays[f"r{root}_pred_cost"] = pred_cost
            arrays[f"r{root}_actual_cost"] = actual_cost
            rows.append({"root": root, "initial_rgb_maxdiff": float(frames[f"r{root}_initial_rgb_maxdiff"]),
                         "score_reconstruction_maxdiff": cost_error, "cases": cases})
            print(f"root={root} encoded 325 aligned observed frames", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with FEATURES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with RESULT.open("x") as f:
        json.dump({"rows": rows, "frame_sha256": file_hash(FRAMES), "features_sha256": file_hash(FEATURES),
                   "seal_sha256": file_hash(META), "scope": "retrospective selected failure cases, evaluator only"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "replay", "analyze"):
        raise SystemExit("expected predict|replay|analyze")
    {"predict": predict, "replay": replay, "analyze": analyze}[sys.argv[1]]()
