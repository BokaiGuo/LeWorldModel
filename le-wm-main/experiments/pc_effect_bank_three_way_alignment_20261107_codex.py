"""Frozen same-chain predictions and evaluator-only physical/encoding alignment."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_effect_bank_physical_alignment_20261107_codex as src
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

BASE = src.prior.ROOT / "results/pc_effect_bank_three_way_alignment_20261107_codex"
PRED, META, FEATURES, RESULT = [Path(str(BASE) + suffix) for suffix in
                               ("_pred.npz", "_pred.json", "_features.npz", "_result.json")]
PROTOCOL = src.prior.ROOT.parent / "docs/PC_EFFECT_BANK_THREE_WAY_ALIGNMENT_PROTOCOL_20261107_CODEX.md"


def chains_for(pool, root):
    prefixes, probes = src.roots_and_chains(pool, root)
    tails = np.concatenate((np.zeros((1, 15, 2), dtype=np.float32), probes))
    return np.concatenate((np.broadcast_to(prefixes[:, None], (8, 9, 10, 2)),
                           np.broadcast_to(tails[None], (8, 9, 15, 2))), axis=2).reshape(72, 25, 2)


def setup():
    plan = json.loads(src.prior.PLAN.read_text())
    common = src.prior.base.source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    return plan, model, scaler, transform


def predict():
    src.verify()
    if PRED.exists() or META.exists():
        raise FileExistsError("prediction output exists")
    start = time.monotonic()
    plan, model, scaler, transform = setup()
    inp = src.prior.base.source.block.prior.inputs.INPUT
    common = src.prior.base.source.block.prior.common
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, src.SEAL, src.prior.POOL, inp,
                                     common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    arrays, errors = {}, []
    with (np.load(src.prior.POOL, allow_pickle=False) as pool,
          np.load(inp, allow_pickle=False) as inputs,
          np.load(src.SCORES, allow_pickle=False) as old, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for root in src.ROOTS:
            actions = chains_for(pool, root)
            norm = scaler.transform(actions.reshape(-1, 2)).reshape(72, 5, 10)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            costs, features = [], []
            for offset in range(0, 72, 32):
                chunk = torch.as_tensor(norm[None, offset:offset + 32], dtype=torch.float32, device="cuda")
                cost, output = wrapper.score(context, chunk)
                costs.append(cost[0].cpu().numpy())
                features.append(output["predicted_emb"][0, :, -1].cpu().numpy())
            cost = np.concatenate(costs).reshape(8, 9)
            error = max(float(np.max(np.abs(cost[:, 0] - old[f"r{root}_ref"]))),
                        float(np.max(np.abs(cost[:, 1:] - old[f"r{root}_probe"]))))
            if error > 0.005:
                raise ValueError(f"score mismatch root={root}: {error}")
            arrays[f"r{root}_predicted"] = np.concatenate(features)
            arrays[f"r{root}_cost"] = cost
            errors.append(error)
            print(f"prediction root={root} error={error:.7g}", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with PRED.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with META.open("x") as handle:
        json.dump({"pred_sha256": file_hash(PRED), "runner_sha256": file_hash(__file__),
                   "protocol_sha256": file_hash(PROTOCOL), "source_seal_sha256": file_hash(src.SEAL),
                   "source_physical_sha256": file_hash(src.PHYS), "model_sha256": plan["model_sha256"],
                   "max_score_error": max(errors), "wall_seconds": time.monotonic() - start},
                  handle, indent=2, allow_nan=False)


def counts(first, second, eligible):
    first, second, eligible = map(np.asarray, (first, second, eligible))
    if first.shape != (8, 8) or second.shape != (8, 8) or eligible.shape != (8,):
        raise ValueError("invalid pair shape")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("nonfinite effect")
    mask = (np.abs(first) > 1e-6) & (np.abs(second) > 1e-6) & eligible[:, None]
    return [int(np.sum(mask & (np.sign(first) == np.sign(second)))), int(mask.sum())]


def replay():
    import gymnasium as gym
    import h5py
    import stable_worldmodel  # noqa: F401

    src.verify()
    if FEATURES.exists() or RESULT.exists():
        raise FileExistsError("evaluation output exists")
    meta = json.loads(META.read_text())
    for key, path in (("pred_sha256", PRED), ("runner_sha256", __file__),
                      ("protocol_sha256", PROTOCOL), ("source_seal_sha256", src.SEAL),
                      ("source_physical_sha256", src.PHYS)):
        if meta[key] != file_hash(path):
            raise ValueError(f"binding mismatch: {key}")
    start = time.monotonic()
    plan, model, _, transform = setup()
    rows_by_root = {row["root"]: row["row"] for row in plan["rows"]}
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays, rows = {}, []
    inp = src.prior.base.source.block.prior.inputs.INPUT
    common = src.prior.base.source.block.prior.common
    try:
        with (np.load(src.prior.POOL, allow_pickle=False) as pool,
              np.load(inp, allow_pickle=False) as inputs,
              np.load(PRED, allow_pickle=False) as pred,
              np.load(src.PHYS, allow_pickle=False) as old,
              h5py.File(common.DATA, "r") as data, torch.inference_mode()):
            for root in src.ROOTS:
                images = []
                actions = chains_for(pool, root)
                reference_distance = old[f"r{root}_distance"].reshape(72, 25)
                reference_term = old[f"r{root}_term"].reshape(72, 25)
                initial_diff = None
                for i, chain in enumerate(actions):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    row = rows_by_root[root]
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    if i == 0:
                        initial = actor.render().copy()
                        initial_diff = float(np.abs(initial.astype(float) - inputs["current"][root].astype(float)).max())
                    for t, action in enumerate(chain):
                        _, _, term, _, info = actor.step(action.copy())
                        if (not np.isclose(info["distance_to_target"], reference_distance[i, t],
                                           rtol=0, atol=1e-9) or term != reference_term[i, t]):
                            raise ValueError(f"physical drift root={root} chain={i} t={t}")
                    images.append(actor.render().copy())
                goal = model.encode({"pixels": transform(inputs["goal"][root])[None, None].cuda()})["emb"][0, 0]
                actual = []
                for offset in range(0, 72, 32):
                    batch = torch.stack([transform(frame) for frame in images[offset:offset + 32]])[:, None].cuda()
                    actual.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
                actual = np.concatenate(actual)
                predicted = pred[f"r{root}_predicted"]
                if actual.shape != predicted.shape:
                    raise ValueError("feature shape mismatch")
                encoded_cost = np.square(actual - goal.cpu().numpy()).sum(axis=-1).reshape(8, 9)
                model_cost = pred[f"r{root}_cost"]
                distance = reference_distance[:, -1].reshape(8, 9)
                effects = [x[:, :1] - x[:, 1:] for x in (model_cost, encoded_cost, distance)]
                eligible = ~reference_term.reshape(8, 9, 25)[:, 0, :10].any(axis=1)
                comparisons = {}
                for name, a, b in (("model_encoded", 0, 1), ("encoded_physical", 1, 2),
                                   ("model_physical", 0, 2)):
                    comparisons[name] = {"all": counts(effects[a], effects[b], np.ones(8, dtype=bool)),
                                         "prefix_unsuccessful": counts(effects[a], effects[b], eligible)}
                rows.append({"root": root, "comparisons": comparisons,
                             "feature_mse": float(np.square(predicted - actual).mean()),
                             "initial_rgb_max_abs_diff": initial_diff})
                arrays[f"r{root}_actual"] = actual
                arrays[f"r{root}_goal"] = goal.cpu().numpy()
                arrays[f"r{root}_encoded_cost"] = encoded_cost
                print(f"replay root={root} verified=72", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    with FEATURES.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    summary = {}
    for name in ("model_encoded", "encoded_physical", "model_physical"):
        summary[name] = {}
        for group in ("all", "prefix_unsuccessful"):
            values = [row["comparisons"][name][group] for row in rows]
            nonempty = [v for v in values if v[1]]
            summary[name][group] = {"roots": len(nonempty), "matching": sum(v[0] for v in values),
                                    "pairs": sum(v[1] for v in values),
                                    "macro_accuracy": float(np.mean([a / b for a, b in nonempty])) if nonempty else None}
    summary["mean_feature_mse"] = float(np.mean([r["feature_mse"] for r in rows]))
    with RESULT.open("x") as handle:
        json.dump({"rows": rows, "summary": summary, "features_sha256": file_hash(FEATURES),
                   "prediction_meta_sha256": file_hash(META), "runner_sha256": file_hash(__file__),
                   "wall_seconds": time.monotonic() - start,
                   "scope": "exposed development; real state and successor evaluation only"},
                  handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "replay"):
        raise SystemExit("expected predict|replay")
    {"predict": predict, "replay": replay}[sys.argv[1]]()
