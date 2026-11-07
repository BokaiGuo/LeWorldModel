"""Sealed B-tail RGB predictions and evaluator-only physical replay."""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_effect_bank_physical_alignment_20261107_codex as physical
from experiments import pc_temporal_tail_physical_rank_20261107_codex as rank
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

old = physical.prior
BASE = old.ROOT / "results/pc_temporal_b_tail_validation_20261107_codex"
SCORES, SEAL, PHYS, META, RESULT = [Path(str(BASE) + suffix) for suffix in
                                    ("_scores.npz", "_seal.json", "_physical.npz",
                                     "_physical.json", "_result.json")]
PROTOCOL = old.ROOT.parent / "docs/PC_TEMPORAL_B_TAIL_VALIDATION_PROTOCOL_20261107_CODEX.md"
ROOTS = list(range(20, 50))


def validate_bank(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != (8, 8, 25, 2) or b.shape != a.shape or not np.isfinite(b).all() or np.any(np.abs(b) > 1):
        raise ValueError("invalid action bank")
    if not np.array_equal(a[:, :, :10], b[:, :, :10]):
        raise ValueError("A/B prefixes differ")
    tails = np.broadcast_to(b[:1, :, 10:], (8, 8, 15, 2))
    if not np.array_equal(b[:, :, 10:], tails):
        raise ValueError("B tails do not share action identities")
    if np.array_equal(a[0, :, 10:], b[0, :, 10:]):
        raise ValueError("A/B tail banks identical")


def seal():
    if SCORES.exists() or SEAL.exists():
        raise FileExistsError("sealed output exists")
    previous = json.loads(old.PLAN.read_text())
    common, inp = old.base.source.block.prior.common, old.base.source.block.prior.inputs.INPUT
    if previous["pool_sha256"] != file_hash(old.POOL) or previous["input_sha256"] != file_hash(inp):
        raise ValueError("candidate or RGB input drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, old.PLAN, old.POOL, inp,
                         common.CHECKPOINT / "weights.pt"], preprocessing="ImageNet 224, action scaler",
                         clock="five packets of five environment steps")
    rows = {r["root"]: r for r in previous["rows"]}
    arrays, records = {}, []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with np.load(old.POOL, allow_pickle=False) as pool, np.load(inp, allow_pickle=False) as rgb, torch.inference_mode():
        wrapper = RGBLeWM(model, audit)
        for root in ROOTS:
            a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
            validate_bank(a, b)
            raw = b.reshape(64, 25, 2)
            norm = scaler.transform(raw.reshape(-1, 2)).reshape(64, 5, 10)
            context = {"pixels": transform(rgb["current"][root])[None, None, None].cuda(),
                       "goal": transform(rgb["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            values = []
            for offset in range(0, 64, 32):
                action = torch.as_tensor(norm[None, offset:offset+32], dtype=torch.float32, device="cuda")
                endpoint, out = wrapper.score(context, action)
                if out["predicted_emb"].shape[2] != 6:
                    raise ValueError("model clock mismatch")
                scores = (out["predicted_emb"][:, :, 1:] - out["goal_emb"][:, -1, None, None]).square().sum(-1)
                if not torch.allclose(endpoint, scores[:, :, -1], rtol=0, atol=0.001):
                    raise ValueError("endpoint mismatch")
                values.append(scores[0].cpu().numpy())
            values = np.concatenate(values)
            indices, historical = pool[f"r{root}_b_indices"], pool[f"r{root}_b_cost"]
            error = float(np.max(np.abs(values[indices, -1] - historical)))
            if error > 0.005:
                raise ValueError(f"historical B score mismatch {root}: {error}")
            arrays[f"r{root}_scores"] = values.reshape(8, 8, 5)
            records.append({"root": root, "row": rows[root]["row"], "previously_queried": len(indices),
                            "max_endpoint_error": error})
            print(f"root={root} predictions sealed in memory", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with SCORES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    paths = {"scores": SCORES, "candidate_pool": old.POOL, "candidate_plan": old.PLAN,
             "inputs": inp, "runner": __file__, "protocol": PROTOCOL,
             "analysis": rank.__file__, "previous_physical": physical.PHYS}
    with SEAL.open("x") as f:
        json.dump({"roots": ROOTS, "rows": records, "sha256": {k: file_hash(v) for k, v in paths.items()},
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "track": RGB,
                   "gpu_seconds_synchronized": elapsed, "prediction_chains": 64 * len(ROOTS),
                   "scope": "exposed development"}, f, indent=2, allow_nan=False)


def verify():
    saved = json.loads(SEAL.read_text())
    paths = {"scores": SCORES, "candidate_pool": old.POOL, "candidate_plan": old.PLAN,
             "inputs": old.base.source.block.prior.inputs.INPUT, "runner": __file__, "protocol": PROTOCOL,
             "analysis": rank.__file__, "previous_physical": physical.PHYS}
    if saved["roots"] != ROOTS or any(saved["sha256"][k] != file_hash(v) for k, v in paths.items()):
        raise ValueError("prediction seal drift")
    return saved


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    saved = verify()
    physical.verify()
    if PHYS.exists() or META.exists():
        raise FileExistsError("physical output exists")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays, total = {}, 0
    try:
        with (np.load(old.POOL, allow_pickle=False) as pool,
              np.load(physical.PHYS, allow_pickle=False) as old_truth,
              h5py.File(old.base.source.block.prior.common.DATA, "r") as data):
            for record in saved["rows"]:
                root, row = record["root"], record["row"]
                a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
                validate_bank(a, b)
                distances, terminated = np.empty((8, 8, 25)), np.empty((8, 8, 25), dtype=bool)
                for i in range(8):
                    for j in range(8):
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(data["proprio"][row])
                        actor._set_goal_state(data["proprio"][row + 25])
                        for t, action in enumerate(b[i, j]):
                            _, _, term, _, info = actor.step(action.copy())
                            distances[i, j, t] = info["distance_to_target"]
                            terminated[i, j, t] = term
                            total += 1
                old_d = old_truth[f"r{root}_distance"][:, :1, :10]
                old_t = old_truth[f"r{root}_term"][:, :1, :10]
                if (not np.allclose(distances[:, :, :10], old_d, rtol=0, atol=1e-9)
                        or not np.array_equal(terminated[:, :, :10], np.broadcast_to(old_t, (8, 8, 10)))):
                    raise ValueError(f"physical prefix mismatch root={root}")
                arrays[f"r{root}_distance"], arrays[f"r{root}_term"] = distances, terminated
                print(f"root={root} B physical replay=64", flush=True)
    finally:
        env.close()
    with PHYS.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with META.open("x") as f:
        json.dump({"seal_sha256": file_hash(SEAL), "physical_sha256": file_hash(PHYS),
                   "environment_steps": total, "scope": "evaluation only"}, f, indent=2)


def summarize():
    verify()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    meta = json.loads(META.read_text())
    if meta["seal_sha256"] != file_hash(SEAL) or meta["physical_sha256"] != file_hash(PHYS):
        raise ValueError("physical binding mismatch")
    rows = []
    with np.load(SCORES, allow_pickle=False) as predictions, np.load(PHYS, allow_pickle=False) as truth:
        for root in ROOTS:
            scores, dist, term = predictions[f"r{root}_scores"], truth[f"r{root}_distance"], truth[f"r{root}_term"]
            if scores.shape != (8, 8, 5) or dist.shape != (8, 8, 25) or term.shape != (8, 8, 25):
                raise ValueError("invalid stored shape")
            groups = {"all": np.ones(8, dtype=bool),
                      "prefix_unsuccessful": ~term[:, 0, :10].any(axis=-1)}
            distance = dist[:, :, 10:].min(axis=-1)
            success = term[:, :, 10:].any(axis=-1)
            row = {"root": root}
            for name, eligible in groups.items():
                row[name] = {"temporal": rank.metrics(scores[:, :, 2:].min(-1), distance, success, eligible),
                             "endpoint": rank.metrics(scores[:, :, -1], distance, success, eligible)}
            rows.append(row)
    summary = {}
    for group in ("all", "prefix_unsuccessful"):
        summary[group] = {name: rank.aggregate(rows, group, name) for name in ("temporal", "endpoint")}
        valid = [r[group] for r in rows if r[group]["temporal"]["valid_pairs"] and
                 r[group]["endpoint"]["valid_pairs"]]
        accuracy = [(r["temporal"]["correct_pairs"] / r["temporal"]["valid_pairs"],
                     r["endpoint"]["correct_pairs"] / r["endpoint"]["valid_pairs"]) for r in valid]
        success = [(r["temporal"]["chosen_success"], r["endpoint"]["chosen_success"]) for r in valid]
        for key, pairs in (("rank", accuracy), ("success", success)):
            summary[group][f"{key}_task_comparison"] = {
                "wins": sum(x > y for x, y in pairs), "losses": sum(x < y for x, y in pairs),
                "ties": sum(x == y for x, y in pairs)}
    g = summary["prefix_unsuccessful"]
    passed = (g["rank_task_comparison"]["wins"] > g["rank_task_comparison"]["losses"]
              and g["temporal"]["root_macro_accuracy"] > g["endpoint"]["root_macro_accuracy"])
    with RESULT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "development_exit": bool(passed),
                   "prediction_seal_sha256": file_hash(SEAL), "physical_sha256": file_hash(PHYS),
                   "scope": "exposed B tails; no independent gate claim"}, f, indent=2, allow_nan=False)
    print(json.dumps({"summary": summary, "development_exit": bool(passed)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "replay", "summarize"):
        raise SystemExit("expected seal|replay|summarize")
    {"seal": seal, "replay": replay, "summarize": summarize}[sys.argv[1]]()
