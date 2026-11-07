"""Prospective exposed-development GPU retention and actual action replay."""
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as base
from experiments import pc_tworoom_vector_retention_posthoc_20261107_codex as rule
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = base.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_VECTOR_RETENTION_40DEV_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_vector_retention_40dev_20261107_codex"
POOL = Path(str(PREFIX) + "_pool.npz")
PLAN = Path(str(PREFIX) + "_plan.json")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
RESULT = Path(str(PREFIX) + "_result.json")
METHODS = rule.METHODS


def select_from_union(a_cost, b_indices, b_cost, retained):
    a_cost = np.asarray(a_cost)
    b_indices, b_cost = np.asarray(b_indices), np.asarray(b_cost)
    if (a_cost.shape != (8, 8) or b_indices.ndim != 1 or b_cost.shape != b_indices.shape
            or len(set(b_indices.tolist())) != len(b_indices)
            or not np.isfinite(a_cost).all() or not np.isfinite(b_cost).all()):
        raise ValueError("invalid scored union")
    needed = np.sort((np.asarray(retained)[:, None] * 8 + np.arange(8)).reshape(-1))
    positions = np.searchsorted(b_indices, needed)
    if np.any(positions >= len(b_indices)) or not np.array_equal(b_indices[positions], needed):
        raise ValueError("missing selected prefix scores or unsorted union")
    ai = int(np.argmin(a_cost))
    idx = int(np.argmin(b_cost[positions]))
    return ("a", ai) if a_cost.flat[ai] <= b_cost[positions[idx]] else ("b", int(needed[idx]))


def seal():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("plan output exists")
    previous = json.loads(base.source.PLAN.read_text())
    if previous["pool_sha256"] != file_hash(base.source.POOL):
        raise ValueError("candidate pool drift")
    model, scaler, transform, normalizer = base.source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model/normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, rule.__file__, base.source.PLAN,
                         base.source.POOL, base.source.block.prior.inputs.INPUT,
                         base.source.block.prior.common.CHECKPOINT / "weights.pt",
                         base.source.block.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    arrays, rows = {}, []
    with (np.load(base.source.POOL, allow_pickle=False) as old,
          np.load(base.source.block.prior.inputs.INPUT, allow_pickle=False) as data,
          torch.inference_mode()):
        for root in range(10, 50):
            torch.cuda.synchronize()
            started = time.perf_counter()
            candidates = base.select_prefixes(old[f"r{root}_block_cem_actions"],
                                              old[f"r{root}_block_cem_scores"],
                                              old[f"r{root}_block_random_actions"])
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            wrapper = RGBLeWM(model, audit)

            def score(chains):
                chains = np.asarray(chains, dtype=np.float32).reshape(-1, 25, 2)
                if not np.isfinite(chains).all() or np.max(np.abs(chains)) > 1:
                    raise ValueError("illegal action")
                norm = scaler.transform(chains.reshape(-1, 2)).reshape(-1, 25, 2)
                values = []
                for offset in range(0, len(norm), 32):
                    batch = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                         dtype=torch.float32, device="cuda")
                    value, _ = wrapper.score(context, batch)
                    values.extend(value[0].cpu().tolist())
                return np.asarray(values, dtype=np.float64)

            def chains_for(seed):
                tail = np.random.default_rng(seed).uniform(-1, 1, (8, 3, 2)).astype(np.float32)
                return np.concatenate((np.broadcast_to(candidates[:, None, :10], (8, 8, 10, 2)),
                                       np.broadcast_to(np.repeat(tail, 5, axis=1)[None], (8, 8, 15, 2))), axis=2)

            a, b = chains_for(29111000 + root), chains_for(29112000 + root)
            parent = score(candidates)
            a_cost = score(a).reshape(8, 8)
            retained = rule.retention(a_cost, parent, 29114000 + root)
            b_indices = np.asarray(sorted({p * 8 + i for method in METHODS
                                           for p in retained[method] for i in range(8)}), dtype=np.int64)
            b_cost = score(b.reshape(64, 25, 2)[b_indices])
            decisions = {m: select_from_union(a_cost, b_indices, b_cost, retained[m]) for m in METHODS}
            for name, (bank, index) in decisions.items():
                raw = (a if bank == "a" else b).reshape(64, 25, 2)[index]
                decisions[name] = {"bank": bank, "index": index, "retained": retained[name],
                                   "action_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()}
            arrays.update({f"r{root}_a": a, f"r{root}_b": b, f"r{root}_parent": parent,
                           f"r{root}_a_cost": a_cost, f"r{root}_b_indices": b_indices,
                           f"r{root}_b_cost": b_cost})
            torch.cuda.synchronize()
            rows.append({"root": root, "row": int(data["rows"][root]), "decisions": decisions,
                         "actual_union_b_queries": len(b_indices), "per_arm_queries": 88,
                         "shared_gpu_seconds": time.perf_counter() - started})
            print(f"root={root} A=64 B-union={len(b_indices)} parent=8", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "track": RGB, "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "pool_sha256": file_hash(POOL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "rule_sha256": file_hash(rule.__file__), "source_plan_sha256": file_hash(base.source.PLAN),
                   "source_pool_sha256": file_hash(base.source.POOL),
                   "input_sha256": file_hash(base.source.block.prior.inputs.INPUT),
                   "scope": "previously exposed development roots"}, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("rule_sha256", rule.__file__),
                      ("source_plan_sha256", base.source.PLAN), ("source_pool_sha256", base.source.POOL),
                      ("input_sha256", base.source.block.prior.inputs.INPUT)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding drift: {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(POOL, allow_pickle=False) as pool,
              h5py.File(base.source.block.prior.common.DATA, "r") as data):
            for task in plan["rows"]:
                root, row = task["root"], task["row"]
                outputs = {}
                for method in METHODS:
                    choice = task["decisions"][method]
                    action = pool[f"r{root}_{choice['bank']}"].reshape(64, 25, 2)[choice["index"]]
                    if hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest() != choice["action_sha256"]:
                        raise ValueError("action drift")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    trace = []
                    for value in action:
                        _, _, term, trunc, info = actor.step(value.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outputs[method] = {"trace": trace, "ever": any(x["terminated"] for x in trace),
                                       "prefix_success": any(x["terminated"] for x in trace[:10]),
                                       "final": trace[-1]["terminated"]}
                rows.append({"root": root, "row": row, "outcomes": outputs})
                print(f"root={root} physical complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL)},
                  f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, physical = json.loads(PLAN.read_text()), json.loads(PHYSICAL.read_text())
    if physical["plan_sha256"] != file_hash(PLAN) or physical["pool_sha256"] != file_hash(POOL):
        raise ValueError("physical binding mismatch")
    rows = physical["rows"]
    if [r["root"] for r in rows] != list(range(10, 50)):
        raise ValueError("incomplete panel")
    summary = {}
    for m in METHODS:
        o = [r["outcomes"][m] for r in rows]
        summary[m] = {"ever": sum(x["ever"] for x in o), "final": sum(x["final"] for x in o),
                      "post_prefix_success": sum(x["ever"] and not x["prefix_success"] for x in o),
                      "mean_final_distance": float(np.mean([x["trace"][-1]["distance"] for x in o])),
                      "wins": sum(r["outcomes"][m]["ever"] and not r["outcomes"]["lookahead"]["ever"] for r in rows),
                      "losses": sum(not r["outcomes"][m]["ever"] and r["outcomes"]["lookahead"]["ever"] for r in rows)}
    vec = summary["vector"]
    exit_pass = vec["wins"] >= 3 and vec["losses"] <= 1 and vec["ever"] > summary["shuffled"]["ever"]
    result = {"summary": summary, "development_exit": exit_pass,
              "avg_union_b_queries": float(np.mean([r["actual_union_b_queries"] for r in plan["rows"]])),
              "mean_shared_gpu_seconds": float(np.mean([r["shared_gpu_seconds"] for r in plan["rows"]])),
              "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL),
              "scope": "exposed development only; not independent confirmation"}
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
