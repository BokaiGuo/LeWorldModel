"""Budget-scoped two-stage retention with source-incumbent preservation."""
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as prev
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit

ROOT = prev.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_VECTOR_BUDGET290_DEV_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_vector_budget290_dev_20261107_codex"
POOL = Path(str(PREFIX) + "_pool.npz")
PLAN = Path(str(PREFIX) + "_plan.json")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
RESULT = Path(str(PREFIX) + "_result.json")


def choose(a_cost, incumbent_cost, ids, b_cost, retained):
    """Only the selected prefixes' queried B indices may participate."""
    a, inc, ids, b = map(np.asarray, (a_cost, incumbent_cost, ids, b_cost))
    if (a.shape != (8, 8) or inc.shape != (2,) or ids.ndim != 1 or b.shape != ids.shape
            or not np.isfinite(a).all() or not np.isfinite(inc).all() or not np.isfinite(b).all()
            or np.any(np.diff(ids) <= 0)):
        raise ValueError("invalid scores")
    keep = np.asarray(retained, dtype=int)
    if keep.shape != (2,) or len(set(keep.tolist())) != 2 or np.any(keep < 0) or np.any(keep >= 8):
        raise ValueError("invalid retained prefixes")
    wanted = np.sort((keep[:, None] * 8 + np.arange(8)).ravel())
    positions = np.searchsorted(ids, wanted)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], wanted):
        raise ValueError("missing B scores")
    aidx, bpos = int(np.argmin(a)), int(np.argmin(b[positions]))
    options = (float(a.flat[aidx]), float(inc[0]), float(inc[1]), float(b[positions[bpos]]))
    which = int(np.argmin(options))
    return (("a", aidx), ("inc", 0), ("inc", 1), ("b", int(wanted[bpos])))[which]


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def seal():
    if POOL.exists() or PLAN.exists():
        raise FileExistsError("output exists")
    original = json.loads(prev.base.source.PLAN.read_text())
    if original["pool_sha256"] != file_hash(prev.base.source.POOL):
        raise ValueError("old pool changed")
    model, scaler, transform, normalizer = prev.base.source.block.prior.common.model_setup()
    fingerprint = prev.model_fingerprint(model)
    if fingerprint != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model/normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prev.rule.__file__,
                         prev.base.source.PLAN, prev.base.source.POOL,
                         prev.base.source.block.prior.inputs.INPUT,
                         prev.base.source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    arrays, rows = {}, []
    with (np.load(prev.base.source.POOL, allow_pickle=False) as old,
          np.load(prev.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in range(10, 50):
            torch.cuda.synchronize()
            started = time.perf_counter()
            cem = old[f"r{root}_block_cem_actions"][:100]
            rand = old[f"r{root}_block_random_actions"][:100]
            cem_cost = old[f"r{root}_block_cem_scores"][:100]
            rand_cost = old[f"r{root}_block_random_scores"][:100]
            parents = prev.base.select_prefixes(cem, cem_cost, rand)
            inc = np.stack([cem[int(np.argmin(cem_cost))], rand[int(np.argmin(rand_cost))]])
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            wrapper = RGBLeWM(model, audit)

            def score(raw):
                raw = np.asarray(raw, dtype=np.float32).reshape(-1, 25, 2)
                if not np.isfinite(raw).all() or np.abs(raw).max() > 1:
                    raise ValueError("illegal raw action")
                norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 25, 2)
                result = []
                for offset in range(0, len(raw), 32):
                    batch = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                         dtype=torch.float32, device="cuda")
                    val, _ = wrapper.score(ctx, batch)
                    result.extend(val[0].cpu().tolist())
                return np.asarray(result)

            def chains(seed):
                tail = np.random.default_rng(seed).uniform(-1, 1, (8, 3, 2)).astype(np.float32)
                return np.concatenate((np.broadcast_to(parents[:, None, :10], (8, 8, 10, 2)),
                                       np.broadcast_to(np.repeat(tail, 5, axis=1)[None], (8, 8, 15, 2))), axis=2)

            a, b = chains(29111000 + root), chains(29112000 + root)
            parent_inc_cost = score(np.concatenate((parents, inc)))
            a_cost = score(a).reshape(8, 8)
            keep = prev.rule.retention(a_cost, parent_inc_cost[:8], 29114000 + root)
            ids = np.asarray(sorted({p * 8 + j for m in prev.METHODS for p in keep[m] for j in range(8)}))
            b_cost = score(b.reshape(64, 25, 2)[ids])
            decisions = {}
            for method in prev.METHODS:
                bank, index = choose(a_cost, parent_inc_cost[8:], ids, b_cost, keep[method])
                selected = {"a": a.reshape(64, 25, 2), "b": b.reshape(64, 25, 2), "inc": inc}[bank][index]
                decisions[method] = {"bank": bank, "index": index, "retained": keep[method],
                                     "action_sha256": digest(selected)}
            arrays.update({f"r{root}_a": a, f"r{root}_b": b, f"r{root}_inc": inc,
                           f"r{root}_a_cost": a_cost, f"r{root}_parent_inc_cost": parent_inc_cost,
                           f"r{root}_b_indices": ids, f"r{root}_b_cost": b_cost})
            torch.cuda.synchronize()
            rows.append({"root": root, "row": int(inputs["rows"][root]), "decisions": decisions,
                         "incumbent_source_indices": [int(np.argmin(cem_cost)), int(np.argmin(rand_cost))],
                         "logical_queries_per_arm": 290, "shared_new_queries": 74 + len(ids),
                         "shared_seconds": time.perf_counter() - started})
            print(f"root={root} added=74+{len(ids)}", flush=True)
    if prev.model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "track": RGB, "pool_sha256": file_hash(POOL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "rule_sha256": file_hash(prev.rule.__file__), "source_plan_sha256": file_hash(prev.base.source.PLAN),
                   "source_pool_sha256": file_hash(prev.base.source.POOL), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "scope": "exposed development"}, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("rule_sha256", prev.rule.__file__),
                      ("source_plan_sha256", prev.base.source.PLAN),
                      ("source_pool_sha256", prev.base.source.POOL)):
        if plan[key] != file_hash(path):
            raise ValueError("plan binding mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(POOL, allow_pickle=False) as pool,
              h5py.File(prev.base.source.block.prior.common.DATA, "r") as data):
            for item in plan["rows"]:
                root, task = item["root"], item["row"]
                outcomes = {}
                for method in prev.METHODS:
                    decision = item["decisions"][method]
                    actions = pool[f"r{root}_{decision['bank']}"].reshape(-1, 25, 2)[decision["index"]]
                    if digest(actions) != decision["action_sha256"]:
                        raise ValueError("action hash mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[method] = {"trace": trace, "ever": any(x["terminated"] for x in trace),
                                        "prefix_success": any(x["terminated"] for x in trace[:10]),
                                        "final": trace[-1]["terminated"]}
                rows.append({"root": root, "row": task, "outcomes": outcomes})
                print(f"root={root} replayed", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN)}, f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(PLAN.read_text())
    phys = json.loads(PHYSICAL.read_text())
    baseline = json.loads(prev.base.source.PHYSICAL.read_text())
    if phys["plan_sha256"] != file_hash(PLAN) or baseline["plan_sha256"] != file_hash(prev.base.source.PLAN):
        raise ValueError("physical source mismatch")
    if [r["root"] for r in phys["rows"]] != list(range(10, 50)):
        raise ValueError("incomplete tasks")
    summary = {}
    for name in prev.METHODS:
        pairs = [(r["outcomes"][name], baseline["rows"][r["root"]]["outcomes"]["block_cem"])
                 for r in phys["rows"]]
        summary[name] = {"ever": sum(x["ever"] for x, _ in pairs),
                         "final": sum(x["final"] for x, _ in pairs),
                         "post_prefix": sum(x["ever"] and not x["prefix_success"] for x, _ in pairs),
                         "wins_vs_cem": sum(x["ever"] and not y["ever"] for x, y in pairs),
                         "losses_vs_cem": sum(not x["ever"] and y["ever"] for x, y in pairs)}
    vec = summary["vector"]
    exit_pass = (vec["wins_vs_cem"] >= 2 and vec["losses_vs_cem"] <= 1
                 and vec["ever"] > summary["lookahead"]["ever"]
                 and vec["ever"] > summary["shuffled"]["ever"])
    result = {"summary": summary, "development_exit": exit_pass,
              "avg_b_union": float(np.mean([r["shared_new_queries"] - 74 for r in plan["rows"]])),
              "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL),
              "baseline_physical_sha256": file_hash(prev.base.source.PHYSICAL),
              "scope": "exposed development, not gate evidence"}
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
