"""Exposed development: temporal effect retention, then sealed physical replay."""
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as old
from experiments import pc_margin_temporal_pair_selection_20261107_codex as temporal
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = old.ROOT
BASE = ROOT / "results/pc_temporal_effect_retention_20261107_codex"
PLAN, POOL, PHYS, RESULT = [Path(str(BASE) + x) for x in
                            ("_plan.json", "_pool.npz", "_physical.json", "_result.json")]
PROTOCOL = ROOT.parent / "docs/PC_TEMPORAL_EFFECT_RETENTION_PROTOCOL_20261107_CODEX.md"
METHODS = ("lookahead", "vector", "gated", "shuffled")


def retention(costs, seed):
    costs = np.asarray(costs, dtype=np.float64)
    if costs.shape != (8, 9, 5) or not np.isfinite(costs).all():
        raise ValueError("expected finite 8x9x5 temporal scores")
    post = costs[:, :, 2:].min(axis=-1)
    probes = post[:, 1:]
    order = np.argsort(probes.min(axis=1), kind="stable")
    first = int(order[0])
    effect = post[:, :1] - probes
    effect -= effect.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    shuffled = np.stack([v[rng.permutation(8)] for v in effect])

    def farthest(values, eligible):
        distance = np.square(values - values[first]).sum(axis=1)
        allowed = np.zeros(8, dtype=bool)
        allowed[eligible] = True
        allowed[first] = False
        distance[~allowed] = -np.inf
        return [first, int(np.argmax(distance))]

    return {"lookahead": order[:2].tolist(), "vector": farthest(effect, order),
            "gated": farthest(effect, order[:4]), "shuffled": farthest(shuffled, order[:4])}


def choose(a, b, ids, keep, incumbent):
    a, b, ids, incumbent = map(np.asarray, (a, b, ids, incumbent))
    if (a.shape != (8, 8) or b.shape != ids.shape or incumbent.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(incumbent).all() or np.any(np.diff(ids) <= 0)):
        raise ValueError("invalid candidate costs")
    keep = np.asarray(keep)
    if keep.shape != (2,) or len(set(keep.tolist())) != 2 or np.any((keep < 0) | (keep > 7)):
        raise ValueError("invalid retention")
    wanted = np.sort((keep[:, None] * 8 + np.arange(8)).ravel())
    loc = np.searchsorted(ids, wanted)
    if np.any(loc >= len(ids)) or not np.array_equal(ids[loc], wanted):
        raise ValueError("missing B candidates")
    ai, bi, ii = int(np.argmin(a)), int(np.argmin(b[loc])), int(np.argmin(incumbent))
    return min([(float(incumbent[ii]), "incumbent", ii), (float(a.flat[ai]), "a", ai),
                (float(b[loc[bi]]), "b", int(wanted[bi]))], key=lambda x: x[0])


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def seal():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("sealed output exists")
    oldplan, tplan = json.loads(old.PLAN.read_text()), json.loads(temporal.PLAN.read_text())
    if oldplan["pool_sha256"] != file_hash(old.POOL) or tplan["scores_sha256"] != file_hash(temporal.SCORES):
        raise ValueError("source pool drift")
    if oldplan["source_pool_sha256"] != file_hash(old.base.source.POOL):
        raise ValueError("initial pool drift")
    common, inputs = old.base.source.block.prior.common, old.base.source.block.prior.inputs.INPUT
    if oldplan["input_sha256"] != file_hash(inputs):
        raise ValueError("RGB input drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != oldplan["model_sha256"] or normalizer != oldplan["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, old.PLAN, old.POOL,
                         old.base.source.POOL, temporal.PLAN, temporal.SCORES, inputs,
                         common.CHECKPOINT / "weights.pt"], preprocessing="ImageNet 224 and action scaler",
                         clock="five packets of five steps")
    records = {r["root"]: r for r in oldplan["rows"]}
    rows, arrays = [], {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (np.load(old.POOL, allow_pickle=False) as pool,
          np.load(old.base.source.POOL, allow_pickle=False) as initial,
          np.load(temporal.SCORES, allow_pickle=False) as ts,
          np.load(inputs, allow_pickle=False) as data, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for root in range(20, 50):
            a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
            source_chains = temporal.source.chains_for(pool, root).reshape(8, 9, 25, 2)
            if not np.array_equal(a, source_chains[:, 1:]):
                raise ValueError("temporal probe identities differ from A")
            costs = ts[f"r{root}_scores"]
            keep = retention(costs, 29114000 + root)
            ids = np.asarray(sorted({p * 8 + j for m in METHODS for p in keep[m] for j in range(8)}))
            incumbents = np.asarray([initial[f"r{root}_{arm}_actions"][
                int(np.argmin(initial[f"r{root}_{arm}_scores"]))] for arm in ("block_cem", "block_random")],
                dtype=np.float32)
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            raw = np.concatenate((b.reshape(64, 25, 2)[ids], incumbents))
            if not np.isfinite(raw).all() or np.any(np.abs(raw) > 1):
                raise ValueError("illegal action")
            norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 5, 10)
            scored = []
            for offset in range(0, len(norm), 32):
                batch = torch.as_tensor(norm[None, offset:offset+32], dtype=torch.float32, device="cuda")
                _, output = wrapper.score(context, batch)
                pred, target = output["predicted_emb"], output["goal_emb"]
                if pred.shape[2] != 6:
                    raise ValueError("model clock mismatch")
                scored.extend((pred[:, :, 1:] - target[:, -1, None, None]).square().sum(-1)[0].cpu().numpy())
            scored = np.asarray(scored)
            ac, bc, inc = costs[:, 1:].min(axis=-1), scored[:-2].min(axis=-1), scored[-2:].min(axis=-1)
            decisions, chosen = {}, []
            for method in METHODS:
                value, bank, index = choose(ac, bc, ids, keep[method], inc)
                action = incumbents[index] if bank == "incumbent" else (a if bank == "a" else b).reshape(64, 25, 2)[index]
                chosen.append(action)
                decisions[method] = {"bank": bank, "index": index, "cost": value,
                                     "retained": keep[method], "action_sha256": digest(action)}
            arrays.update({f"r{root}_actions": np.asarray(chosen, dtype=np.float32),
                           f"r{root}_b_ids": ids, f"r{root}_b_times": scored[:-2],
                           f"r{root}_inc_times": scored[-2:]})
            rows.append({"root": root, "row": records[root]["row"], "decisions": decisions,
                         "b_union_queries": len(ids), "logical_per_arm_chains": 600 + 72 + 16 + 2})
            print(f"root={root} B-union={len(ids)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    paths = {"pool": POOL, "old_plan": old.PLAN, "old_pool": old.POOL,
             "initial_pool": old.base.source.POOL, "temporal_plan": temporal.PLAN,
             "temporal_scores": temporal.SCORES, "inputs": inputs, "protocol": PROTOCOL, "runner": __file__}
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "sha256": {k: file_hash(v) for k, v in paths.items()},
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "track": RGB,
                   "new_gpu_seconds_synchronized": elapsed, "scope": "exposed development"}, f, indent=2)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYS.exists():
        raise FileExistsError(PHYS)
    plan = json.loads(PLAN.read_text())
    paths = {"pool": POOL, "old_plan": old.PLAN, "old_pool": old.POOL,
             "initial_pool": old.base.source.POOL, "temporal_plan": temporal.PLAN,
             "temporal_scores": temporal.SCORES, "inputs": old.base.source.block.prior.inputs.INPUT,
             "protocol": PROTOCOL, "runner": __file__}
    if any(plan["sha256"][k] != file_hash(v) for k, v in paths.items()):
        raise ValueError("sealed input drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(POOL, allow_pickle=False) as pool, h5py.File(old.base.source.block.prior.common.DATA, "r") as data:
            for task in plan["rows"]:
                actions, outcomes = pool[f"r{task['root']}_actions"], {}
                for i, method in enumerate(METHODS):
                    if digest(actions[i]) != task["decisions"][method]["action_sha256"]:
                        raise ValueError("action changed")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    trace = []
                    for action in actions[i]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[method] = {"ever": any(x["terminated"] for x in trace),
                                        "prefix_success": any(x["terminated"] for x in trace[:10]),
                                        "final": trace[-1]["terminated"], "trace": trace}
                rows.append({"root": task["root"], "outcomes": outcomes})
                print(f"root={task['root']} replayed", flush=True)
    finally:
        env.close()
    with PHYS.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL), "rows": rows}, f)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, phys = json.loads(PLAN.read_text()), json.loads(PHYS.read_text())
    if phys["plan_sha256"] != file_hash(PLAN) or phys["pool_sha256"] != file_hash(POOL):
        raise ValueError("replay binding mismatch")
    if [x["root"] for x in phys["rows"]] != list(range(20, 50)):
        raise ValueError("incomplete panel")
    summary = {}
    for method in METHODS:
        outcomes = [r["outcomes"][method] for r in phys["rows"]]
        summary[method] = {"ever": sum(x["ever"] for x in outcomes), "final": sum(x["final"] for x in outcomes),
                           "post_prefix": sum(x["ever"] and not x["prefix_success"] for x in outcomes),
                           "mean_final_distance": float(np.mean([x["trace"][-1]["distance"] for x in outcomes])),
                           "sources": {s: sum(r["decisions"][method]["bank"] == s for r in plan["rows"])
                                       for s in ("incumbent", "a", "b")}}
    for baseline in ("lookahead", "shuffled"):
        summary["gated"][f"wins_vs_{baseline}"] = sum(
            r["outcomes"]["gated"]["ever"] and not r["outcomes"][baseline]["ever"] for r in phys["rows"])
        summary["gated"][f"losses_vs_{baseline}"] = sum(
            not r["outcomes"]["gated"]["ever"] and r["outcomes"][baseline]["ever"] for r in phys["rows"])
    passed = all(summary["gated"][f"wins_vs_{m}"] >= 2 and summary["gated"][f"losses_vs_{m}"] <= 1
                 for m in ("lookahead", "shuffled"))
    payload = {"summary": summary, "development_exit": bool(passed), "plan_sha256": file_hash(PLAN),
               "physical_sha256": file_hash(PHYS), "scope": "exposed fixed first-plan development"}
    with RESULT.open("x") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
