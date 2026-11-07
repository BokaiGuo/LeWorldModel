"""Fixed-bank single-factor retention experiment, choices sealed before replay."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as old
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

ROOT = old.ROOT
PREFIX = ROOT / "results/pc_tworoom_quality_gated_retention_20261107_codex"
PLAN, POOL = Path(str(PREFIX) + "_plan.json"), Path(str(PREFIX) + "_pool.npz")
PHYSICAL, RESULT = Path(str(PREFIX) + "_physical.json"), Path(str(PREFIX) + "_result.json")
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_QUALITY_GATED_RETENTION_PROTOCOL_20261107_CODEX.md"
METHODS = ("lookahead", "vector", "gated", "shuffled")


def choose_retained(costs, seed):
    costs = np.asarray(costs, dtype=np.float64)
    if costs.shape != (8, 8) or not np.isfinite(costs).all():
        raise ValueError("expected finite 8x8 costs")
    order = np.argsort(costs.min(axis=1), kind="stable")
    first = int(order[0])
    centered = costs - costs.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    shuffled = np.stack([row[rng.permutation(8)] for row in centered])

    def farthest(v, eligible):
        distances = np.square(v - v[first]).sum(axis=1)
        allowed = np.zeros(8, dtype=bool)
        allowed[eligible] = True
        allowed[first] = False
        distances[~allowed] = -np.inf
        return [first, int(np.argmax(distances))]

    return {"lookahead": order[:2].tolist(),
            "vector": farthest(centered, order),
            "gated": farthest(centered, order[:4]),
            "shuffled": farthest(shuffled, order[:4])}


def select(cost_a, cost_b, b_indices, keep, incumbent_costs):
    a = np.asarray(cost_a)
    b, ids = np.asarray(cost_b), np.asarray(b_indices)
    incumbents = np.asarray(incumbent_costs)
    if (a.shape != (8, 8) or ids.ndim != 1 or b.shape != ids.shape
            or incumbents.shape != (2,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(incumbents).all()
            or len(set(ids.tolist())) != len(ids) or np.any(np.diff(ids) <= 0)):
        raise ValueError("invalid score pool")
    keep = np.asarray(keep)
    if keep.shape != (2,) or len(set(keep.tolist())) != 2 or np.any(keep < 0) or np.any(keep > 7):
        raise ValueError("invalid retention")
    wanted = np.sort((keep[:, None] * 8 + np.arange(8)).ravel())
    loc = np.searchsorted(ids, wanted)
    if np.any(loc >= len(ids)) or not np.array_equal(ids[loc], wanted):
        raise ValueError("missing B scores")
    ai, bi, ii = int(np.argmin(a)), int(np.argmin(b[loc])), int(np.argmin(incumbents))
    choices = [(float(incumbents[ii]), "incumbent", ii),
               (float(a.flat[ai]), "a", ai), (float(b[loc[bi]]), "b", int(wanted[bi]))]
    return min(choices, key=lambda x: x[0])


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def seal():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("sealed artifact exists")
    previous = json.loads(old.PLAN.read_text())
    source_plan = json.loads(old.base.source.PLAN.read_text())
    if (previous["pool_sha256"] != file_hash(old.POOL)
            or previous["source_pool_sha256"] != file_hash(old.base.source.POOL)
            or source_plan["pool_sha256"] != file_hash(old.base.source.POOL)):
        raise ValueError("candidate integrity failure")
    common = old.base.source.block.prior.common
    inputs_path = old.base.source.block.prior.inputs.INPUT
    if file_hash(inputs_path) != previous["input_sha256"]:
        raise ValueError("input drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, old.PLAN, old.POOL,
                         old.base.source.POOL, inputs_path, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    arrays, rows = {}, []
    with (np.load(old.POOL, allow_pickle=False) as pool,
          np.load(old.base.source.POOL, allow_pickle=False) as initial,
          np.load(inputs_path, allow_pickle=False) as data, torch.inference_mode()):
        for record in previous["rows"]:
            root = record["root"]
            parents = old.base.select_prefixes(initial[f"r{root}_block_cem_actions"],
                                                initial[f"r{root}_block_cem_scores"],
                                                initial[f"r{root}_block_random_actions"])
            incumbents = []
            for arm in ("block_cem", "block_random"):
                values = initial[f"r{root}_{arm}_scores"]
                incumbents.append(initial[f"r{root}_{arm}_actions"][int(np.argmin(values))])
            incumbents = np.asarray(incumbents, dtype=np.float32)
            a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            wrapper = RGBObservationScoringSession(model, audit)

            def score(actions):
                raw = np.asarray(actions, dtype=np.float32).reshape(-1, 25, 2)
                if not np.isfinite(raw).all() or np.any(np.abs(raw) > 1):
                    raise ValueError("illegal actions")
                norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 5, 10)
                out = []
                for offset in range(0, len(raw), 32):
                    value, _ = wrapper.score(context, torch.as_tensor(norm[offset:offset + 32][None],
                                                                      dtype=torch.float32, device="cuda"))
                    out.extend(value[0].cpu().tolist())
                return np.asarray(out, np.float64)

            fixed = score(np.concatenate((a.reshape(64, 25, 2), parents, incumbents)))
            ac, parent_cost, inc_cost = fixed[:64].reshape(8, 8), fixed[64:72], fixed[72:74]
            keep = choose_retained(ac, 29114000 + root)
            ids = np.asarray(sorted({p * 8 + j for name in METHODS for p in keep[name] for j in range(8)}))
            bc = score(b.reshape(64, 25, 2)[ids])
            decisions, chosen_actions = {}, []
            for method in METHODS:
                cost, bank, index = select(ac, bc, ids, keep[method], inc_cost)
                selected = (incumbents[index] if bank == "incumbent"
                            else (a if bank == "a" else b).reshape(64, 25, 2)[index])
                chosen_actions.append(selected)
                decisions[method] = {"bank": bank, "index": index, "cost": cost,
                                     "retained": keep[method], "action_sha256": digest(selected)}
            arrays[f"r{root}_actions"] = np.asarray(chosen_actions, dtype=np.float32)
            arrays[f"r{root}_a_cost"] = ac
            arrays[f"r{root}_b_ids"] = ids
            arrays[f"r{root}_b_cost"] = bc
            arrays[f"r{root}_inc_cost"] = inc_cost
            rows.append({"root": root, "row": record["row"], "decisions": decisions,
                         "union_b_queries": len(ids), "logical_per_arm_queries": 690,
                         "parent_cost": parent_cost.tolist()})
            print(f"root={root} B-union={len(ids)}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with POOL.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "pool_sha256": file_hash(POOL), "old_pool_sha256": file_hash(old.POOL),
                   "old_plan_sha256": file_hash(old.PLAN), "source_pool_sha256": file_hash(old.base.source.POOL),
                   "input_sha256": file_hash(inputs_path), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "track": RGB}, stream, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("old_pool_sha256", old.POOL), ("old_plan_sha256", old.PLAN),
                      ("source_pool_sha256", old.base.source.POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding drift: {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(POOL, allow_pickle=False) as pool,
              h5py.File(old.base.source.block.prior.common.DATA, "r") as data):
            for task in plan["rows"]:
                actions, outcomes = pool[f"r{task['root']}_actions"], {}
                for i, method in enumerate(METHODS):
                    if digest(actions[i]) != task["decisions"][method]["action_sha256"]:
                        raise ValueError("action hash mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    trace = []
                    for action in actions[i]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[method] = {"ever": any(t["terminated"] for t in trace),
                                        "final": trace[-1]["terminated"], "trace": trace}
                rows.append({"root": task["root"], "outcomes": outcomes})
                print(f"root={task['root']} physical complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL), "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, phys = json.loads(PLAN.read_text()), json.loads(PHYSICAL.read_text())
    if phys["plan_sha256"] != file_hash(PLAN) or phys["pool_sha256"] != file_hash(POOL):
        raise ValueError("physical drift")
    if [r["root"] for r in plan["rows"]] != list(range(10, 50)):
        raise ValueError("panel incomplete")
    result = {}
    for method in METHODS:
        choices = [r["outcomes"][method]["ever"] for r in phys["rows"]]
        gated = [r["outcomes"]["gated"]["ever"] for r in phys["rows"]]
        result[method] = {"ever": sum(choices),
                          "final": sum(r["outcomes"][method]["final"] for r in phys["rows"]),
                          "gated_wins": sum(a and not b for a, b in zip(gated, choices)),
                          "gated_losses": sum(b and not a for a, b in zip(gated, choices)),
                          "sources": {bank: sum(r["decisions"][method]["bank"] == bank for r in plan["rows"])
                                      for bank in ("incumbent", "a", "b")}}
    passed = all(result[m]["gated_wins"] >= 2 and result[m]["gated_losses"] <= 1
                 for m in ("lookahead", "vector", "shuffled"))
    payload = {"summary": result, "development_exit": passed, "plan_sha256": file_hash(PLAN),
               "physical_sha256": file_hash(PHYSICAL), "scope": "previously exposed development"}
    with RESULT.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
