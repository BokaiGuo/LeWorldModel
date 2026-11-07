"""Fixed-prefix uniform versus A-conditioned local suffix development test."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_quality_gated_retention_20261107_codex as q
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

PREFIX = q.ROOT / "results/pc_tworoom_local_b_tail_20261107_codex"
PLAN = Path(str(PREFIX) + "_plan.json")
POOL = Path(str(PREFIX) + "_pool.npz")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
PROTOCOL = q.ROOT.parent / "docs/PC_TWOROOM_LOCAL_B_TAIL_PROTOCOL_20261107_CODEX.md"


def local_chains(a, costs, keep, seed):
    a = np.asarray(a, dtype=np.float32)
    costs = np.asarray(costs, dtype=np.float64)
    keep = np.asarray(keep, dtype=np.int64)
    if (a.shape != (8, 8, 25, 2) or costs.shape != (8, 8) or keep.shape != (2,)
            or len(set(keep.tolist())) != 2 or np.any(keep < 0) or np.any(keep > 7)
            or not np.isfinite(a).all() or np.any(np.abs(a) > 1) or not np.isfinite(costs).all()):
        raise ValueError("invalid source pool")
    rng = np.random.default_rng(seed)
    out, best_ids = [], []
    for parent in keep:
        best = int(np.argmin(costs[parent]))
        nominal = a[parent, best]
        tail = nominal[10:].reshape(3, 5, 2)
        if not np.array_equal(tail, np.broadcast_to(tail[:, :1], tail.shape)):
            raise ValueError("nominal tail not block constant")
        noise = rng.normal(0, 0.25, (8, 3, 2))
        blocks = np.clip(tail[:, 0][None] + noise, -1, 1).astype(np.float32)
        suffix = np.repeat(blocks, 5, axis=1)
        out.append(np.concatenate((np.broadcast_to(nominal[None, :10], (8, 10, 2)), suffix), axis=1))
        best_ids.append(best)
    return np.concatenate(out), best_ids


def plan():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("output already exists")
    previous = json.loads(q.PLAN.read_text())
    if previous["pool_sha256"] != file_hash(q.POOL) or previous["old_pool_sha256"] != file_hash(q.old.POOL):
        raise ValueError("source pool drift")
    common = q.old.base.source.block.prior.common
    input_path = q.old.base.source.block.prior.inputs.INPUT
    if previous["input_sha256"] != file_hash(input_path):
        raise ValueError("input drift")
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, q.PLAN, q.POOL, q.old.POOL, input_path,
                         common.CHECKPOINT / "weights.pt"], preprocessing="ImageNet 224, TwoRoom scaler",
                         clock="25 environment steps, 5-step packets")
    arrays, rows = {}, []
    with np.load(q.old.POOL, allow_pickle=False) as old, np.load(q.POOL, allow_pickle=False) as prior, \
            np.load(input_path, allow_pickle=False) as inputs, torch.inference_mode():
        for row in previous["rows"]:
            root = row["root"]
            original = row["decisions"]["lookahead"]
            ac = prior[f"r{root}_a_cost"]
            a = old[f"r{root}_a"]
            keep = q.choose_retained(ac, 29114000 + root)["lookahead"]
            if keep != original["retained"]:
                raise ValueError("retention drift")
            new_actions, centers = local_chains(a, ac, keep, 29132000 + root)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            normalized = scaler.transform(new_actions.reshape(-1, 2)).reshape(16, 5, 10)
            session = RGBObservationScoringSession(model, audit)
            scores, _ = session.score(context, torch.as_tensor(normalized[None], device="cuda", dtype=torch.float32))
            costs = scores[0].cpu().numpy().astype(np.float64)
            uniform = prior[f"r{root}_actions"][0]
            if q.digest(uniform) != original["action_sha256"]:
                raise ValueError("uniform action drift")
            local_idx = int(np.argmin(costs))
            common_action = uniform
            # Uniform selected B is not a common candidate. Recover common incumbent or A explicitly.
            if original["bank"] == "b":
                raise ValueError("prior lookahead selected B; common best must be reconstructed")
            common_cost = float(original["cost"])
            if float(costs[local_idx]) < common_cost:
                selected, bank, score = new_actions[local_idx], "local_b", float(costs[local_idx])
            else:
                selected, bank, score = common_action, original["bank"], common_cost
            arrays[f"r{root}_local_candidates"] = new_actions
            arrays[f"r{root}_local_cost"] = costs
            arrays[f"r{root}_selected"] = np.stack((uniform, selected))
            rows.append({"root": root, "row": row["row"], "keep": keep, "a_centers": centers,
                         "uniform_score": common_cost, "uniform_action_sha256": q.digest(uniform),
                         "local_bank": bank, "local_index": local_idx, "local_score": score,
                         "local_action_sha256": q.digest(selected), "logical_queries_each": 690,
                         "new_model_queries": 16})
            print(f"root={root} selected={bank}", flush=True)
    if model_fingerprint(model) != previous["model_sha256"]:
        raise ValueError("model changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": RGB, "rows": rows, "pool_sha256": file_hash(POOL),
                   "source_sha256": {str(p): file_hash(p) for p in (q.PLAN, q.POOL, q.old.POOL, input_path,
                                                                      PROTOCOL, Path(__file__))},
                   "model_sha256": previous["model_sha256"], "normalizer_sha256": normalizer,
                   "scope": "exposed development; fixed ordinary lookahead, not descriptor gain"},
                  f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    sealed = json.loads(PLAN.read_text())
    if sealed["pool_sha256"] != file_hash(POOL):
        raise ValueError("pool drift")
    if any(file_hash(Path(p)) != expected for p, expected in sealed["source_sha256"].items()):
        raise ValueError("source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(POOL, allow_pickle=False) as pool, \
                h5py.File(q.old.base.source.block.prior.common.DATA, "r") as data:
            for item in sealed["rows"]:
                actions = pool[f"r{item['root']}_selected"]
                outcomes = {}
                for i, name in enumerate(("uniform", "local")):
                    if q.digest(actions[i]) != item[f"{name}_action_sha256"]:
                        raise ValueError("selected action drift")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][item["row"]])
                    actor._set_goal_state(data["proprio"][item["row"] + 25])
                    trace = []
                    for action in actions[i]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if term or trunc:
                            break
                    outcomes[name] = {"ever_success": any(x["terminated"] for x in trace), "trace": trace}
                rows.append({"root": item["root"], "outcomes": outcomes})
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL), "rows": rows},
                  f, separators=(",", ":"), allow_nan=False)
    for name in ("uniform", "local"):
        print(name, sum(x["outcomes"][name]["ever_success"] for x in rows), "/", len(rows))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
