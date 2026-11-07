"""Score matched local perturbations; seal choices before physical replay."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_LOCAL_PERTURB_STABILITY_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_local_perturb_stability_20261107_codex"
POOL = Path(str(PREFIX) + "_pool.npz")
PLAN = Path(str(PREFIX) + "_plan.json")
OUT = Path(str(PREFIX) + "_physical.json")


def perturb(parents, seed):
    parents = np.asarray(parents, dtype=np.float32)
    if parents.shape != (8, 25, 2) or not np.isfinite(parents).all() or np.abs(parents).max() > 1:
        raise ValueError("invalid parents")
    tails = parents[:, 10:, :].reshape(8, 3, 5, 2)
    if not np.all(tails == tails[:, :, :1, :]):
        raise ValueError("parents not constant within packets")
    rng = np.random.default_rng(seed)
    original = tails[:, :, 0, :]
    raw = original[:, None] + rng.normal(0, 0.05, (8, 4, 3, 2))
    blocks = np.clip(raw, -1, 1).astype(np.float32)
    chains = np.concatenate((np.broadcast_to(parents[:, None, :10], (8, 4, 10, 2)),
                             np.repeat(blocks, 5, axis=2)), axis=2)
    return chains, int(np.count_nonzero((raw < -1) | (raw > 1)))


def choose(cost, perturb_cost, seed):
    cost = np.asarray(cost, dtype=np.float64)
    perturb_cost = np.asarray(perturb_cost, dtype=np.float64)
    if cost.shape != (64,) or perturb_cost.shape != (8, 4) or not np.isfinite(cost).all() or not np.isfinite(perturb_cost).all():
        raise ValueError("invalid scores")
    indices = np.argsort(cost, kind="stable")[:8]
    permutation = np.random.default_rng(seed).permutation(8)
    # A cyclic permutation in random order is a guaranteed derangement.
    mapping = np.empty(8, dtype=int)
    mapping[permutation] = np.roll(permutation, 1)
    robust = (cost[indices] + perturb_cost.sum(axis=1)) / 5
    shuffled = (cost[indices] + perturb_cost[mapping].sum(axis=1)) / 5
    direct = int(np.argmin(np.concatenate((cost, perturb_cost.ravel()))))
    return {"parents": indices.tolist(), "direct": direct,
            "robust": int(indices[np.argmin(robust)]),
            "shuffled": int(indices[np.argmin(shuffled)]), "mapping": mapping.tolist(),
            "robust_cost": robust.tolist(), "shuffled_cost": shuffled.tolist()}


def plan():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("sealed output exists")
    prior = json.loads(source.PLAN.read_text())
    if prior["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("source pool drift")
    common = source.source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    model_hash = model_fingerprint(model)
    if model_hash != prior["model_sha256"] or normalizer != prior["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.PLAN, source.POOL,
                         source.source.block.prior.inputs.INPUT, common.CHECKPOINT / "weights.pt",
                         common.CHECKPOINT / "config.json"], preprocessing="TwoRoom RGB 224 and action scaler",
                         clock="five packets of five environment steps")
    wrapper = RGBLeWM(model, audit)
    arrays, rows = {}, []
    with (np.load(source.POOL, allow_pickle=False) as old,
          np.load(source.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in range(10):
            base = old[f"r{root}_b_actions"].reshape(64, 25, 2)
            costs = old[f"r{root}_b_cost"].reshape(64)
            parents = np.argsort(costs, kind="stable")[:8]
            children, clipped = perturb(base[parents], 29151000 + root)
            normalized = scaler.transform(children.reshape(-1, 2)).reshape(1, 32, 5, 10)
            current = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            scores, _ = wrapper.score(current, torch.as_tensor(normalized, dtype=torch.float32, device="cuda"))
            scores = scores.detach().cpu().numpy().reshape(8, 4)
            decision = choose(costs, scores, 29152000 + root)
            if decision["parents"] != parents.tolist():
                raise ValueError("parent identity mismatch")
            arrays[f"r{root}_children"] = children
            arrays[f"r{root}_child_cost"] = scores
            rows.append({"root": root, "row": int(inputs["rows"][root]), "clipped": clipped, **decision})
            print(f"root={root} scored 32 local perturbations", flush=True)
    if model_fingerprint(model) != model_hash:
        raise ValueError("model drift")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    atomic_write_json(PLAN, {"rows": rows, "source_plan_sha256": file_hash(source.PLAN),
                            "source_pool_sha256": file_hash(source.POOL), "pool_sha256": file_hash(POOL),
                            "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                            "model_sha256": model_hash, "normalizer_sha256": normalizer, "track": RGB})


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    plan_record = json.loads(PLAN.read_text())
    for key, path in (("source_plan_sha256", source.PLAN), ("source_pool_sha256", source.POOL),
                      ("pool_sha256", POOL), ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if plan_record[key] != file_hash(path):
            raise ValueError(f"dependency drift: {key}")
    common = source.source.block.prior.common
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(source.POOL, allow_pickle=False) as old, np.load(POOL, allow_pickle=False) as new,
              h5py.File(common.DATA, "r") as data):
            for row in plan_record["rows"]:
                root, offset = row["root"], row["row"]
                base = old[f"r{root}_b_actions"].reshape(64, 25, 2)
                children = new[f"r{root}_children"].reshape(32, 25, 2)
                choices = {"direct": np.concatenate((base, children))[row["direct"]],
                           "robust": base[row["robust"]], "shuffled": base[row["shuffled"]]}
                result = {}
                for name, chain in choices.items():
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][offset])
                    actor._set_goal_state(data["proprio"][offset + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        if trunc:
                            raise ValueError("unexpected truncation")
                        trace.append({"distance": float(info["distance_to_target"]), "success": bool(term)})
                    result[name] = {"ever": any(x["success"] for x in trace),
                                    "prefix_success": any(x["success"] for x in trace[:10]),
                                    "final_success": trace[-1]["success"], "trace": trace}
                rows.append({"root": root, "arms": result})
                print(f"root={root} executed all arms", flush=True)
    finally:
        env.close()
    wins = sum(r["arms"]["robust"]["ever"] and not r["arms"]["direct"]["ever"] for r in rows)
    losses = sum(r["arms"]["direct"]["ever"] and not r["arms"]["robust"]["ever"] for r in rows)
    counts = {a: sum(r["arms"][a]["ever"] for r in rows) for a in ("direct", "robust", "shuffled")}
    output = {"rows": rows, "success": counts, "wins": wins, "losses": losses,
              "development_exit": wins >= 2 and losses <= 1 and counts["robust"] > counts["shuffled"],
              "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL),
              "scope": "exposed fixed pool; open loop; not independent confirmation"}
    atomic_write_json(OUT, output)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
