"""Fixed A/B action-identity diagnostic, with physical truth evaluated separately."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_block_cem_50dev_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_BLOCK_HELDOUT_EFFECT_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_block_heldout_effect_20261107_codex"
POOL = Path(str(PREFIX) + "_pool.npz")
PLAN = Path(str(PREFIX) + "_plan.json")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
RESULT = Path(str(PREFIX) + "_result.json")
METHODS = ("parent", "row_mean", "additive", "identity", "shuffled")


def predictions(a, b, cost, parent, seed):
    """Predict held-out B costs using only A values and action identities."""
    a, b, cost, parent = map(np.asarray, (a, b, cost, parent))
    if a.shape != (8, 3, 2) or b.shape != (8, 3, 2) or cost.shape != (8, 8) or parent.shape != (8,):
        raise ValueError("unexpected A/B shapes")
    nearest = np.argsort(((b[:, None] - a[None]) ** 2).sum(axis=(2, 3)), axis=1, kind="stable")[:, :3]
    row_mean = cost.mean(axis=1)
    centered = cost - row_mean[:, None]
    rng = np.random.default_rng(seed)
    shuffled = np.stack([row[rng.permutation(8)] for row in centered])
    return {
        "parent": np.broadcast_to(parent[:, None], (8, 8)).copy(),
        "row_mean": np.broadcast_to(row_mean[:, None], (8, 8)).copy(),
        "additive": row_mean[:, None] + centered.mean(axis=0)[nearest].mean(axis=1)[None, :],
        "identity": row_mean[:, None] + centered[:, nearest].mean(axis=2),
        "shuffled": row_mean[:, None] + shuffled[:, nearest].mean(axis=2),
    }


def select_prefixes(cem, cem_scores, random):
    selected = []
    for bank, order in ((cem, np.argsort(cem_scores, kind="stable")),
                        (random, np.arange(len(random)))):
        count = 0
        for i in order:
            if all(not np.array_equal(bank[i, :10], prior[:10]) for prior in selected):
                selected.append(bank[i])
                count += 1
            if count == 4:
                break
        if count != 4:
            raise ValueError("not enough distinct prefixes")
    return np.asarray(selected)


def plan():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("sealed output exists")
    previous = json.loads(source.PLAN.read_text())
    if previous["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("source pool drift")
    model, scaler, transform, normalizer = source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("source model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.PLAN, source.POOL,
                                      source.block.prior.inputs.INPUT,
                                      source.block.prior.common.CHECKPOINT / "weights.pt",
                                      source.block.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="five packets of five environment steps")
    arrays, rows = {}, []
    with (np.load(source.POOL, allow_pickle=False) as old,
          np.load(source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in range(10):
            candidates = select_prefixes(old[f"r{root}_block_cem_actions"],
                                         old[f"r{root}_block_cem_scores"],
                                         old[f"r{root}_block_random_actions"])
            raw_a = np.random.default_rng(29111000 + root).uniform(-1, 1, (8, 3, 2)).astype(np.float32)
            raw_b = np.random.default_rng(29112000 + root).uniform(-1, 1, (8, 3, 2)).astype(np.float32)
            wrapper = RGBLeWM(model, audit)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}

            def score(raw):
                norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 25, 2)
                values = []
                for offset in range(0, len(raw), 32):
                    batch = torch.as_tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                            dtype=torch.float32, device="cuda")
                    value, _ = wrapper.score(context, batch)
                    values.extend(value[0].cpu().tolist())
                return np.asarray(values)

            parent = score(candidates)
            matrices = {}
            for label, tail in (("a", raw_a), ("b", raw_b)):
                expanded = np.repeat(tail, 5, axis=1)
                chains = np.concatenate((np.broadcast_to(candidates[:, None, :10], (8, 8, 10, 2)),
                                         np.broadcast_to(expanded[None], (8, 8, 15, 2))), axis=2)
                matrices[label] = score(chains.reshape(64, 25, 2)).reshape(8, 8)
                arrays[f"r{root}_{label}_actions"] = chains
            pred = predictions(raw_a, raw_b, matrices["a"], parent, 29113000 + root)
            choices = {method: int(np.argmin(pred[method])) for method in METHODS}
            arrays[f"r{root}_a"] = raw_a
            arrays[f"r{root}_b"] = raw_b
            arrays[f"r{root}_a_cost"] = matrices["a"]
            arrays[f"r{root}_b_cost"] = matrices["b"]
            arrays[f"r{root}_parent"] = parent
            for method in METHODS:
                arrays[f"r{root}_pred_{method}"] = pred[method]
            rows.append({"root": root, "row": int(inputs["rows"][root]), "choices": choices,
                         "model_queries": 136})
            print(f"root={root} model predictions sealed in memory", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": RGB, "rows": rows, "pool_sha256": file_hash(POOL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_plan_sha256": file_hash(source.PLAN), "source_pool_sha256": file_hash(source.POOL),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "input_sha256": file_hash(source.block.prior.inputs.INPUT),
                   "scope": "exposed development diagnostic"}, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan_record = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("input_sha256", source.block.prior.inputs.INPUT)):
        if plan_record[key] != file_hash(path):
            raise ValueError(f"binding drift: {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (h5py.File(source.block.prior.common.DATA, "r") as data,
              np.load(POOL, allow_pickle=False) as pool):
            for row in plan_record["rows"]:
                outcomes = []
                for chain in pool[f"r{row['root']}_b_actions"].reshape(64, 25, 2):
                    if not np.isfinite(chain).all() or np.max(np.abs(chain)) > 1:
                        raise ValueError("illegal chain")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes.append({"prefix_success": any(x["terminated"] for x in trace[:10]),
                                     "ever": any(x["terminated"] for x in trace),
                                     "final_distance": trace[-1]["distance"], "trace": trace})
                rows.append({"root": row["root"], "row": row["row"], "outcomes": outcomes})
                print(f"root={row['root']} evaluated 64 chains", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL)},
                  f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan_record, physical_record = json.loads(PLAN.read_text()), json.loads(PHYSICAL.read_text())
    if physical_record["plan_sha256"] != file_hash(PLAN) or physical_record["pool_sha256"] != file_hash(POOL):
        raise ValueError("physical binding drift")
    rows = []
    with np.load(POOL, allow_pickle=False) as pool:
        for plan_row, actual in zip(plan_record["rows"], physical_record["rows"], strict=True):
            root = plan_row["root"]
            if root != actual["root"] or len(actual["outcomes"]) != 64:
                raise ValueError("incomplete outcomes")
            target = pool[f"r{root}_b_cost"]
            outcomes = {}
            for method in METHODS:
                prediction = pool[f"r{root}_pred_{method}"]
                index = plan_row["choices"][method]
                if index != int(np.argmin(prediction)):
                    raise ValueError("selection changed")
                outcome = actual["outcomes"][index]
                outcomes[method] = {"mse": float(np.square(prediction - target).mean()),
                                    "choice": index, "distance": outcome["final_distance"],
                                    "ever": outcome["ever"], "prefix_success": outcome["prefix_success"]}
            rows.append({"root": root, "outcomes": outcomes,
                         "pool_ever": any(x["ever"] for x in actual["outcomes"])})
    counts = {}
    for method in METHODS:
        counts[method] = {
            "mean_mse": float(np.mean([r["outcomes"][method]["mse"] for r in rows])),
            "mean_final_distance": float(np.mean([r["outcomes"][method]["distance"] for r in rows])),
            "success": sum(r["outcomes"][method]["ever"] for r in rows),
            "prefix_success": sum(r["outcomes"][method]["prefix_success"] for r in rows),
            "mse_better_than_additive": sum(r["outcomes"][method]["mse"] < r["outcomes"]["additive"]["mse"] for r in rows),
            "distance_better_than_additive": sum(r["outcomes"][method]["distance"] < r["outcomes"]["additive"]["distance"] for r in rows)}
    eligible = lambda m: counts[m]["mse_better_than_additive"] >= 6 and counts[m]["distance_better_than_additive"] >= 6
    result = {"rows": rows, "summary": counts, "pool_success_roots": sum(r["pool_ever"] for r in rows),
              "development_exit": bool(eligible("identity") and not eligible("shuffled")),
              "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL),
              "scope": "held-out diagnostic; not deployable planning or independent confirmation"}
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical", "summarize"):
        raise SystemExit("expected plan|physical|summarize")
    {"plan": plan, "physical": physical, "summarize": summarize}[sys.argv[1]]()
