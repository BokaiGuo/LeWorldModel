"""Automatically compose model-generated programs on the exposed exact wall tasks."""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy
from experiments.pc_budget_allocation_20261103_codex import allocate_samples, optimize
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_wall_response_side_20261103 as wall
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter as StructureMeter

PROTOCOL = policy.ROOT.parent / "docs/PC_SUCCESSOR_RESPONSE_COMPOSITION_E0_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = policy.ROOT / "results/pc_successor_response_composition_e0_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")
ARMS = ("successor_response", "successor_nominal", "successor_random",
        "no_composition", "same_bank_greedy", "full_cem")
BUDGETS = (960, 1920)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    files = (PROTOCOL, Path(__file__), Path(policy.__file__), Path(policy.generator.__file__),
             Path(wall.__file__), Path(wall.physics.__file__), Path(optimize.__code__.co_filename),
             Path(inspect.getfile(Meter)), Path(inspect.getfile(StructureMeter)))
    return {str(path): sha(path) for path in files}


def endpoints(task, actions):
    actions = np.asarray(actions, dtype=float)
    if (actions.ndim != 3 or actions.shape[2] != 2 or actions.shape[1] not in (5, 10, 15)
            or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1 + 1e-12)):
        raise ValueError("invalid normalized action chain")
    padded = np.pad(actions, ((0, 0), (0, 16 - actions.shape[1]), (0, 0))) * wall.HIGH
    trajectory, _ = wall.rollout(task["start"], padded)
    return trajectory[:, actions.shape[1], :2]


def plan_one(task, index, budget, arm):
    started = time.monotonic()
    meter = Meter(budget)
    structure = StructureMeter({"prediction_samples": 168, "svd": 8, "linear_solve": 8})
    trace, phase = [], ["unassigned"]
    goal = np.asarray(task["goal"])

    def predict(chains):
        chains = np.asarray(chains, dtype=float)
        def raw(batch):
            positions = endpoints(task, batch)
            return positions, np.linalg.norm(positions - goal, axis=1)
        position, score = meter.predict(chains, raw)
        trace.append({"phase": phase[0], "actions": chains.tolist(), "positions": position.tolist(),
                      "scores": score.tolist(), "tokens": len(chains) * chains.shape[1] // 5})
        return position, score

    seed = 2026120300 + index
    bank, first, first_evidence, second_evidence, first_scores, programs = [], [], [], [], [], []
    if arm == "full_cem":
        phase[0] = "full_cem"
        count = budget // 3
        chain, score, _ = optimize([], 15, count, np.random.default_rng(seed), lambda x: predict(x)[1])
        selected = {"name": "full_cem", "chain": chain.tolist(), "score": score}
        incumbent = None
    else:
        phase[0] = "first_response"
        first, first_evidence = policy.generator.generate(
            lambda x: predict(np.asarray(x)[None])[0][0], goal, policy.NOMINALS, structure, mode="svd")
        phase[0] = "first_score"
        for item in first:
            first_scores.append(float(predict(np.asarray(item["prefix"])[None])[1][0]))
        if arm == "same_bank_greedy":
            chosen = int(np.argmin(first_scores))
            bank = [{"name": first[chosen]["name"], "prefix": first[chosen]["prefix"]}]
        elif arm == "no_composition":
            bank = [{"name": item["name"], "prefix": item["prefix"]} for item in first]
        elif arm in ("successor_response", "successor_nominal", "successor_random"):
            for i, item in enumerate(first):
                prefix = np.asarray(item["prefix"])
                if arm == "successor_response":
                    phase[0] = f"successor_response:{i}"
                    second, evidence = policy.generator.generate(
                        lambda x: predict(np.concatenate((prefix, np.asarray(x)))[None])[0][0],
                        goal, policy.NOMINALS, structure, mode="svd")
                    second_evidence.append(evidence)
                else:
                    second = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist()}]
                    rng = np.random.default_rng(seed + 100 + i)
                    for j, nominal in enumerate(policy.NOMINALS):
                        action = nominal.copy()
                        if arm == "successor_random":
                            direction = rng.normal(size=action.shape)
                            action = np.clip(action + .5 * direction / np.max(np.abs(direction)), -1, 1)
                        second.append({"name": f"{arm}_{j}", "prefix": action.tolist()})
                for item2 in second:
                    bank.append({"name": f'{item["name"]}/{item2["name"]}',
                                 "prefix": np.concatenate((prefix, item2["prefix"])).tolist()})
        else:
            raise ValueError("unknown method")
        affordable = (budget - meter.token_samples) // 3
        suffix_total = 3 * affordable // 4
        counts = allocate_samples(suffix_total, 0, len(bank), 1)
        if min(counts) < 1:
            raise ValueError("insufficient suffix budget")
        for i, (item, count) in enumerate(zip(bank, counts, strict=True)):
            phase[0] = f"suffix:{i}"
            chain, score, _ = optimize(item["prefix"], 15, count, np.random.default_rng(seed),
                                       lambda x: predict(x)[1])
            programs.append({"name": item["name"], "chain": chain.tolist(), "score": score, "samples": count})
        incumbent = min(programs, key=lambda x: x["score"])
        phase[0] = "joint_refine"
        count = (budget - meter.token_samples) // 3
        chain, score = policy.refine(incumbent["chain"], incumbent["score"], count,
                                     np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        selected = {"name": "joint_refined" if score < incumbent["score"] else incumbent["name"],
                    "chain": chain.tolist(), "score": score}
    return {"task": task, "root": index, "budget": budget, "arm": arm, "first_bank": first,
            "first_scores": first_scores, "first_response": first_evidence, "successor_response": second_evidence,
            "bank": bank, "programs": programs, "incumbent": incumbent, "selected": selected,
            "counts": meter.as_dict(), "structure_counts": structure.counts, "structure_seconds": structure.seconds,
            "wall_seconds": time.monotonic() - started, "trace": trace}


def plan():
    if any(path.exists() for path in (PLANS, AUDIT, INTERVENTION)):
        raise FileExistsError("output already exists")
    sources = hashes()
    rows = []
    for index, task in enumerate(wall.tasks()):
        for budget in BUDGETS:
            for arm in ARMS:
                rows.append(plan_one(task, index, budget, arm))
        print(f"root {index}: six arms at both budgets", flush=True)
    if sources != hashes():
        raise ValueError("dependency drift")
    with PLANS.open("x") as file:
        json.dump({"dependencies": sources, "rows": rows, "PC2_pass": False, "PC3_pass": False},
                  file, separators=(",", ":"), allow_nan=False)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    source = json.loads(PLANS.read_text())
    if source["dependencies"] != hashes():
        raise ValueError("dependency mismatch")
    rows = []
    for row in source["rows"]:
        tokens, selected_found = 0, False
        selected = np.asarray(row["selected"]["chain"])
        goal = np.asarray(row["task"]["goal"])
        for batch in row["trace"]:
            actions = np.asarray(batch["actions"])
            positions = endpoints(row["task"], actions)
            scores = np.linalg.norm(positions - goal, axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
            if actions.shape[1:] == selected.shape:
                match = np.flatnonzero(np.all(actions == selected, axis=(1, 2)))
                selected_found |= any(abs(scores[i] - row["selected"]["score"]) < 1e-12 for i in match)
            cost = len(actions) * actions.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("token mismatch")
            tokens += cost
        if not selected_found or tokens != row["counts"]["prediction_token_samples"] or tokens > row["budget"]:
            raise ValueError("selection or budget mismatch")
        if row["incumbent"] and row["selected"]["score"] > row["incumbent"]["score"] + 1e-12:
            raise ValueError("incumbent lost")
        rows.append({"root": row["root"], "budget": row["budget"], "arm": row["arm"], "tokens": tokens})
    with AUDIT.open("x") as file:
        json.dump({"plans_sha256": sha(PLANS), "rows": rows}, file, indent=2, allow_nan=False)
    print(f"audited {len(rows)} fully traced plans", flush=True)


def intervene():
    if INTERVENTION.exists():
        raise FileExistsError(INTERVENTION)
    source, audited = json.loads(PLANS.read_text()), json.loads(AUDIT.read_text())
    if source["dependencies"] != hashes() or audited["plans_sha256"] != sha(PLANS):
        raise ValueError("sealed dependency mismatch")
    rows = []
    for row in source["rows"]:
        if row["arm"] != "successor_response":
            continue
        chain = np.asarray(row["selected"]["chain"])
        goal = np.asarray(row["task"]["goal"])
        idx = int(np.argmin(row["first_scores"]))
        controls = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(row["first_bank"][idx]["prefix"])}
        suffixes = [np.tile(x, (10, 1)) for x in
                    ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        results = {}
        for name, prefix in controls.items():
            prefix_distance = float(np.linalg.norm(endpoints(row["task"], prefix[None])[0] - goal))
            full = np.stack([np.concatenate((prefix, suffix)) for suffix in suffixes])
            common = np.linalg.norm(endpoints(row["task"], full) - goal, axis=1)
            best, score, scores = optimize(
                prefix, 15, 120, np.random.default_rng(2026120400 + row["root"]),
                lambda actions: np.linalg.norm(endpoints(row["task"], actions) - goal, axis=1))
            results[name] = {"prefix": prefix.tolist(), "prefix_distance": prefix_distance,
                             "common_suffix_distances": common.tolist(),
                             "active_changes": (common - common[0]).tolist(),
                             "own_optimized_distance": score, "own_best": best.tolist(), "own_scores": scores}
        initial = float(np.linalg.norm(np.asarray(row["task"]["start"])[:2] - goal))
        chosen = results["selected"]
        nonprogress = chosen["prefix_distance"] >= initial - 1e-10
        active = all(chosen["active_changes"][-1] < results[x]["active_changes"][-1] - 1e-10
                     for x in ("zero", "greedy"))
        own = all(chosen["own_optimized_distance"] < results[x]["own_optimized_distance"] - 1e-10
                  for x in ("zero", "greedy"))
        rows.append({"root": row["root"], "budget": row["budget"], "chain": chain.tolist(),
                     "controls": results, "nonprogress": bool(nonprogress), "active_response": bool(active),
                     "own_plan_advantage": bool(own), "limited_witness": bool(nonprogress and active and own)})
    with INTERVENTION.open("x") as file:
        json.dump({"plans_sha256": sha(PLANS), "audit_sha256": sha(AUDIT), "rows": rows,
                   "PC2_pass": False, "PC3_pass": False}, file, indent=2, allow_nan=False)
    print(f"intervened on {len(rows)} sealed selected chains", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit", "intervene"):
        raise SystemExit("choose plan, audit or intervene")
    {"plan": plan, "audit": audit, "intervene": intervene}[sys.argv[1]]()
