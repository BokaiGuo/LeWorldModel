"""Seal the RGB policy on exact dynamics before same-chain interventions."""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy
from experiments.pc_budget_allocation_20261103_codex import optimize
from experiments.pc_equal_resource_dev_20261103_codex import Meter

ROOT = policy.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SAME_POLICY_EXACT_MECHANISM_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_same_policy_exact_mechanism_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout(actions):
    actions = np.asarray(actions, dtype=float)
    if (actions.ndim != 3 or actions.shape[2] != 2 or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1 + 1e-12)):
        raise ValueError("invalid actions")
    state = np.zeros((len(actions), 3))
    for step in range(actions.shape[1]):
        state[:, 0] += .1 * actions[:, step, 0] * np.cos(state[:, 2])
        state[:, 1] += .1 * actions[:, step, 0] * np.sin(state[:, 2])
        state[:, 2] += .3 * actions[:, step, 1]
    return state


def predictor(goal, meter):
    def predict(actions):
        def raw(batch):
            positions = rollout(batch)[:, :2]
            return positions, np.linalg.norm(positions - goal, axis=1)
        return meter.predict(np.asarray(actions, dtype=float), raw)
    return predict


def dependencies():
    return {str(path): sha(path) for path in (
        PROTOCOL, Path(__file__), Path(policy.__file__), Path(policy.generator.__file__),
        Path(optimize.__code__.co_filename), Path(inspect.getfile(Meter)))}


def plan():
    if PLANS.exists() or INTERVENTION.exists():
        raise FileExistsError("sealed output already exists")
    paths = dependencies()
    rows = []
    for root in range(12):
        angle = 2 * np.pi * (root + .193) / 12
        goal = .9 * np.array([np.cos(angle), np.sin(angle)])
        methods = {}
        for arm in policy.ARMS:
            meter = Meter(policy.CAP)
            result = policy.decide(arm, root, 0, goal, predictor(goal, meter), meter)
            methods[arm] = {"decision": result, "counts": meter.as_dict()}
        svd = methods["svd"]["decision"]
        programs = svd["programs"]
        greedy_name = min(svd["prefix_scores"], key=lambda x: x["score"])["name"]
        greedy = next(item for item in programs if item["name"] == greedy_name)
        lookahead = min(programs, key=lambda x: x["terminal_score"])
        rows.append({"root": root, "goal": goal.tolist(), "methods": methods,
                     "same_bank_greedy_prefix": greedy["chain"][:5],
                     "cached_lookahead": lookahead})
    if paths != dependencies():
        raise ValueError("dependency changed during planning")
    with PLANS.open("x") as stream:
        json.dump({"protocol": "exact-model dev only", "dependencies": paths, "rows": rows,
                   "PC2_pass": False, "PC3_pass": False}, stream, indent=2, allow_nan=False)
    print("12 task plans sealed before interventions", flush=True)


def intervene():
    if INTERVENTION.exists():
        raise FileExistsError(INTERVENTION)
    source = json.loads(PLANS.read_text())
    if source["dependencies"] != dependencies():
        raise ValueError("sealed dependency mismatch")
    rows = []
    for row in source["rows"]:
        goal = np.asarray(row["goal"])
        chain = np.asarray(row["methods"]["svd"]["decision"]["selected"]["chain"])
        controls = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(row["same_bank_greedy_prefix"])}
        suffixes = [np.tile(x, (10, 1)) for x in (
            (0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        outcomes = {}
        for name, prefix in controls.items():
            start = float(np.linalg.norm(rollout(prefix[None])[0, :2] - goal))
            chains = np.stack([np.concatenate((prefix, suffix)) for suffix in suffixes])
            common = np.linalg.norm(rollout(chains)[:, :2] - goal, axis=1)
            _, optimized, scores = optimize(prefix, 15, 120, np.random.default_rng(2026120200 + row["root"]),
                                            lambda x: np.linalg.norm(rollout(x)[:, :2] - goal, axis=1))
            outcomes[name] = {"prefix": prefix.tolist(), "prefix_distance": start,
                              "common_suffix_distances": common.tolist(),
                              "active_changes": (common - common[0]).tolist(),
                              "own_optimized_distance": optimized, "own_scores": scores}
        chosen = outcomes["selected"]
        nonprogress = chosen["prefix_distance"] >= np.linalg.norm(goal) - 1e-10
        active = all(chosen["active_changes"][-1] < outcomes[x]["active_changes"][-1] - 1e-10
                     for x in ("zero", "greedy"))
        own = all(chosen["own_optimized_distance"] < outcomes[x]["own_optimized_distance"] - 1e-10
                  for x in ("zero", "greedy"))
        rows.append({"root": row["root"], "chain": chain.tolist(), "controls": outcomes,
                     "nonprogress": bool(nonprogress), "active_response": bool(active),
                     "own_plan_advantage": bool(own), "limited_witness": bool(nonprogress and active and own)})
    summary = {key: sum(row[key] for row in rows) for key in (
        "nonprogress", "active_response", "own_plan_advantage", "limited_witness")}
    with INTERVENTION.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "summary": summary, "rows": rows,
                   "PC2_pass": False, "PC3_pass": False}, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "intervene"):
        raise SystemExit("choose plan or intervene")
    (plan if sys.argv[1] == "plan" else intervene)()
