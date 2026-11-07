"""Physically replay the complete scored pools of selected failure roots."""
import json

import h5py
import numpy as np

from experiments import pc_rgb_conditional_temporal_cem_20261107_codex as src
from experiments.pc_input_permissions import file_hash

PROTOCOL = src.old.ROOT.parent / "docs/PC_CONDITIONAL_FAILURE_POOL_PROTOCOL_20261107_CODEX.md"
OUTPUT = src.old.ROOT / "results/pc_conditional_failure_pool_20261107_codex.json"


def summarize(costs, trajectories, selected):
    costs = np.asarray(costs, dtype=np.float64)
    if (costs.ndim != 1 or not np.isfinite(costs).all() or len(costs) != len(trajectories)
            or not 0 <= selected < len(costs)):
        raise ValueError("invalid pool")
    ever = np.asarray([any(x["terminated"] for x in trace) for trace in trajectories])
    final = np.asarray([trace[-1]["terminated"] for trace in trajectories])
    order = np.argsort(costs, kind="stable")
    ranks = np.flatnonzero(ever[order])
    best = int(order[ranks[0]]) if len(ranks) else None
    if selected != int(order[0]):
        raise ValueError("saved selected chain is not model score minimum")
    return {"candidate_count": len(costs), "ever_success_count": int(ever.sum()),
            "final_success_count": int(final.sum()), "selected": int(selected),
            "selected_ever": bool(ever[selected]), "selected_final": bool(final[selected]),
            "coverage_but_selection_failed": bool(ever.any() and not ever[selected]),
            "best_success_score_rank_one_based": int(ranks[0] + 1) if len(ranks) else None,
            "best_success_index": best, "best_success_score": float(costs[best]) if best is not None else None,
            "selected_score": float(costs[selected])}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(src.PLAN.read_text())
    physical = json.loads(src.PHYS.read_text())
    source = src.old.base.source
    inputs = source.block.prior.inputs.INPUT
    paths = {"pool": src.POOL, "old_plan": src.old.PLAN, "source_pool": source.POOL,
             "inputs": inputs, "protocol": src.PROTOCOL, "runner": src.__file__}
    if (any(plan["sha256"][k] != file_hash(v) for k, v in paths.items())
            or physical["plan_sha256"] != file_hash(src.PLAN)
            or physical["pool_sha256"] != file_hash(src.POOL)):
        raise ValueError("artifact drift")
    if [r["root"] for r in physical["rows"]] != list(range(20, 50)):
        raise ValueError("incomplete physical panel")
    truth_by_root = {r["root"]: r for r in physical["rows"]}
    failed_roots = [r["root"] for r in physical["rows"]
                    if any(not r["outcomes"][m]["ever"] for m in src.METHODS)]
    if failed_roots != [38, 39]:
        raise ValueError("failure panel changed")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(src.POOL, allow_pickle=False) as pool,
              h5py.File(source.block.prior.common.DATA, "r") as data):
            for task in plan["rows"]:
                root = task["root"]
                if root not in failed_roots:
                    continue
                arms = {}
                for method_i, method in enumerate(src.METHODS):
                    actions = np.concatenate((pool[f"r{root}_inc"], pool[f"r{root}_actions"][method_i]))
                    costs = np.concatenate((pool[f"r{root}_inc_scores"][:, -1],
                                            pool[f"r{root}_scores"][method_i, :, -1]))
                    selected = task["decisions"][method]["index"]
                    if (actions.shape != (194, 25, 2) or not np.isfinite(actions).all()
                            or np.any(np.abs(actions) > 1)
                            or src.digest(actions[selected]) != task["decisions"][method]["action_sha256"]):
                        raise ValueError("action pool mismatch")
                    traces = []
                    for index, chain in enumerate(actions):
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(data["proprio"][task["row"]])
                        actor._set_goal_state(data["proprio"][task["row"] + 25])
                        trace = []
                        for action in chain:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append({"distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                        if index == selected:
                            old = truth_by_root[root]["outcomes"][method]["trace"]
                            if len(old) != len(trace) or any(
                                    abs(a["distance"] - b["distance"]) > 1e-5
                                    or a["terminated"] != b["terminated"]
                                    or a["truncated"] != b["truncated"]
                                    for a, b in zip(trace, old, strict=True)):
                                raise ValueError("selected physical replay mismatch")
                        traces.append(trace)
                    arms[method] = {"summary": summarize(costs, traces, selected), "trajectories": traces}
                    print(f"root={root} arm={method} replayed={len(traces)}", flush=True)
                rows.append({"root": root, "row": task["row"], "arms": arms})
    finally:
        env.close()
    with OUTPUT.open("x") as file:
        json.dump({"rows": rows, "plan_sha256": file_hash(src.PLAN), "pool_sha256": file_hash(src.POOL),
                   "physical_sha256": file_hash(src.PHYS), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__),
                   "scope": "post-hoc failure roots; complete arm-specific 194-candidate pools"},
                  file, separators=(",", ":"), allow_nan=False)
    print(json.dumps({r["root"]: {m: r["arms"][m]["summary"] for m in src.METHODS} for r in rows}, indent=2))


if __name__ == "__main__":
    main()
