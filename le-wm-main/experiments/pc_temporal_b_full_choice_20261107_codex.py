"""Equal-candidate temporal choice on exposed tasks, sealed before replay."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_temporal_b_tail_validation_20261107_codex as bdev
from experiments import pc_temporal_effect_retention_20261107_codex as old
from experiments.pc_causal_prefix_scores_20261107_codex import select
from experiments.pc_input_permissions import file_hash

BASE = old.ROOT / "results/pc_temporal_b_full_choice_20261107_codex"
POOL, PLAN, PHYS, RESULT = [Path(str(BASE) + x) for x in
                            ("_pool.npz", "_plan.json", "_physical.json", "_result.json")]
PROTOCOL = old.ROOT.parent / "docs/PC_TEMPORAL_B_FULL_CHOICE_PROTOCOL_20261107_CODEX.md"
METHODS = ("endpoint", "all_time", "post_prefix")


def choose(actions, scores, method):
    if method not in METHODS:
        raise ValueError("unknown method")
    if np.asarray(scores).shape != (130, 5):
        raise ValueError("expected full temporal scores")
    # Prefix consistency always covers the full clock before selecting a window.
    from experiments.pc_causal_prefix_scores_20261107_codex import canonicalize
    fixed, audit = canonicalize(actions, scores, packet_steps=5, max_disagreement=0.005)
    selected = {"endpoint": fixed[:, 4:], "all_time": fixed, "post_prefix": fixed[:, 2:]}[method]
    per_chain = selected.min(axis=1)
    minimum = per_chain.min()
    matches = np.flatnonzero(per_chain == minimum)
    index = min(matches, key=lambda i: np.ascontiguousarray(actions[i]).tobytes())
    return int(index), float(minimum), audit


def seal():
    if POOL.exists() or PLAN.exists():
        raise FileExistsError("sealed output exists")
    bdev.verify()
    previous = json.loads(old.PLAN.read_text())
    if any(previous["sha256"][key] != file_hash(path) for key, path in
           (("pool", old.POOL), ("old_pool", old.old.POOL), ("initial_pool", old.old.base.source.POOL),
            ("temporal_scores", old.temporal.SCORES), ("inputs", old.old.base.source.block.prior.inputs.INPUT))):
        raise ValueError("source drift")
    paths = {"old_plan": old.PLAN, "old_pool": old.POOL, "candidate_pool": old.old.POOL,
             "initial_pool": old.old.base.source.POOL, "a_scores": old.temporal.SCORES,
             "b_seal": bdev.SEAL, "b_scores": bdev.SCORES, "protocol": PROTOCOL, "runner": __file__,
             "selector": Path(select.__code__.co_filename)}
    rows, arrays = [], {}
    with (np.load(old.old.POOL, allow_pickle=False) as candidate,
          np.load(old.old.base.source.POOL, allow_pickle=False) as initial,
          np.load(old.temporal.SCORES, allow_pickle=False) as a_scores,
          np.load(bdev.SCORES, allow_pickle=False) as b_scores,
          np.load(old.POOL, allow_pickle=False) as saved):
        for task in previous["rows"]:
            root = task["root"]
            a, b = candidate[f"r{root}_a"], candidate[f"r{root}_b"]
            bdev.validate_bank(a, b)
            inc = np.asarray([initial[f"r{root}_{name}_actions"][
                int(np.argmin(initial[f"r{root}_{name}_scores"]))] for name in ("block_cem", "block_random")])
            actions = np.concatenate((inc, a.reshape(64, 25, 2), b.reshape(64, 25, 2)))
            scores = np.concatenate((saved[f"r{root}_inc_times"],
                                     a_scores[f"r{root}_scores"][:, 1:].reshape(64, 5),
                                     b_scores[f"r{root}_scores"].reshape(64, 5)))
            decisions, chosen = {}, []
            for method in METHODS:
                index, value, audit = choose(actions, scores, method)
                picked = actions[index]
                decisions[method] = {"index": index, "cost": value, "action_sha256": old.digest(picked),
                                     "source": "incumbent" if index < 2 else "a" if index < 66 else "b",
                                     "prefix_max_spread": audit["max_spread"]}
                chosen.append(picked)
            arrays[f"r{root}_actions"] = np.asarray(chosen, dtype=np.float32)
            rows.append({"root": root, "row": task["row"], "decisions": decisions,
                         "logical_full_chains_per_arm": 600 + 130})
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    paths["pool"] = POOL
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "sha256": {k: file_hash(v) for k, v in paths.items()},
                   "track": "primary_rgb", "scope": "exposed post-hoc development"}, f, indent=2)


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYS.exists():
        raise FileExistsError(PHYS)
    plan = json.loads(PLAN.read_text())
    paths = {"pool": POOL, "old_plan": old.PLAN, "old_pool": old.POOL, "candidate_pool": old.old.POOL,
             "initial_pool": old.old.base.source.POOL, "a_scores": old.temporal.SCORES,
             "b_seal": bdev.SEAL, "b_scores": bdev.SCORES, "protocol": PROTOCOL, "runner": __file__,
             "selector": Path(select.__code__.co_filename)}
    if any(plan["sha256"][k] != file_hash(v) for k, v in paths.items()):
        raise ValueError("sealed input drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(POOL, allow_pickle=False) as pool, h5py.File(old.old.base.source.block.prior.common.DATA, "r") as data:
            for task in plan["rows"]:
                outcomes = {}
                for i, method in enumerate(METHODS):
                    action = pool[f"r{task['root']}_actions"][i]
                    if old.digest(action) != task["decisions"][method]["action_sha256"]:
                        raise ValueError("chosen action drift")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    trace = []
                    for a in action:
                        _, _, term, trunc, info = actor.step(a.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[method] = {"trace": trace, "ever": any(x["terminated"] for x in trace),
                                        "final": trace[-1]["terminated"]}
                rows.append({"root": task["root"], "outcomes": outcomes})
                print(f"root={task['root']} executed", flush=True)
    finally:
        env.close()
    with PHYS.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL)}, f)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    p, r = json.loads(PLAN.read_text()), json.loads(PHYS.read_text())
    if r["plan_sha256"] != file_hash(PLAN) or r["pool_sha256"] != file_hash(POOL):
        raise ValueError("replay binding drift")
    if [x["root"] for x in r["rows"]] != list(range(20, 50)):
        raise ValueError("incomplete roots")
    summary = {}
    for m in METHODS:
        values = [x["outcomes"][m] for x in r["rows"]]
        summary[m] = {"ever": sum(x["ever"] for x in values), "final": sum(x["final"] for x in values),
                      "mean_final_distance": float(np.mean([x["trace"][-1]["distance"] for x in values])),
                      "source_counts": {name: sum(x["decisions"][m]["source"] == name for x in p["rows"])
                                        for name in ("incumbent", "a", "b")},
                      "wins_vs_endpoint": sum(x["outcomes"][m]["ever"] and
                                              not x["outcomes"]["endpoint"]["ever"] for x in r["rows"]),
                      "losses_vs_endpoint": sum(not x["outcomes"][m]["ever"] and
                                                x["outcomes"]["endpoint"]["ever"] for x in r["rows"])}
    a, e = summary["all_time"], summary["endpoint"]
    passed = a["wins_vs_endpoint"] >= 2 and a["losses_vs_endpoint"] <= 1 and a["final"] >= e["final"] - 1
    with RESULT.open("x") as f:
        json.dump({"summary": summary, "development_exit": bool(passed),
                   "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYS),
                   "scope": "exposed fixed candidate first-plan"}, f, indent=2, allow_nan=False)
    print(json.dumps({"summary": summary, "development_exit": bool(passed)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "replay", "summarize"):
        raise SystemExit("expected seal|replay|summarize")
    {"seal": seal, "replay": replay, "summarize": summarize}[sys.argv[1]]()
