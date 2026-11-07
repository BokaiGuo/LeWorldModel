"""Physical replay of every previously queried local-tail candidate."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_conditional_local_tail_20261107_codex as source
from experiments.pc_input_permissions import file_hash

PREFIX = source.ROOT / "results/pc_conditional_local_tail_full_pool_20261107_codex"
PHYSICAL, RESULT = Path(str(PREFIX) + "_physical.json"), Path(str(PREFIX) + "_result.json")
PROTOCOL = source.ROOT.parent / "docs/PC_CONDITIONAL_LOCAL_TAIL_FULL_POOL_PROTOCOL_20261107_CODEX.md"


def validate_ids(ids, keep):
    ids, keep = np.asarray(ids), np.asarray(keep)
    if (ids.ndim != 1 or keep.shape != (2,) or len(set(ids.tolist())) != len(ids)
            or np.any(np.diff(ids) <= 0) or len(set(keep.tolist())) != 2
            or np.any(keep < 0) or np.any(keep > 7)):
        raise ValueError("invalid ids")
    wanted = np.sort((keep[:, None] * 8 + np.arange(8)).ravel())
    indexes = np.searchsorted(ids, wanted)
    if np.any(indexes >= len(ids)) or not np.array_equal(ids[indexes], wanted):
        raise ValueError("missing retained candidates")
    return indexes


def replay():
    import gymnasium as gym
    import h5py
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(source.PLAN.read_text())
    original = json.loads(source.PHYSICAL.read_text())
    if (plan["pool_sha256"] != file_hash(source.POOL)
            or original["plan_sha256"] != file_hash(source.PLAN)
            or original["pool_sha256"] != file_hash(source.POOL)
            or plan["runner_sha256"] != file_hash(source.__file__)
            or plan["protocol_sha256"] != file_hash(source.PROTOCOL)):
        raise ValueError("source binding mismatch")
    if [r["root"] for r in plan["rows"]] != list(range(10, 50)) or \
       [r["root"] for r in original["rows"]] != list(range(10, 50)):
        raise ValueError("incomplete panel")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, matched = [], 0
    try:
        with (np.load(source.POOL, allow_pickle=False) as pool,
              h5py.File(source.source.old.base.source.block.prior.common.DATA, "r") as data):
            for row, old in zip(plan["rows"], original["rows"], strict=True):
                root, outcomes = row["root"], []
                ids = pool[f"r{root}_ids"]
                actions = pool[f"r{root}_local"].reshape(64, 25, 2)
                if actions.shape != (64, 25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                    raise ValueError("illegal candidate")
                for method in source.METHODS:
                    validate_ids(ids, row["decisions"][method]["retained"])
                for index in ids:
                    chain = actions[index]
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    for method in source.METHODS:
                        decision = row["decisions"][method]
                        if decision["bank"] == "local" and decision["index"] == int(index):
                            if trace != old["outcomes"][method]["trace"]:
                                raise ValueError("selected trajectory mismatch")
                            matched += 1
                    outcomes.append({"index": int(index), "action_sha256": source.source.digest(chain),
                                     "trace": trace})
                rows.append({"root": root, "row": row["row"], "outcomes": outcomes})
                print(f"root={root} replayed={len(outcomes)}", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "matched_selected_trajectories": matched,
                   "source_plan_sha256": file_hash(source.PLAN), "source_pool_sha256": file_hash(source.POOL),
                   "source_physical_sha256": file_hash(source.PHYSICAL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__))},
                  f, separators=(",", ":"), allow_nan=False)


def outcome_summary(ids, scores, outcomes, keep, selected_success):
    pos = validate_ids(ids, keep)
    if len(scores) != len(ids) or len(outcomes) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("invalid outcome length")
    if any(int(ids[i]) != o["index"] or len(o["trace"]) != 25 for i, o in enumerate(outcomes)):
        raise ValueError("outcome identity mismatch")
    success, tail = [], []
    for i in pos:
        trace = outcomes[i]["trace"]
        before = any(x["terminated"] for x in trace[:10])
        success.append(any(x["terminated"] for x in trace))
        tail.append(not before and any(x["terminated"] for x in trace[10:]))
    scores_local = np.asarray(scores)[pos]
    order = np.argsort(scores_local, kind="stable")
    return {"pool_success": bool(any(success)), "pool_tail_success": bool(any(tail)),
            "best_local_success": bool(success[int(order[0])]),
            "selected_success": bool(selected_success),
            "best_success_rank": next((n + 1 for n, j in enumerate(order) if success[j]), None)}


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, physical, old = (json.loads(path.read_text()) for path in (source.PLAN, PHYSICAL, source.PHYSICAL))
    for key, path in (("source_plan_sha256", source.PLAN), ("source_pool_sha256", source.POOL),
                      ("source_physical_sha256", source.PHYSICAL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", Path(__file__))):
        if physical[key] != file_hash(path):
            raise ValueError("physical binding mismatch")
    if [r["root"] for r in physical["rows"]] != list(range(10, 50)):
        raise ValueError("incomplete panel")
    rows = []
    with np.load(source.POOL, allow_pickle=False) as pool:
        for p, q, original in zip(plan["rows"], physical["rows"], old["rows"], strict=True):
            root = p["root"]
            if root != q["root"] or root != original["root"]:
                raise ValueError("root mismatch")
            ids, scores = pool[f"r{root}_ids"], pool[f"r{root}_scores"]
            for record in q["outcomes"]:
                chain = pool[f"r{root}_local"].reshape(64, 25, 2)[record["index"]]
                if source.source.digest(chain) != record["action_sha256"]:
                    raise ValueError("action hash mismatch")
            methods = {m: outcome_summary(ids, scores, q["outcomes"], p["decisions"][m]["retained"],
                                           original["outcomes"][m]["ever"]) for m in source.METHODS}
            rows.append({"root": root, "methods": methods})
    summary = {m: {"local_pool_success": sum(r["methods"][m]["pool_success"] for r in rows),
                   "local_pool_tail_success": sum(r["methods"][m]["pool_tail_success"] for r in rows),
                   "best_local_success": sum(r["methods"][m]["best_local_success"] for r in rows),
                   "selected_success": sum(r["methods"][m]["selected_success"] for r in rows),
                   "failed_selection_pool_success_roots": [r["root"] for r in rows
                       if not r["methods"][m]["selected_success"] and r["methods"][m]["pool_success"]],
                   "failed_selection_pool_no_success_roots": [r["root"] for r in rows
                       if not r["methods"][m]["selected_success"] and not r["methods"][m]["pool_success"]]}
               for m in source.METHODS}
    with RESULT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "physical_sha256": file_hash(PHYSICAL),
                   "scope": "posthoc exposed development; oracle labels never used for selection"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("replay", "summarize"):
        raise SystemExit("expected replay|summarize")
    {"replay": replay, "summarize": summarize}[sys.argv[1]]()
