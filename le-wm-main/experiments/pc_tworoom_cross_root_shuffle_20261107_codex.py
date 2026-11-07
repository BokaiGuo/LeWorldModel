"""Post-hoc cross-root descriptor negative control, with sealed actions."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as original
from experiments.pc_input_permissions import file_hash

PROTOCOL = original.ROOT.parent / "docs/PC_TWOROOM_CROSS_ROOT_DESCRIPTOR_SHUFFLE_AUDIT_20261107_CODEX.md"
PLAN = original.ROOT / "results/pc_tworoom_cross_root_shuffle_plan_20261107_codex.json"
PHYSICAL = original.ROOT / "results/pc_tworoom_cross_root_shuffle_physical_20261107_codex.json"
RESULT = original.ROOT / "results/pc_tworoom_cross_root_shuffle_result_20261107_codex.json"


def rotate_sources(n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("at least two roots required")
    return list(range(1, n)) + [0]


def choose(costs, vectors):
    costs, vectors = np.asarray(costs), np.asarray(vectors)
    if costs.shape != (8, 8) or vectors.shape != (8, 8) or not np.isfinite(costs).all() or not np.isfinite(vectors).all():
        raise ValueError("finite 8 by 8 costs and vectors required")
    look = costs.min(axis=1)
    first = int(np.argmin(look))
    eligible = np.argsort(look, kind="stable")[:6]
    eligible = eligible[eligible != first]
    distances = np.linalg.norm(vectors[eligible] - vectors[first], axis=1)
    return [first, int(eligible[np.argmax(distances)])]


def action_scaler():
    with h5py.File(original.common.DATA, "r") as dataset:
        actions = np.asarray(dataset["action"][:])
    actions = actions[~np.isnan(actions).any(axis=1)]
    return StandardScaler().fit(actions), hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


def prepare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    prior = json.loads(original.PLAN.read_text())
    if (prior["protocol_sha256"] != file_hash(original.PROTOCOL)
            or prior["runner_sha256"] != file_hash(original.__file__)
            or prior["input_sha256"] != file_hash(original.inputs.INPUT)):
        raise ValueError("source plan drift")
    scaler, action_hash = action_scaler()
    if action_hash != prior["normalizer_action_sha256"]:
        raise ValueError("normalizer drift")
    source_rows = prior["rows"]
    if len(source_rows) != 10 or [r["root"] for r in source_rows] != list(range(10)):
        raise ValueError("root panel mismatch")
    mapping = rotate_sources(len(source_rows))
    rows = []
    for index, row in enumerate(source_rows):
        matrix = np.asarray(row["a_cost"])
        donor = np.asarray(source_rows[mapping[index]]["a_cost"])
        centered = donor - donor.mean(axis=1, keepdims=True)
        selected = choose(matrix, centered)
        own = choose(matrix, matrix - matrix.mean(axis=1, keepdims=True))
        if own != row["selected"]["descriptor"]:
            raise ValueError("original selection mismatch")
        pool = np.asarray(row["pool"])
        cross = np.concatenate((np.broadcast_to(pool[:, None, :10], (8, 8, 10, 2)),
                                np.broadcast_to(np.asarray(row["a_tails"])[None], (8, 8, 15, 2))), axis=2)
        choices = np.concatenate((np.asarray(row["generation_actions"]), pool[4:], cross.reshape(64, 25, 2)))
        values = np.r_[row["generation_costs"], row["parent_cost"][4:], matrix.ravel()]
        incumbent = int(np.argmin(values))
        if incumbent != row["incumbent_index"] or values[incumbent] != row["incumbent_cost"]:
            raise ValueError("incumbent mismatch")
        available = [(float(values[incumbent]), "incumbent", choices[incumbent])]
        for p in selected:
            refinement = row["refinements"][p]
            chain = np.asarray(refinement["actions"])
            if chain.shape != (36, 25, 2) or not np.array_equal(chain[:, :10], np.broadcast_to(pool[p, :10], (36, 10, 2))):
                raise ValueError("fixed-prefix violation")
            j = int(np.argmin(refinement["costs"]))
            available.append((float(refinement["costs"][j]), f"prefix_{p}", chain[j]))
        cost, origin, chosen = min(available, key=lambda x: x[0])
        actions = original.to_physical(chosen, scaler)
        rows.append({"root": row["root"], "row": row["row"], "donor_root": source_rows[mapping[index]]["root"],
                     "selected": selected, "descriptor_selected": own, "origin": origin, "cost": cost,
                     "actions": actions.tolist(), "scored_chains_counterfactual": 440})
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "source_sha256": file_hash(original.PLAN), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "normalizer_sha256": action_hash,
                   "scope": "post-hoc offline cross-root attribution control; not a deployable online method"},
                  stream, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    if (plan["source_sha256"] != file_hash(original.PLAN) or plan["protocol_sha256"] != file_hash(PROTOCOL)
            or plan["runner_sha256"] != file_hash(Path(__file__))):
        raise ValueError("plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(original.common.DATA, "r") as dataset:
            for row in plan["rows"]:
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(dataset["proprio"][row["row"]])
                actor._set_goal_state(dataset["proprio"][row["row"] + 25])
                actions = np.asarray(row["actions"], dtype=np.float32)
                if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                    raise ValueError("invalid physical action")
                trace = []
                for action in actions:
                    _, _, terminated, truncated, info = actor.step(action.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(terminated), "truncated": bool(truncated)})
                rows.append({"root": row["root"], "row": row["row"], "trace": trace,
                             "ever_success": any(item["terminated"] for item in trace),
                             "final_success": trace[-1]["terminated"]})
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN)}, stream, indent=2, allow_nan=False)


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, new, old, prior = (json.loads(p.read_text()) for p in (PLAN, PHYSICAL, original.PHYSICAL, original.PLAN))
    if (new["plan_sha256"] != file_hash(PLAN) or old["plan_sha256"] != file_hash(original.PLAN)
            or plan["source_sha256"] != file_hash(original.PLAN)
            or plan["protocol_sha256"] != file_hash(PROTOCOL)
            or plan["runner_sha256"] != file_hash(Path(__file__))):
        raise ValueError("artifact drift")
    rows = []
    for p, t, baseline, prior_row in zip(plan["rows"], new["rows"], old["rows"], prior["rows"], strict=True):
        if len({p["row"], t["row"], baseline["row"], prior_row["row"]}) != 1:
            raise ValueError("row identity mismatch")
        comparisons = {}
        for arm in ("descriptor", "lookahead", "shuffled"):
            ref = baseline["outcomes"][arm]
            same = np.array_equal(np.asarray(p["actions"], dtype=np.float32),
                                  np.asarray(prior_row["outcomes"][arm]["actions"], dtype=np.float32))
            if same and t["trace"] != ref["trace"]:
                raise ValueError("same action physical mismatch")
            comparisons[arm] = {"same_actions": same, "ever_success": ref["ever_success"],
                                "final_success": ref["trace"][-1]["terminated"]}
        rows.append({"root": p["root"], "row": p["row"], "donor_root": p["donor_root"],
                     "selected": p["selected"], "descriptor_selected": p["descriptor_selected"],
                     "ever_success": t["ever_success"], "final_success": t["final_success"],
                     "comparisons": comparisons})
    summary = {"roots": len(rows), "ever": sum(r["ever_success"] for r in rows),
               "final": sum(r["final_success"] for r in rows),
               "different_prefix_sets": sum(r["selected"] != r["descriptor_selected"] for r in rows)}
    for arm in ("descriptor", "lookahead", "shuffled"):
        summary[arm] = {"same_actions": sum(r["comparisons"][arm]["same_actions"] for r in rows),
                        "control_only": sum(r["ever_success"] and not r["comparisons"][arm]["ever_success"] for r in rows),
                        "baseline_only": sum(not r["ever_success"] and r["comparisons"][arm]["ever_success"] for r in rows)}
    with RESULT.open("x") as stream:
        json.dump({"summary": summary, "rows": rows, "plan_sha256": file_hash(PLAN),
                   "physical_sha256": file_hash(PHYSICAL), "original_physical_sha256": file_hash(original.PHYSICAL),
                   "scope": "post-hoc control on exposed roots; no gate passage"}, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "physical", "analyze"):
        raise SystemExit("expected prepare|physical|analyze")
    {"prepare": prepare, "physical": physical, "analyze": analyze}[sys.argv[1]]()
