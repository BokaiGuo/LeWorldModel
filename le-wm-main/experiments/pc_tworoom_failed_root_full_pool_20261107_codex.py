"""Physically replay every remaining scored proposal for four exposed failed roots."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments import pc_tworoom_failed_root_full_refinement_20261107_codex as refinement

PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_FAILED_ROOT_FULL_POOL_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_tworoom_failed_root_full_pool_20261107_codex.json"


def candidates_for(row):
    generation = np.asarray(row["generation_actions"])
    costs = np.asarray(row["generation_costs"])
    pool = np.asarray(row["pool"])
    tails = np.asarray(row["a_tails"])
    matrix = np.asarray(row["a_cost"])
    if generation.shape != (300, 25, 2) or costs.shape != (300,) or pool.shape != (8, 25, 2):
        raise ValueError("generation dimensions changed")
    if tails.shape != (8, 15, 2) or matrix.shape != (8, 8):
        raise ValueError("tail dimensions changed")
    chains = [(f"cem_{i}", generation[i], float(costs[i])) for i in range(300)]
    chains.extend((f"global_{i}", pool[4 + i], float(row["parent_cost"][4 + i])) for i in range(4))
    for prefix in range(8):
        for tail in range(8):
            chains.append((f"cross_{prefix}_{tail}",
                           np.concatenate((pool[prefix, :10], tails[tail]), axis=0),
                           float(matrix[prefix, tail])))
    return chains


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    old = json.loads(source.PHYSICAL.read_text())
    full = json.loads(refinement.OUTPUT.read_text())
    if (old["plan_sha256"] != file_hash(source.PLAN) or full["plan_sha256"] != file_hash(source.PLAN)
            or tuple(row["row"] for row in full["rows"]) != refinement.ROOTS):
        raise ValueError("source or root mismatch")
    with h5py.File(source.common.DATA, "r") as data:
        action = np.asarray(data["action"][:])
    action = action[~np.isnan(action).any(axis=1)]
    if hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest() != plan["normalizer_action_sha256"]:
        raise ValueError("normalizer mismatch")
    scaler = StandardScaler().fit(action)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(source.common.DATA, "r") as data:
            for row_id in refinement.ROOTS:
                row = next(item for item in plan["rows"] if item["row"] == row_id)
                original = next(item for item in old["rows"] if item["row"] == row_id)
                refinements = next(item for item in full["rows"] if item["row"] == row_id)
                results, matching = [], 0
                for identity, normalized, cost in candidates_for(row):
                    physical = source.to_physical(normalized, scaler).astype(np.float32)
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row_id])
                    actor._set_goal_state(data["proprio"][row_id + 25])
                    trace = []
                    for action in physical:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    for arm in source.ARMS:
                        if np.array_equal(physical, np.asarray(row["outcomes"][arm]["actions"], dtype=np.float32)):
                            if trace != original["outcomes"][arm]["trace"]:
                                raise ValueError("same action trace mismatch")
                            matching += 1
                    results.append({"id": identity, "model_cost": cost,
                                    "action_sha256": hashlib.sha256(physical.tobytes()).hexdigest(),
                                    "ever_success": any(x["terminated"] for x in trace), "trace": trace})
                if len(results) != 368 or len(refinements["candidates"]) != 288:
                    raise ValueError("scored pool count mismatch")
                success = sum(item["ever_success"] for item in results)
                rows.append({"row": row_id, "lookahead_cost": row["outcomes"]["lookahead"]["cost"],
                             "descriptor_cost": row["outcomes"]["descriptor"]["cost"],
                             "retained": row["selected"], "additional_pool_successes": success,
                             "refinement_successes": refinements["success_count"],
                             "matching_original_traces": matching, "candidates": results})
                print(f"row={row_id}: added {len(results)} chains, {success} successes; "
                      f"refinements={refinements['success_count']}", flush=True)
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(source.PLAN),
                   "physical_sha256": file_hash(source.PHYSICAL), "refinements_sha256": file_hash(refinement.OUTPUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "exposed 4 roots, complete 656 scored chains, evaluation-only"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    run()
