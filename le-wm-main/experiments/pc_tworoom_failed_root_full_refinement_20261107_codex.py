"""Exhaustive physical replay of already-scored refinements at the four failed roots."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments import pc_tworoom_retention_counterfactual_coverage_20261107_codex as previous

ROOTS = (62735, 82490, 86996, 168250)
PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_FAILED_ROOT_FULL_REFINEMENT_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_tworoom_failed_root_full_refinement_20261107_codex.json"


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    coverage = json.loads(previous.OUTPUT.read_text())
    if (coverage["plan_sha256"] != file_hash(source.PLAN)
            or tuple(r["row"] for r in coverage["rows"] if not r["lookahead_success"]) != ROOTS):
        raise ValueError("failed panel or provenance changed")
    with h5py.File(source.common.DATA, "r") as dataset:
        raw = np.asarray(dataset["action"][:])
    raw = raw[~np.isnan(raw).any(axis=1)]
    if hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest() != plan["normalizer_action_sha256"]:
        raise ValueError("normalizer drift")
    scaler = StandardScaler().fit(raw)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    output = []
    try:
        with h5py.File(source.common.DATA, "r") as dataset:
            for row_id in ROOTS:
                row = next(r for r in plan["rows"] if r["row"] == row_id)
                old = next(r for r in coverage["rows"] if r["row"] == row_id)
                candidates = []
                for prefix, refinement in enumerate(row["refinements"]):
                    for index, (candidate, cost) in enumerate(zip(refinement["actions"], refinement["costs"], strict=True)):
                        chain = source.to_physical(np.asarray(candidate), scaler).astype(np.float32)
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(dataset["proprio"][row_id])
                        actor._set_goal_state(dataset["proprio"][row_id + 25])
                        trace = []
                        for action in chain:
                            _, _, term, trunc, info = actor.step(action)
                            trace.append({"distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                        if index == old["representatives"][prefix]["sample_index"]:
                            if trace != old["representatives"][prefix]["trace"]:
                                raise ValueError("representative trajectory differs")
                        candidates.append({"prefix": prefix, "sample": index, "model_cost": cost,
                                           "action_sha256": hashlib.sha256(chain.tobytes()).hexdigest(),
                                           "ever_success": any(item["terminated"] for item in trace),
                                           "trace": trace})
                if len(candidates) != 288:
                    raise ValueError("refinement candidate count changed")
                successes = [item for item in candidates if item["ever_success"]]
                output.append({"row": row_id, "selected": row["selected"],
                               "lookahead_cost": row["outcomes"]["lookahead"]["cost"],
                               "descriptor_cost": row["outcomes"]["descriptor"]["cost"],
                               "success_count": len(successes), "candidates": candidates})
                print(f"row={row_id}: {len(candidates)} chains, {len(successes)} successes", flush=True)
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": output, "plan_sha256": file_hash(source.PLAN),
                   "coverage_sha256": file_hash(previous.OUTPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)),
                   "scope": "post-hoc 4 roots; all scored refinements only; physical evaluation"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    run()
