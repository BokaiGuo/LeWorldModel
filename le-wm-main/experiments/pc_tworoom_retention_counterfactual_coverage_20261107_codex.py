"""Evaluation-only physical replay of every already-scored prefix representative."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source

PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_RETENTION_COUNTERFACTUAL_COVERAGE_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_tworoom_retention_counterfactual_coverage_20261107_codex.json"


def analyze_root(plan, original, representatives):
    if len(representatives) != 8 or any(r["prefix"] != i for i, r in enumerate(representatives)):
        raise ValueError("prefix identities must cover all eight")
    selected = original["outcomes"]["lookahead"]
    success = [r for r in representatives if r["ever_success"]]
    failures = []
    if not selected["ever_success"]:
        for rep in success:
            failures.append({"prefix": rep["prefix"],
                             "lookahead_kept": rep["prefix"] in plan["selected"]["lookahead"],
                             "descriptor_kept": rep["prefix"] in plan["selected"]["descriptor"],
                             "lower_model_cost_than_lookahead": rep["model_cost"] < plan["outcomes"]["lookahead"]["cost"],
                             "lower_model_cost_than_descriptor": rep["model_cost"] < plan["outcomes"]["descriptor"]["cost"]})
    return {"row": plan["row"], "lookahead_success": selected["ever_success"],
            "descriptor_success": original["outcomes"]["descriptor"]["ever_success"],
            "success_prefixes": [r["prefix"] for r in success],
            "oracle_any": bool(selected["ever_success"] or success),
            "additional_success_cases": failures, "representatives": representatives}


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    physical = json.loads(source.PHYSICAL.read_text())
    if (physical["plan_sha256"] != file_hash(source.PLAN)
            or plan["runner_sha256"] != file_hash(source.__file__)
            or plan["input_sha256"] != file_hash(source.inputs.INPUT)):
        raise ValueError("source artifacts changed")
    with h5py.File(source.common.DATA, "r") as dataset:
        raw = np.asarray(dataset["action"][:])
    raw = raw[~np.isnan(raw).any(axis=1)]
    if hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest() != plan["normalizer_action_sha256"]:
        raise ValueError("normalizer source mismatch")
    scaler = StandardScaler().fit(raw)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, matched_trace_count = [], 0
    try:
        with h5py.File(source.common.DATA, "r") as dataset:
            for root, original in zip(plan["rows"], physical["rows"], strict=True):
                if root["row"] != original["row"]:
                    raise ValueError("root mismatch")
                reps = []
                for prefix, refinement in enumerate(root["refinements"]):
                    j = int(np.argmin(refinement["costs"]))
                    normalized = np.asarray(refinement["actions"][j])
                    chain = source.to_physical(normalized, scaler).astype(np.float32)
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(dataset["proprio"][root["row"]])
                    actor._set_goal_state(dataset["proprio"][root["row"] + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action)
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    for arm in source.ARMS:
                        previous = np.asarray(root["outcomes"][arm]["actions"], dtype=np.float32)
                        if np.array_equal(chain, previous):
                            if trace != original["outcomes"][arm]["trace"]:
                                raise ValueError("same chain has different physical trace")
                            matched_trace_count += 1
                    reps.append({"prefix": prefix, "sample_index": j, "model_cost": refinement["costs"][j],
                                 "action_sha256": hashlib.sha256(chain.tobytes()).hexdigest(),
                                 "ever_success": any(step["terminated"] for step in trace),
                                 "final_distance": trace[-1]["distance"], "trace": trace})
                rows.append(analyze_root(root, original, reps))
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "matched_original_trace_count": matched_trace_count,
                   "plan_sha256": file_hash(source.PLAN), "physical_sha256": file_hash(source.PHYSICAL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "post-hoc exposed panel, 8 model-best refined representatives, evaluation-only"},
                  stream, indent=2, allow_nan=False)
    print(f"{len(rows)} tasks; lookahead {sum(r['lookahead_success'] for r in rows)}, "
          f"union oracle {sum(r['oracle_any'] for r in rows)}, matched traces {matched_trace_count}")


if __name__ == "__main__":
    run()
