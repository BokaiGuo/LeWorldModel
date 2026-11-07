"""Evaluator-only physical replay after the public planner has sealed its output."""
import json

import h5py
import numpy as np

from experiments import pc_tworoom_goal_gradient_development_20261107_codex as legacy
from experiments import pc_rgb_gradient_public_plan_20261107_codex as planner
from experiments.pc_input_permissions import file_hash

OUT = legacy.ROOT / "results/pc_rgb_gradient_public_evaluation_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(planner.OUT.read_text())
    old = json.loads(legacy.PLAN.read_text())
    old_physical = json.loads(legacy.PHYSICAL.read_text())
    if (plan["public_sha256"] != file_hash(planner.PUBLIC)
            or plan["protocol_sha256"] != file_hash(planner.PROTOCOL)
            or plan["runner_sha256"] != file_hash(planner.__file__)
            or old["dependencies"] != legacy.dependencies()
            or old_physical["plan_sha256"] != file_hash(legacy.PLAN)):
        raise ValueError("sealed source mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    results = []
    try:
        with h5py.File(legacy.prior.source.common.DATA, "r") as db, np.load(planner.PUBLIC, allow_pickle=False) as public:
            planner.validate_public(public)
            for new, previous, previous_physical in zip(plan["rows"], old["rows"], old_physical["rows"], strict=True):
                if len({new["episode"], previous["episode"], previous_physical["episode"]}) != 1:
                    raise ValueError("episode mismatch")
                index, = np.flatnonzero(public["episodes"] == new["episode"])
                candidates = np.asarray(new["candidates"], np.float32)
                if candidates.shape != (5, 25, 2) or np.abs(candidates).max() > 1.000001:
                    raise ValueError("invalid action chains")
                traces = []
                for actions in candidates:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(db["proprio"][previous["row"]])
                    actor._set_goal_state(db["proprio"][previous["row"] + 25])
                    if not np.array_equal(actor.render(), public["current"][index]):
                        raise ValueError("public image is not evaluator reset")
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "success": bool(term), "truncated": bool(trunc)})
                        if trunc:
                            break
                    traces.append({"trace": trace, "ever_success": any(t["success"] for t in trace),
                                   "complete": len(trace) == 25})
                prior_candidates = np.asarray(previous["candidates"], np.float32)
                prior_scores = np.asarray(previous["scores"])
                scores = np.asarray(new["scores"])
                results.append({"episode": new["episode"], "candidates": traces,
                                "selected_index": new["selected_index"],
                                "candidate_exact": bool(np.array_equal(candidates, prior_candidates)),
                                "score_max_abs_error": float(np.max(np.abs(scores - prior_scores))),
                                "selected_equal": new["selected_index"] == previous["selected_index"],
                                "physical_trace_exact": traces == previous_physical["candidates"],
                                "initial_success": previous["initial_success"]})
                print(f"episode={new['episode']} replayed", flush=True)
    finally:
        env.close()
    with OUT.open("x") as stream:
        json.dump({"track": "primary_rgb", "plan_sha256": file_hash(planner.OUT),
                   "old_plan_sha256": file_hash(legacy.PLAN), "old_physical_sha256": file_hash(legacy.PHYSICAL),
                   "runner_sha256": file_hash(__file__), "rows": results,
                   "scope": "evaluation-only replay on exposed episodes"}, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
