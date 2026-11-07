"""Evaluator-side physical replay and endpoint RGB capture."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_rgb_gradient_public_evaluate_20261107_codex as previous
from experiments.pc_input_permissions import file_hash

ROOT = previous.legacy.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_GRADIENT_ENDPOINT_DECOMPOSITION_PROTOCOL_20261107_CODEX.md"
FRAMES = ROOT / "results/pc_rgb_gradient_endpoint_frames_20261107_codex.npz"
OUT = ROOT / "results/pc_rgb_gradient_endpoint_frame_trace_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if FRAMES.exists() or OUT.exists():
        raise FileExistsError("endpoint artifacts already exist")
    plan = json.loads(previous.planner.OUT.read_text())
    legacy = json.loads(previous.legacy.PLAN.read_text())
    evaluation = json.loads(previous.OUT.read_text())
    if (evaluation["plan_sha256"] != file_hash(previous.planner.OUT)
            or evaluation["old_plan_sha256"] != file_hash(previous.legacy.PLAN)
            or plan["public_sha256"] != file_hash(previous.planner.PUBLIC)):
        raise ValueError("sealed source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    frames, rows = {}, []
    print("Capture physical endpoint images for sealed candidate actions.", flush=True)
    try:
        with h5py.File(previous.legacy.prior.source.common.DATA, "r") as db, np.load(previous.planner.PUBLIC, allow_pickle=False) as public:
            previous.planner.validate_public(public)
            for pred, old, reference in zip(plan["rows"], legacy["rows"], evaluation["rows"], strict=True):
                if len({pred["episode"], old["episode"], reference["episode"]}) != 1:
                    raise ValueError("episode identity mismatch")
                idx, = np.flatnonzero(public["episodes"] == pred["episode"])
                final_images, traces = [], []
                for candidate in pred["candidates"]:
                    actions = np.asarray(candidate, np.float32)
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1.000001:
                        raise ValueError("invalid action witness")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(db["proprio"][old["row"]])
                    actor._set_goal_state(db["proprio"][old["row"] + 25])
                    if not np.array_equal(actor.render(), public["current"][idx]):
                        raise ValueError("initial RGB drift")
                    steps = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        steps.append({"distance": float(info["distance_to_target"]),
                                      "success": bool(term), "truncated": bool(trunc)})
                        if trunc:
                            raise ValueError("unexpected truncation before declared endpoint")
                    traces.append({"trace": steps, "ever_success": any(x["success"] for x in steps),
                                   "complete": len(steps) == 25})
                    final_images.append(actor.render().copy())
                if traces != reference["candidates"]:
                    raise ValueError("replayed physical trace mismatch")
                key = f"e{pred['episode']}"
                frames[key] = np.asarray(final_images, np.uint8)
                rows.append({"episode": pred["episode"], "key": key,
                             "distance": [x["trace"][-1]["distance"] for x in traces],
                             "ever_success": [x["ever_success"] for x in traces],
                             "initial_success": reference["initial_success"]})
                print(f"episode={pred['episode']} frames=5", flush=True)
    finally:
        env.close()
    with FRAMES.open("xb") as stream:
        np.savez_compressed(stream, **frames)
    with OUT.open("x") as stream:
        json.dump({"plan_sha256": file_hash(previous.planner.OUT), "evaluation_sha256": file_hash(previous.OUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "frames_sha256": file_hash(FRAMES), "rows": rows}, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
