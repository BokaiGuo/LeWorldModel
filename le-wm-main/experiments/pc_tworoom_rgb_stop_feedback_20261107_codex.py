"""RGB-only tail decision with stop-on-Gym-done environment evaluation."""
import hashlib
import json
import time

import numpy as np
import torch

from experiments import pc_tworoom_live_prefix_feedback_20261107_codex as live
from experiments import pc_tworoom_stop_aware_feedback_20261107_codex as stopped
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUTPUT = live.ROOT / "results/pc_tworoom_rgb_stop_feedback_20261107_codex.json"
PROTOCOL = live.ROOT.parent / "docs/PC_TWOROOM_RGB_STOP_FEEDBACK_PROTOCOL_20261107_CODEX.md"


def select_tail_rgb(pixels, goal, tails, *, transform, scaler, boundary, device):
    """The decision interface has no environment, info, state, or success input."""
    pixels, goal = np.asarray(pixels), np.asarray(goal)
    tails = np.asarray(tails, dtype=np.float32)
    if (pixels.ndim != 3 or goal.shape != pixels.shape or pixels.dtype != np.uint8
            or goal.dtype != np.uint8):
        raise ValueError("current and goal must be matching uint8 RGB images")
    if tails.shape != (2, 15, 2) or not np.isfinite(tails).all() or np.abs(tails).max() > 1:
        raise ValueError("two finite legal 15-step tails required")
    ctx = {"pixels": transform(pixels)[None, None, None].to(device),
           "goal": transform(goal)[None, None, None].to(device),
           "action": torch.empty((1, 1, 0, 10), device=device)}
    normal = scaler.transform(tails.reshape(-1, 2)).reshape(1, 2, 3, 10)
    scores, _ = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device=device))
    costs = scores[0].detach().cpu().tolist()
    return live.previous.choose(costs), costs


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    previous = live.previous
    cp = json.loads(previous.cem.PLAN.read_text())
    old = json.loads(previous.base.PLAN.read_text())
    reference = json.loads(stopped.OUTPUT.read_text())
    if (reference["runner_sha256"] != file_hash(stopped.__file__)
            or reference["protocol_sha256"] != file_hash(stopped.PROTOCOL)
            or reference["historical_sha256"] != file_hash(live.RESULT)):
        raise ValueError("reference drift")
    historical = json.loads(live.RESULT.read_text())
    if historical["public"] != file_hash(live.PANEL.PUBLIC) or historical["truth"] != file_hash(live.PANEL.TRUTH):
        raise ValueError("panel drift")
    if historical["cem_plan"] != file_hash(previous.cem.PLAN) or historical["prior_plan"] != file_hash(previous.base.PLAN):
        raise ValueError("plan drift")
    model, scaler, transform, normalizer, boundary = previous.base.wide.setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != reference["model_sha256"] or normalizer != reference["normalizer_sha256"]:
        raise ValueError("model drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(live.PANEL.PUBLIC, allow_pickle=False) as public, np.load(live.PANEL.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, (p, src) in enumerate(zip(cp["rows"], old["rows"], strict=True)):
                for arm in ("selected", "reversed"):
                    ref = reference["rows"][len(rows)]
                    if (p["seed"], arm) != (ref["seed"], ref["arm"]) or p["seed"] != src["seed"] or p["seed"] != int(public["seeds"][i]):
                        raise ValueError("root mismatch")
                    actions = np.asarray(p["arms"][arm]["actions"], np.float32)
                    tails = live.candidate_tails(actions, previous.aligned.reconstruct(src, arm))
                    env.reset(seed=p["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    np.testing.assert_array_equal(env.render(), public["current"][i])
                    trace, dispatched = [], []
                    done = stopped.execute_until_done(env, actions[:10], trace, dispatched)
                    selected, costs, elapsed, rgb_hash = None, None, 0.0, None
                    if not done:
                        current = env.render().copy()
                        rgb_hash = hashlib.sha256(current.tobytes()).hexdigest()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        selected, costs = select_tail_rgb(current, public["goal"][i], tails,
                                                          transform=transform, scaler=scaler,
                                                          boundary=boundary, device="cuda")
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        stopped.execute_until_done(env, tails[selected], trace, dispatched)
                    if ref["initial_success"]:
                        raise ValueError("different t0 policy requires separate analysis")
                    if selected != ref["selected"] or rgb_hash != ref["observed_rgb_sha256"]:
                        raise ValueError("decision or observation differs")
                    np.testing.assert_allclose(np.asarray(dispatched), np.asarray(ref["actions"]), rtol=0, atol=0)
                    np.testing.assert_allclose(np.asarray(trace), np.asarray(ref["trace"]), rtol=0, atol=1e-5)
                    rows.append({"seed": p["seed"], "arm": arm, "actions": dispatched, "trace": trace,
                                 "selected": selected, "scores": costs, "observed_rgb_sha256": rgb_hash,
                                 "success_after_action": any(x[1] for x in trace),
                                 "truncated": any(x[2] for x in trace),
                                 "query_chains": 2 if selected is not None else 0,
                                 "model_packets": 6 if selected is not None else 0,
                                 "feedback_score_seconds": elapsed})
                print(f"root={i}: RGB-only interface and terminated execution verified", flush=True)
    finally:
        env.close()
    if len(rows) != 80 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel/model drift")
    result = {"track": "primary_rgb", "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "reference_sha256": file_hash(stopped.OUTPUT),
              "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows,
              "scope": "exposed fixed candidates; not independent or equal-total-cost"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    run()
