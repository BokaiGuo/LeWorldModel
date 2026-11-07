"""Separate stop-on-termination replay of the exposed RGB feedback panel."""
import hashlib
import json
import time

import numpy as np
import torch

from experiments import pc_tworoom_live_prefix_feedback_20261107_codex as live
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUTPUT = live.ROOT / "results/pc_tworoom_stop_aware_feedback_20261107_codex.json"
PROTOCOL = live.ROOT.parent / "docs/PC_TWOROOM_STOP_AWARE_FEEDBACK_PROTOCOL_20261107_CODEX.md"


def execute_until_done(env, actions, trace, dispatched):
    """Execute only until the first Gym termination or truncation."""
    for action in actions:
        _, _, terminated, truncated, info = env.step(action.copy())
        trace.append([float(info["distance_to_target"]), bool(terminated), bool(truncated)])
        dispatched.append(action.tolist())
        if terminated or truncated:
            return True
    return False


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = live.previous
    cp = json.loads(prior.cem.PLAN.read_text())
    old = json.loads(prior.base.PLAN.read_text())
    historical = json.loads(live.RESULT.read_text())
    if historical["runner"] != file_hash(live.__file__) or historical["protocol"] != file_hash(live.PROTOCOL):
        raise ValueError("historical source drift")
    if historical["cem_plan"] != file_hash(prior.cem.PLAN) or historical["prior_plan"] != file_hash(prior.base.PLAN):
        raise ValueError("plan drift")
    if historical["public"] != file_hash(live.PANEL.PUBLIC) or historical["truth"] != file_hash(live.PANEL.TRUTH):
        raise ValueError("panel drift")
    model, scaler, transform, normalizer, boundary = prior.base.wide.setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != historical["model"] or normalizer != historical["normalizer"]:
        raise ValueError("model drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(live.PANEL.PUBLIC, allow_pickle=False) as public, np.load(live.PANEL.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, (p, src) in enumerate(zip(cp["rows"], old["rows"], strict=True)):
                for arm in ("selected", "reversed"):
                    past = historical["rows"][len(rows)]
                    if (p["seed"], arm) != (past["seed"], past["arm"]) or p["seed"] != src["seed"] or p["seed"] != int(public["seeds"][i]):
                        raise ValueError("root mismatch")
                    original = np.asarray(p["arms"][arm]["actions"], np.float32)
                    tails = live.candidate_tails(original, prior.aligned.reconstruct(src, arm))
                    _, info = env.reset(seed=p["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    np.testing.assert_array_equal(env.render(), public["current"][i])
                    trace, dispatched = [], []
                    initial_success = float(info["distance_to_target"]) < 16.0
                    done = initial_success or execute_until_done(env, original[:10], trace, dispatched)
                    costs, chosen, elapsed, rgb_hash = None, None, 0.0, None
                    if not done:
                        observed = env.render().copy()
                        rgb_hash = hashlib.sha256(observed.tobytes()).hexdigest()
                        ctx = {"pixels": transform(observed)[None, None, None].cuda(),
                               "goal": transform(public["goal"][i])[None, None, None].cuda(),
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        normal = scaler.transform(tails.reshape(-1, 2)).reshape(1, 2, 3, 10)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        scores, _ = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        costs = scores[0].cpu().tolist()
                        chosen = prior.choose(costs)
                        execute_until_done(env, tails[chosen], trace, dispatched)
                    if chosen is not None:
                        if chosen != past["selected"] or rgb_hash != past["observed_rgb_sha256"]:
                            raise ValueError("feedback differs from historical decision")
                    np.testing.assert_allclose(np.asarray(dispatched).reshape(-1, 2), np.asarray(past["actions"][:len(dispatched)]).reshape(-1, 2), rtol=0, atol=0)
                    np.testing.assert_allclose(np.asarray(trace).reshape(-1, 3), np.asarray(past["trace"][:len(trace)]).reshape(-1, 3), rtol=0, atol=1e-5)
                    rows.append({"seed": p["seed"], "arm": arm, "initial_success": initial_success,
                                 "actions": dispatched, "trace": trace, "scores": costs, "selected": chosen,
                                 "observed_rgb_sha256": rgb_hash, "feedback_score_seconds": elapsed,
                                 "query_chains": 2 if chosen is not None else 0,
                                 "model_packets": 6 if chosen is not None else 0,
                                 "success": initial_success or any(x[1] for x in trace),
                                 "truncated": any(x[2] for x in trace)})
                print(f"root {i}: stop-aware replay complete", flush=True)
    finally:
        env.close()
    if len(rows) != 80 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel or model drift")
    result = {"track": "primary_rgb", "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "historical_sha256": file_hash(live.RESULT),
              "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    run()
