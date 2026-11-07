"""Execute a fixed-tail RGB feedback decision in one continuous environment."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments import pc_tworoom_cem_prefix_feedback_20261107_codex as previous
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT, PANEL = previous.ROOT, previous.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_LIVE_PREFIX_FEEDBACK_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_tworoom_live_prefix_feedback_20261107_codex"
RESULT = Path(str(BASE) + "_run.json")
IMAGES = Path(str(BASE) + "_observations.npz")
COMPARISON = Path(str(BASE) + "_comparison.json")


def candidate_tails(optimized, old_chain):
    optimized, old_chain = np.asarray(optimized, np.float32), np.asarray(old_chain, np.float32)
    if optimized.shape != (25, 2) or old_chain.shape != (25, 2):
        raise ValueError("wrong chain shape")
    if not np.isfinite(optimized).all() or not np.isfinite(old_chain).all():
        raise ValueError("nonfinite action")
    if np.abs(optimized).max() > 1 or np.abs(old_chain).max() > 1:
        raise ValueError("out-of-bounds action")
    np.testing.assert_array_equal(optimized[:10], old_chain[:10])
    return np.stack((optimized[10:], old_chain[10:]))


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if RESULT.exists() or IMAGES.exists() or COMPARISON.exists():
        raise FileExistsError("new result already exists")
    cp = json.loads(previous.cem.PLAN.read_text())
    old = json.loads(previous.base.PLAN.read_text())
    checks = ((cp, "prior_plan_sha256", previous.base.PLAN), (cp, "public_sha256", PANEL.PUBLIC),
              (cp, "protocol_sha256", previous.cem.PROTOCOL), (cp, "runner_sha256", previous.cem.__file__),
              (old, "public_sha256", PANEL.PUBLIC), (old, "protocol_sha256", previous.base.PROTOCOL),
              (old, "runner_sha256", previous.base.__file__), (old, "wide_sha256", previous.base.wide.__file__),
              (old, "repair_sha256", previous.base.wide.repair.__file__))
    for obj, name, path in checks:
        if obj[name] != file_hash(path):
            raise ValueError(f"source drift: {name}")
    if json.loads(PANEL.REFERENCE.read_text())["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("truth drift")
    model, scaler, transform, normalizer, boundary = previous.base.wide.setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != cp["model_sha256"] or normalizer != cp["normalizer_sha256"]:
        raise ValueError("model drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, images = [], []
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, np.load(PANEL.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, (p, src) in enumerate(zip(cp["rows"], old["rows"], strict=True)):
                if p["seed"] != src["seed"] or p["seed"] != int(public["seeds"][i]):
                    raise ValueError("root mismatch")
                for arm in ("selected", "reversed"):
                    original = np.asarray(p["arms"][arm]["actions"], np.float32)
                    tails = candidate_tails(original, previous.aligned.reconstruct(src, arm))
                    env.reset(seed=p["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    trace, dispatched = [], []
                    def step(action):
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append([float(info["distance_to_target"]), bool(term), bool(trunc)])
                        dispatched.append(action.tolist())
                    for action in original[:10]:
                        step(action)
                    observed = actor.render().copy()
                    images.append(observed)
                    ctx = {"pixels": transform(observed)[None, None, None].cuda(),
                           "goal": transform(public["goal"][i])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    normal = scaler.transform(tails.reshape(-1, 2)).reshape(1, 2, 3, 10)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    scores, _ = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    costs = scores[0].cpu().tolist()
                    chosen = previous.choose(costs)
                    for action in tails[chosen]:
                        step(action)
                    if len(trace) != 25 or len(dispatched) != 25:
                        raise ValueError("action clock mismatch")
                    rows.append({"seed": p["seed"], "arm": arm, "scores": costs, "selected": chosen,
                                 "observed_rgb_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                                 "actions": dispatched, "trace": trace, "ever_success": any(x[1] for x in trace),
                                 "final_success": trace[-1][1], "query_chains": 2, "model_packets": 6,
                                 "feedback_score_seconds": elapsed})
                print(f"root {i}: both live feedback arms executed", flush=True)
    finally:
        env.close()
    if len(rows) != 80 or model_fingerprint(model) != fingerprint:
        raise ValueError("model/panel drift")
    with IMAGES.open("xb") as stream:
        np.savez_compressed(stream, rgb=np.stack(images))
    with RESULT.open("x") as stream:
        json.dump({"track": "primary_rgb", "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "cem_plan": file_hash(previous.cem.PLAN), "prior_plan": file_hash(previous.base.PLAN),
                   "public": file_hash(PANEL.PUBLIC), "truth": file_hash(PANEL.TRUTH),
                   "model": fingerprint, "normalizer": normalizer, "images": file_hash(IMAGES),
                   "rows": rows, "scope": "exposed fixed candidates; extra feedback cost; not independent"},
                  stream, allow_nan=False)


def compare():
    if COMPARISON.exists():
        raise FileExistsError(COMPARISON)
    live = json.loads(RESULT.read_text())
    prior = json.loads(previous.PLAN.read_text())
    old = json.loads(previous.aligned.OUTPUT.read_text())
    if live["protocol"] != file_hash(PROTOCOL) or live["runner"] != file_hash(__file__) or live["images"] != file_hash(IMAGES):
        raise ValueError("live source drift")
    if prior["cem_plan_sha256"] != file_hash(previous.cem.PLAN) or old["plan_sha256"] != file_hash(previous.cem.PLAN):
        raise ValueError("prior source drift")
    rows = []
    with np.load(IMAGES, allow_pickle=False) as images:
        if images["rgb"].shape[0] != 80:
            raise ValueError("image count mismatch")
        for i, (current, decision, historical) in enumerate(zip(live["rows"], prior["rows"], old["rows"], strict=True)):
            if (current["seed"], current["arm"]) != (decision["seed"], decision["arm"]) or (current["seed"], current["arm"]) != (historical["seed"], historical["arm"]):
                raise ValueError("identity mismatch")
            if hashlib.sha256(images["rgb"][i].tobytes()).hexdigest() != current["observed_rgb_sha256"]:
                raise ValueError("image mismatch")
            reference_trace = historical["traces"][current["selected"]]
            previous.aligned.validate_trace(current["trace"], reference_trace)
            rows.append({"seed": current["seed"], "arm": current["arm"],
                         "selection_matches_prior": current["selected"] == decision["selected"],
                         "max_score_difference": float(np.max(np.abs(np.asarray(current["scores"]) - decision["scores"]))),
                         "trace_matches_selected_prior": True,
                         "baseline_success": historical["success"][0], "live_success": current["ever_success"]})
    summary = {arm: {"roots": sum(x["arm"] == arm for x in rows),
                     "baseline_successes": sum(x["baseline_success"] for x in rows if x["arm"] == arm),
                     "live_successes": sum(x["live_success"] for x in rows if x["arm"] == arm)}
               for arm in ("selected", "reversed")}
    with COMPARISON.open("x") as stream:
        json.dump({"run": file_hash(RESULT), "prior_decisions": file_hash(previous.PLAN),
                   "prior_physical": file_hash(previous.aligned.OUTPUT), "rows": rows, "summary": summary,
                   "selection_matches": sum(x["selection_matches_prior"] for x in rows),
                   "max_score_difference": max(x["max_score_difference"] for x in rows)}, stream, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "compare"):
        raise SystemExit("expected run|compare")
    {"run": run, "compare": compare}[sys.argv[1]]()
