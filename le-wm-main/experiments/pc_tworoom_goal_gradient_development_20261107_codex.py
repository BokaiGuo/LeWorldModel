"""Sealed RGB goal-gradient candidate comparison and physical replay."""
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_precise_spectral_development_20261107_codex as prior
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_derivative_backend_20261107_codex import precise_action_derivatives
from experiments.pc_rgb_goal_gradient_candidates_20261107_codex import make_candidates
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest
from experiments.pc_tworoom_search_region_multihorizon_derivative_20261107_codex import normalize_actions

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_GOAL_GRADIENT_PROPOSAL_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_goal_gradient_development_plan_20261107_codex.json"
PHYSICAL = ROOT / "results/pc_tworoom_goal_gradient_development_physical_20261107_codex.json"


def dependencies():
    paths = [__file__, PROTOCOL, prior.PLAN, prior.PHYSICAL, prior.source.INPUT,
             prior.source.common.DATA, prior.source.common.CHECKPOINT / "weights.pt",
             prior.source.common.CHECKPOINT / "config.json",
             Path(make_candidates.__code__.co_filename),
             ROOT / "experiments/pc_rgb_derivative_backend_20261107_codex.py"]
    return {str(p): file_hash(p) for p in paths}


def plan():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PLAN.exists() or PHYSICAL.exists():
        raise FileExistsError("sealed outputs already exist")
    old = json.loads(prior.PLAN.read_text())
    physical_old = json.loads(prior.PHYSICAL.read_text())
    if old["dependencies"] != prior.dependencies() or physical_old["plan_sha256"] != file_hash(prior.PLAN):
        raise ValueError("spectral source drift")
    model, scaler, transform, normalizer = prior.source.common.model_setup()
    digest = model_digest(model)
    if digest != old["model_digest"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.PLAN, prior.source.INPUT,
                         prior.source.common.CHECKPOINT / "weights.pt",
                         prior.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="five env steps per packet")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Compare goal-gradient candidates to sealed spectral and random proposals.", flush=True)
    try:
        with np.load(prior.source.INPUT, allow_pickle=False) as inputs, h5py.File(prior.source.common.DATA, "r") as db:
            for item in old["rows"]:
                episode, row = item["episode"], item["row"]
                ids = np.flatnonzero(inputs["episodes"] == episode)
                if len(ids) != 1 or int(inputs["rows"][ids[0]]) != row:
                    raise ValueError("ambiguous task identity")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(db["proprio"][row])
                actor._set_goal_state(db["proprio"][row + 25])
                image = actor.render().copy()
                if (prior.img_hash(image) != item["current_rgb_sha256"]
                        or prior.img_hash(inputs["goal"][ids[0]]) != item["goal_rgb_sha256"]):
                    raise ValueError("image identity mismatch")
                context = {"pixels": transform(image)[None, None, None].cuda(),
                           "goal": transform(inputs["goal"][ids[0]])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                nominal = np.asarray(item["arms"]["spectral"]["candidates"][0], dtype=np.float32)
                if not np.array_equal(nominal, np.asarray(item["arms"]["random"]["candidates"][0], np.float32)):
                    raise ValueError("historical parents disagree")
                raw = torch.tensor(nominal, device="cuda", requires_grad=True)
                def score_raw(actions):
                    score, _ = guard.score(context, normalize_actions(actions, scaler.mean_, scaler.scale_))
                    return score[0, 0]

                torch.cuda.synchronize()
                started = time.perf_counter()
                with precise_action_derivatives():
                    value = score_raw(raw)
                    gradient, = torch.autograd.grad(value, raw)
                torch.cuda.synchronize()
                gradient_seconds = time.perf_counter() - started
                candidates, direction = make_candidates(nominal, gradient.detach().cpu().numpy())
                direct = float((gradient.detach() * torch.as_tensor(direction, device="cuda", dtype=raw.dtype)).sum())
                torch.cuda.synchronize()
                started = time.perf_counter()
                with precise_action_derivatives(), torch.no_grad():
                    d = torch.as_tensor(direction, device="cuda", dtype=raw.dtype)
                    plus = float(score_raw(raw.detach() + .001 * d))
                    minus = float(score_raw(raw.detach() - .001 * d))
                torch.cuda.synchronize()
                finite_difference_seconds = time.perf_counter() - started
                fd = (plus - minus) / .002
                error = abs(fd - direct)
                grad_ok = error <= .1 + .01 * abs(direct)
                normalized = scaler.transform(candidates.reshape(-1, 2)).astype(np.float32).reshape(1, 5, 5, 10)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    scores, _ = guard.score(context, torch.tensor(normalized, device="cuda"))
                torch.cuda.synchronize()
                scoring_seconds = time.perf_counter() - started
                scores = scores[0].cpu().numpy()
                rows.append({"episode": episode, "row": row, "initial_success": item["initial_success"],
                             "current_rgb_sha256": item["current_rgb_sha256"], "candidates": candidates.tolist(),
                             "gradient": gradient.detach().cpu().tolist(), "direction": direction.tolist(),
                             "autodiff_directional": direct, "finite_difference_directional": fd,
                             "gradient_error": error, "gradient_check_pass": grad_ok,
                             "gradient_nominal_cost": float(value.detach()), "scores": scores.tolist(),
                             "selected_index": int(np.argmin(scores)), "gradient_seconds": gradient_seconds,
                             "finite_difference_seconds": finite_difference_seconds, "scoring_seconds": scoring_seconds,
                             "gradient_forward_calls": 1, "backward_calls": 1, "fd_forward_calls": 2,
                             "candidate_chains_scored": 5})
                print(f"episode={episode} derivative_ok={grad_ok} selected={rows[-1]['selected_index']}", flush=True)
    finally:
        env.close()
    guard.verify_unchanged()
    with PLAN.open("x") as stream:
        json.dump({"track": RGB, "scope": "exposed development, unequal compute",
                   "dependencies": dependencies(), "model_digest": digest,
                   "normalizer_sha256": normalizer, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    data = json.loads(PLAN.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("sealed plan dependency drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Physically replay all previously sealed gradient candidates.", flush=True)
    try:
        with h5py.File(prior.source.common.DATA, "r") as db:
            for item in data["rows"]:
                traces = []
                for candidate in item["candidates"]:
                    actions = np.asarray(candidate, np.float32)
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.000001:
                        raise ValueError("illegal physical action witness")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(db["proprio"][item["row"]])
                    actor._set_goal_state(db["proprio"][item["row"] + 25])
                    if prior.img_hash(actor.render().copy()) != item["current_rgb_sha256"]:
                        raise ValueError("planning and replay image mismatch")
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "success": bool(term), "truncated": bool(trunc)})
                        if trunc:
                            break
                    traces.append({"trace": trace, "ever_success": any(t["success"] for t in trace),
                                   "complete": len(trace) == 25})
                rows.append({"episode": item["episode"], "selected_index": item["selected_index"],
                             "candidates": traces})
                print(f"episode={item['episode']} replayed", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"track": RGB, "scope": "post-selection physical evaluation, exposed tasks",
                   "plan_sha256": file_hash(PLAN), "rows": rows}, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
