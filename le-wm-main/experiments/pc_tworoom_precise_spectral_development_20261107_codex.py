"""RGB spectral versus random proposal: sealed model plan and physical replay."""
import hashlib
import json
import sys
import time

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_precise_spectral_proposals_20261107_codex import jacobian_and_directions, make_candidates
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest
from experiments.pc_tworoom_search_region_multihorizon_derivative_20261107_codex import normalize_actions

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_PRECISE_SPECTRAL_PROPOSAL_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_precise_spectral_development_plan_20261107_codex.json"
PHYSICAL = ROOT / "results/pc_tworoom_precise_spectral_development_physical_20261107_codex.json"
EPISODES = (9000, 9013, 9022, 9032)


def img_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def dependencies():
    paths = [__file__, PROTOCOL, source.OUTPUT, source.INPUT, source.common.DATA,
             source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json",
             ROOT / "experiments/pc_rgb_precise_spectral_proposals_20261107_codex.py",
             ROOT / "experiments/pc_rgb_derivative_backend_20261107_codex.py"]
    return {str(p): file_hash(p) for p in paths}


def checked_previous():
    old = json.loads(source.OUTPUT.read_text())
    if (old["input_sha256"] != file_hash(source.INPUT)
            or old["protocol_sha256"] != file_hash(source.PROTOCOL)
            or old["runner_sha256"] != file_hash(source.__file__)):
        raise ValueError("sealed CEM source drift")
    return old


def plan():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PLAN.exists() or PHYSICAL.exists():
        raise FileExistsError("output exists")
    old = checked_previous()
    model, scaler, transform, normalizer = source.common.model_setup()
    if model_fingerprint(model) != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.OUTPUT, source.INPUT,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="5 env steps per packet")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Equal candidate count: RGB latent Jacobian spectral directions versus random directions.", flush=True)
    try:
        with np.load(source.INPUT, allow_pickle=False) as inputs, h5py.File(source.common.DATA, "r") as db:
            for episode in EPISODES:
                indexes = np.flatnonzero(inputs["episodes"] == episode)
                matches = [r for r in old["rows"] if r["episode"] == episode]
                if len(indexes) != 1 or len(matches) != 1:
                    raise ValueError("ambiguous task")
                i, entry = int(indexes[0]), matches[0]
                task_row = int(inputs["rows"][i])
                if entry["row"] != task_row:
                    raise ValueError("root identity mismatch")
                raw_plan = np.asarray(entry["arms"]["cem16"]["decisions"][0]["actions"], dtype=np.float32)
                if raw_plan.shape != (25, 2) or not np.isfinite(raw_plan).all() or np.max(np.abs(raw_plan)) > 1.000001:
                    raise ValueError("invalid sealed action chain")
                raw = np.clip(raw_plan, -0.7, 0.7)
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(db["proprio"][task_row])
                actor._set_goal_state(db["proprio"][task_row + 25])
                pixels = actor.render().copy()
                context = {"pixels": transform(pixels)[None, None, None].cuda(),
                           "goal": transform(inputs["goal"][i])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                raw_t = torch.tensor(raw, dtype=torch.float32, device="cuda")
                def rollout(x):
                    _, ctx = guard.score(context, normalize_actions(x, scaler.mean_, scaler.scale_))
                    return ctx["predicted_emb"][0, 0, -1]
                torch.cuda.synchronize()
                start = time.perf_counter()
                jac, singular, dirs = jacobian_and_directions(rollout, raw_t)
                torch.cuda.synchronize()
                spectral_seconds = time.perf_counter() - start
                random_dirs = np.random.default_rng(27127000 + episode).normal(size=(2, 25, 2))
                directions = {"spectral": dirs.cpu().numpy(), "random": random_dirs}
                arms = {}
                for arm in ("spectral", "random"):
                    candidates = make_candidates(raw, directions[arm])
                    normalized = scaler.transform(candidates.reshape(-1, 2)).astype(np.float32).reshape(1, 5, 5, 10)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        scores, _ = guard.score(context, torch.tensor(normalized, device="cuda"))
                    torch.cuda.synchronize()
                    score_seconds = time.perf_counter() - start
                    values = scores[0].cpu().numpy()
                    chosen = int(np.argmin(values))
                    arms[arm] = {"directions": directions[arm].tolist(), "candidates": candidates.tolist(),
                                 "scores": values.tolist(), "selected_index": chosen,
                                 "scoring_seconds": score_seconds, "scored_chains": 5,
                                 "jvp_calls": 50 if arm == "spectral" else 0,
                                 "jvp_forward_and_reverse_autodiff": arm == "spectral",
                                 "jacobian_svd_seconds": spectral_seconds if arm == "spectral" else 0}
                rows.append({"episode": episode, "row": task_row, "initial_success": entry["initial_success"],
                             "current_rgb_sha256": img_hash(pixels), "goal_rgb_sha256": img_hash(inputs["goal"][i]),
                             "nominal_clipped_coordinates": int(np.count_nonzero(raw != raw_plan)),
                             "singular_values": singular.cpu().tolist(), "jacobian_shape": list(jac.shape),
                             "arms": arms})
                print(f"episode={episode} score spectral={arms['spectral']['scores'][arms['spectral']['selected_index']]:.3f} "
                      f"random={arms['random']['scores'][arms['random']['selected_index']]:.3f}", flush=True)
    finally:
        env.close()
    guard.verify_unchanged()
    with PLAN.open("x") as f:
        json.dump({"scope": "exposed RGB development, candidate count not matched compute",
                   "track": RGB, "dependencies": dependencies(), "model_digest": digest,
                   "normalizer_sha256": normalizer, "rows": rows}, f, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    planned = json.loads(PLAN.read_text())
    if planned["dependencies"] != dependencies():
        raise ValueError("plan dependency drift")
    rows = []
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    print("Replay both preselected model choices and all 10 candidate chains per episode.", flush=True)
    try:
        with h5py.File(source.common.DATA, "r") as db:
            for item in planned["rows"]:
                arms = {}
                for arm in ("spectral", "random"):
                    traces = []
                    for candidate in item["arms"][arm]["candidates"]:
                        actions = np.asarray(candidate, dtype=np.float32)
                        if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.000001:
                            raise ValueError("invalid action witness")
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(db["proprio"][item["row"]])
                        actor._set_goal_state(db["proprio"][item["row"] + 25])
                        if img_hash(actor.render().copy()) != item["current_rgb_sha256"]:
                            raise ValueError("planning/execution initial RGB mismatch")
                        trace = []
                        for action in actions:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append({"distance": float(info["distance_to_target"]),
                                          "success": bool(term), "truncated": bool(trunc)})
                            if trunc:
                                break
                        traces.append({"trace": trace, "ever_success": any(t["success"] for t in trace),
                                       "complete": len(trace) == 25})
                    arms[arm] = {"selected_index": item["arms"][arm]["selected_index"], "candidates": traces}
                rows.append({"episode": item["episode"], "initial_success": item["initial_success"], "arms": arms})
                print(f"episode={item['episode']} complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"scope": "post-hoc physical evaluation of sealed RGB candidates", "track": RGB,
                   "plan_sha256": file_hash(PLAN), "rows": rows}, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
