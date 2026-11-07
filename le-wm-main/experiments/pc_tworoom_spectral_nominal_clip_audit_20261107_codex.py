"""Post-hoc action clipping audit for the sealed RGB spectral experiment."""
import json
import sys

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_precise_spectral_development_20261107_codex as prior
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_SPECTRAL_NOMINAL_CLIP_AUDIT_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_spectral_nominal_clip_audit_plan_20261107_codex.json"
PHYSICAL = ROOT / "results/pc_tworoom_spectral_nominal_clip_audit_physical_20261107_codex.json"


def action_pair(raw, recorded):
    raw, recorded = np.asarray(raw, np.float32), np.asarray(recorded, np.float32)
    if raw.shape != (25, 2) or not np.isfinite(raw).all() or np.abs(raw).max() > 1.000001:
        raise ValueError("invalid original action")
    clipped = np.clip(raw, -0.7, 0.7)
    if not np.array_equal(clipped, recorded):
        raise ValueError("stored parent does not match clipping")
    return np.stack([raw, clipped])


def load_previous():
    plan = json.loads(prior.PLAN.read_text())
    physical = json.loads(prior.PHYSICAL.read_text())
    old = prior.checked_previous()
    if physical["plan_sha256"] != file_hash(prior.PLAN) or plan["dependencies"] != prior.dependencies():
        raise ValueError("prior artifacts drift")
    return plan, physical, old


def plan():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PLAN.exists() or PHYSICAL.exists():
        raise FileExistsError("output exists")
    previous, _, original = load_previous()
    model, scaler, transform, normalizer = prior.source.common.model_setup()
    digest = model_digest(model)
    if digest != previous["model_digest"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.PLAN, prior.PHYSICAL, prior.source.OUTPUT,
                         prior.source.INPUT, prior.source.common.CHECKPOINT / "weights.pt",
                         prior.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="25 env steps, 5 packets")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Score original versus clipped CEM plans, before physical replay.", flush=True)
    try:
        with np.load(prior.source.INPUT, allow_pickle=False) as inputs, h5py.File(prior.source.common.DATA, "r") as db:
            for entry in previous["rows"]:
                episode, row = entry["episode"], entry["row"]
                index, = np.flatnonzero(inputs["episodes"] == episode)
                item, = (r for r in original["rows"] if r["episode"] == episode)
                actions = action_pair(item["arms"]["cem16"]["decisions"][0]["actions"],
                                      entry["arms"]["spectral"]["candidates"][0])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(db["proprio"][row])
                actor._set_goal_state(db["proprio"][row + 25])
                frame = actor.render().copy()
                if prior.img_hash(frame) != entry["current_rgb_sha256"]:
                    raise ValueError("render mismatch")
                context = {"pixels": transform(frame)[None, None, None].cuda(),
                           "goal": transform(inputs["goal"][index])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                normal = scaler.transform(actions.reshape(-1, 2)).astype(np.float32).reshape(1, 2, 5, 10)
                with torch.no_grad():
                    costs, ctx = guard.score(context, torch.tensor(normal, device="cuda"))
                    timeline = (ctx["predicted_emb"][:, :, 1:] -
                                ctx["goal_emb"][:, None, -1, None]).square().sum(-1)
                rows.append({"episode": episode, "row": row, "initial_success": entry["initial_success"],
                             "current_rgb_sha256": prior.img_hash(frame), "actions": actions.tolist(),
                             "clipped_coordinates": int(np.count_nonzero(actions[0] != actions[1])),
                             "max_action_difference": float(np.abs(actions[0] - actions[1]).max()),
                             "endpoint_cost": costs[0].cpu().tolist(),
                             "temporal_min_cost": timeline.min(-1).values[0].cpu().tolist()})
                print(f"episode={episode} clipped={rows[-1]['clipped_coordinates']}", flush=True)
    finally:
        env.close()
    guard.verify_unchanged()
    with PLAN.open("x") as f:
        json.dump({"track": RGB, "scope": "post-hoc matched RGB plans before physical replay",
                   "source_plan_sha256": file_hash(prior.PLAN), "source_physical_sha256": file_hash(prior.PHYSICAL),
                   "original_sha256": file_hash(prior.source.OUTPUT), "runner_sha256": file_hash(__file__),
                   "protocol_sha256": file_hash(PROTOCOL), "model_digest": digest,
                   "normalizer_sha256": normalizer, "rows": rows}, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    previous, old_physical, original = load_previous()
    planned = json.loads(PLAN.read_text())
    if (planned["source_plan_sha256"] != file_hash(prior.PLAN)
            or planned["source_physical_sha256"] != file_hash(prior.PHYSICAL)
            or planned["original_sha256"] != file_hash(prior.source.OUTPUT)
            or planned["runner_sha256"] != file_hash(__file__)
            or planned["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("plan binding drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Replay original and clipped 25-step plans without feedback.", flush=True)
    try:
        with h5py.File(prior.source.common.DATA, "r") as db:
            for p, prior_result in zip(planned["rows"], old_physical["rows"], strict=True):
                if p["episode"] != prior_result["episode"]:
                    raise ValueError("episode mismatch")
                old, = (r for r in original["rows"] if r["episode"] == p["episode"])
                traces = []
                for action in np.asarray(p["actions"], np.float32):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(db["proprio"][p["row"]])
                    actor._set_goal_state(db["proprio"][p["row"] + 25])
                    if prior.img_hash(actor.render().copy()) != p["current_rgb_sha256"]:
                        raise ValueError("RGB mismatch")
                    trace = []
                    for a in action:
                        _, _, term, trunc, info = actor.step(a.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "success": bool(term), "truncated": bool(trunc)})
                        if trunc:
                            break
                    traces.append(trace)
                previous_trace = prior_result["arms"]["spectral"]["candidates"][0]["trace"]
                if traces[1] != previous_trace:
                    raise ValueError("clipped physical trace mismatch")
                dispatched = old["arms"]["cem16"]["decisions"][0]["dispatched"]
                np.testing.assert_array_equal(np.asarray(dispatched, np.float32),
                                              np.asarray(p["actions"][0], np.float32)[:len(dispatched)])
                recorded = old["arms"]["cem16"]["trace"][:len(dispatched)]
                if len(recorded) != len(dispatched) or len(traces[0]) < len(recorded):
                    raise ValueError("original dispatched trace incomplete")
                for a, b in zip(traces[0][:len(recorded)], recorded, strict=True):
                    if abs(a["distance"] - b["distance"]) > 1e-6 or a["success"] != b["terminated"]:
                        raise ValueError("original dispatched trace mismatch")
                rows.append({"episode": p["episode"], "initial_success": p["initial_success"], "traces": traces})
                print(f"episode={p['episode']} physical complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"track": RGB, "scope": "post-hoc open-loop physical evaluation",
                   "plan_sha256": file_hash(PLAN), "rows": rows}, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
