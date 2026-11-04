"""Two-root closed-loop shadow RGB integration smoke; no physical-state planner input."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pc_pointmaze_cp12_renderer_source_audit_20261103_codex import gl_info
from experiments.pc_pointmaze_shadow_closed_loop_adapter_20261103_codex import RGBClock
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

h = prior.h
ROOT = prior.ROOT
PREFIX = "pc_pointmaze_shadow_closed_loop_dev_20261103_CODEX"
OUTPUT = ROOT / "results" / (PREFIX + ".json")
JOURNAL = OUTPUT.with_name(PREFIX + "_events.jsonl")
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SHADOW_CLOSED_LOOP_DEV_PROTOCOL_20261103_CODEX.md"
V4 = ROOT / "results/pc_pointmaze_shadow_renderer_audit_v4_20261103_CODEX.npz"
EXPECTED = {
    PROTOCOL: "82cbce45d006d8e01c516a509a873793727ad4b608fd1f37874749f3832d0c41",
    ROOT / "experiments/pc_pointmaze_shadow_closed_loop_adapter_20261103_codex.py":
        "2b7619518e91351479e97096c73eb039526079da807a387b5f6736d4050cd296",
    ROOT / "experiments/pc_pointmaze_shadow_rgb_bridge_20261103_codex.py":
        "97bc794e5cbb73a78e64138c0843caa4505deb983d7f802fe2a1fb717dbae33f",
    ROOT / "experiments/pc_pointmaze_shadow_rgb_bridge_worker_20261103_codex.py":
        "478d322dac6614033f30f145308c9935501ade7168e427175ccd2995bd679e9c",
    ROOT / "experiments/pc_pointmaze_shadow_renderer_worker_v4_20261103_codex.py":
        "71e43269d5577cd1dd7439763a8c660793b5ff3f2d0d147e24ab5badd679d34c",
    ROOT / "experiments/pc_pointmaze_cp12_rgb_history_planning_20261103_codex.py":
        "d4b21f1b1a7c3339ba0e900c8c5d3e12312dc86b1a3d03e7c1b7d9ceb16f2ca4",
    V4: "0d254110f530af01acdb357a56f12cb41682476f00071d218c20a146649d12e7",
}
SELECTED = ((0, "r8343_a0_p2"), (8, "r8345_a0_p2"))


def image_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def save_arrays(path, **arrays):
    with path.open("xb") as out:
        np.savez_compressed(out, **arrays)
    return prior.sha(path)


def main():
    if any((ROOT / "results").glob(PREFIX + "*")):
        raise FileExistsError("existing run artifacts; refuse overwrite or automatic retry")
    for path, digest in EXPECTED.items():
        if prior.sha(path) != digest:
            raise ValueError(f"locked input changed: {path}")
    source, archive, names = prior.files()
    prior.load_sealed(prior.CONTEXT)
    if any(names[i] != name for i, name in SELECTED):
        raise ValueError("frozen branch roster changed")
    gl = gl_info()
    if gl["gl_vendor"] != "NVIDIA Corporation" or "NVIDIA" not in gl["gl_renderer"]:
        raise RuntimeError("NVIDIA host renderer required")
    dependencies = {**prior.deps(), **{str(p): prior.sha(p) for p in EXPECTED},
                    str(Path(__file__)): prior.sha(Path(__file__)),
                    str(prior.CONTEXT): prior.sha(prior.CONTEXT),
                    str(prior.CONTEXT.with_suffix(".npz")): prior.sha(prior.CONTEXT.with_suffix(".npz"))}
    counts = {"goal_encodes": 0, "feedback_encodes": 0, "predict_samples": 0,
              "predict_batches": 0, "shadow_resets": 0, "legacy_resets": 0,
              "shadow_commands": 0, "legacy_commands": 0}
    started = time.monotonic()
    rows = []
    with JOURNAL.open("x") as journal:
        def event(kind, **data):
            journal.write(json.dumps({"event": kind, "time_ns": time.time_ns(), **data},
                                     allow_nan=False, sort_keys=True) + "\n")
            journal.flush()

        try:
            with np.load(prior.SOURCE.with_suffix(".npz"), allow_pickle=False) as src, \
                 np.load(prior.CONTEXT.with_suffix(".npz"), allow_pickle=False) as old_ctx, \
                 np.load(V4, allow_pickle=False) as v4, \
                 np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
                 h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
                event("load_attempt")
                device = h.require_cuda()
                if device.type != "cuda":
                    raise RuntimeError("CUDA required")
                args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                          config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
                model, dataset = h.load_model_and_dataset(args, device)
                before = h.parameter_hash(model)
                if before != source["model_before"]:
                    raise ValueError("frozen parameter hash mismatch")
                mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
                if (mean.cpu().tolist() != archive["provenance"]["action_mean"]
                        or std.cpu().tolist() != archive["provenance"]["action_std"]
                        or not bool(torch.isfinite(std).all() and (std > 0).all())):
                    raise ValueError("action normalization mismatch")
                if (not np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, atol=1e-7, rtol=0)
                        or not np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, atol=1e-7, rtol=0)):
                    raise ValueError("proprio normalization mismatch")
                lower, upper = ((-1-mean)/std).repeat(5), ((1-mean)/std).repeat(5)
                zero_prior = ((-mean)/std).repeat(5)
                weights, _, _ = h.get_encoder()
                pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
                pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
                camera, old = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
                metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
                goal_rgb = obstacle["goal_rgb"]
                goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
                with torch.no_grad():
                    target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
                counts["goal_encodes"] += 1
                event("load_returned", model_hash=before, device=torch.cuda.get_device_name(device))

                # This model-facing function accepts only RGB, fixed camera metadata and the frozen target.
                def encode_feedback(previous_rgb, current_rgb):
                    estimate = h.causal_state(previous_rgb, current_rgb, metadata, offset, .05)
                    with torch.no_grad():
                        encoded = h.encode_with_state(model, current_rgb, estimate, device)
                        cost = h.projected_cost(encoded["proprio"].reshape(1, 256, 16), target, pinv, pos_std)
                    if not torch.isfinite(cost).all() or not np.isfinite(estimate).all():
                        raise ValueError("nonfinite RGB-derived feedback")
                    counts["feedback_encodes"] += 1
                    return encoded, estimate, float(cost.item())

                for source_index, name in SELECTED:
                    prefix = src[name+"_actions"][:12].copy()
                    if prefix.shape != (12, 2) or not np.array_equal(prefix, old_ctx[name+"_actions"]):
                        raise ValueError("prefix action binding failed")
                    start = src["starts"][source_index].copy()  # Environment setup only; not planner input.
                    if start.shape != (4,) or not np.isfinite(start).all():
                        raise ValueError("invalid setup")
                    event("reset_attempt", branch=name, bridges=2)
                    with ShadowRGBBridge(start) as shadow, h.NativeRGBBridge(start) as legacy:
                        counts["shadow_resets"] += 1
                        counts["legacy_resets"] += 1
                        feed, control = RGBClock(shadow), RGBClock(legacy)
                        shadow_frames, raw_frames = [shadow.initial_rgb.copy()], [legacy.initial_rgb.copy()]
                        executed = []
                        def check_prefix(t):
                            if (image_hash(shadow_frames[t]) != v4["shadow_hashes"][2*source_index, t]
                                    or not np.array_equal(raw_frames[t], old_ctx[name+"_frames"][t])):
                                raise RuntimeError(f"prefix RGB binding mismatch {name} tick {t}")
                        check_prefix(0)
                        for t, action in enumerate(prefix):
                            event("command_attempt", branch=name, clock=t+1, phase="prefix", action=action.tolist())
                            begin = time.monotonic()
                            shadow_frames.append(feed.advance(action))
                            counts["shadow_commands"] += 1
                            shadow_seconds = time.monotonic()-begin
                            begin = time.monotonic()
                            raw_frames.append(control.advance(action))
                            counts["legacy_commands"] += 1
                            legacy_seconds = time.monotonic()-begin
                            executed.append(np.asarray(action).copy())
                            check_prefix(t+1)
                            event("command_returned", branch=name, clock=t+1, phase="prefix",
                                  shadow_sha=image_hash(shadow_frames[-1]), raw_sha=image_hash(raw_frames[-1]),
                                  shadow_seconds=shadow_seconds, legacy_seconds=legacy_seconds)
                        branch = {"name": name, "source_index": source_index, "prefix_checks": 13, "steps": []}
                        for iteration in range(5):
                            previous_rgb, current_rgb = feed.pair()
                            torch.cuda.synchronize()
                            t0 = time.monotonic()
                            context, estimated, realized = encode_feedback(previous_rgb, current_rgb)
                            seed = 2026110400 + 10*source_index + iteration
                            generator = torch.Generator(device="cpu").manual_seed(seed)
                            noise = torch.randn((3, 16, 10), generator=generator, dtype=torch.float32).to(device)
                            def evaluate(candidate):
                                n = candidate.shape[0]
                                future = candidate.unsqueeze(0)
                                event("predict_attempt", branch=name, iteration=iteration, samples=n)
                                with torch.no_grad():
                                    pred = unroll_with_action_history(model, context, future[:0], future, 1)["proprio"][-1]
                                    cost = h.projected_cost(pred.reshape(n, 256, 16), target, pinv, pos_std)
                                counts["predict_samples"] += n
                                counts["predict_batches"] += 1
                                event("predict_returned", branch=name, iteration=iteration, samples=n)
                                return cost
                            chosen, trace, final, scores, winner = prior.cem(evaluate, zero_prior, lower, upper, noise)
                            torch.cuda.synchronize()
                            planner_seconds = time.monotonic()-t0
                            raw = (chosen.reshape(5, 2)*std+mean).clamp(-1, 1)
                            torch.testing.assert_close(((raw-mean)/std).reshape(10), chosen, rtol=1e-5, atol=1e-6)
                            packet = raw.cpu().numpy()
                            action = packet[0].copy()
                            saved = {"previous_rgb": previous_rgb, "current_rgb": current_rgb,
                                     "rgb_derived_state": np.asarray(estimated), "noise": noise.cpu().numpy(),
                                     "plan_raw": packet, "chosen": chosen.cpu().numpy(),
                                     "final_candidates": final.cpu().numpy(), "final_scores": scores.cpu().numpy(),
                                     "realized_rgb_score": np.asarray(realized), "seed": np.asarray(seed),
                                     "winner": np.asarray(winner)}
                            for j, values in enumerate(trace):
                                for label, value in zip(("candidates", "scores", "elite", "mean", "std"), values):
                                    saved[f"iteration{j}_{label}"] = value.cpu().numpy()
                            snapshot = OUTPUT.with_name(f"{PREFIX}_{name}_decision{iteration}.npz")
                            digest = save_arrays(snapshot, **saved)
                            event("decision_sealed", branch=name, iteration=iteration, path=str(snapshot),
                                  sha256=digest, action=action.tolist(), seed=seed, rgb_score=realized)
                            event("command_attempt", branch=name, clock=13+iteration, phase="feedback",
                                  action=action.tolist())
                            begin = time.monotonic()
                            shadow_frames.append(feed.advance(action))
                            counts["shadow_commands"] += 1
                            shadow_seconds = time.monotonic()-begin
                            begin = time.monotonic()
                            raw_frames.append(control.advance(action))
                            counts["legacy_commands"] += 1
                            legacy_seconds = time.monotonic()-begin
                            executed.append(action)
                            event("command_returned", branch=name, clock=13+iteration, phase="feedback",
                                  shadow_sha=image_hash(shadow_frames[-1]), raw_sha=image_hash(raw_frames[-1]),
                                  shadow_seconds=shadow_seconds, legacy_seconds=legacy_seconds)
                            branch["steps"].append({"iteration": iteration, "seed": seed, "action": action.tolist(),
                                "rgb_score_before": realized, "final_scores": scores.cpu().tolist(),
                                "winner": winner, "planner_seconds": planner_seconds,
                                "shadow_seconds": shadow_seconds, "legacy_seconds": legacy_seconds,
                                "decision_path": str(snapshot), "decision_sha256": digest})
                        previous_rgb, current_rgb = feed.pair()
                        _, final_estimated, final_score = encode_feedback(previous_rgb, current_rgb)
                        if feed.clock != 17 or control.clock != 17:
                            raise RuntimeError("terminal clock mismatch")
                        trajectory = OUTPUT.with_name(f"{PREFIX}_{name}_trajectory.npz")
                        branch["trajectory_sha256"] = save_arrays(
                            trajectory, actions=np.stack(executed), shadow_frames=np.stack(shadow_frames),
                            legacy_frames=np.stack(raw_frames), final_rgb_derived_state=np.asarray(final_estimated))
                        branch["trajectory_path"] = str(trajectory)
                        branch["final_rgb_score"] = final_score
                        branch["raw_shadow_different_frames"] = sum(
                            not np.array_equal(a, b) for a, b in zip(shadow_frames, raw_frames, strict=True))
                        rows.append(branch)
                        event("branch_completed", branch=name, final_rgb_score=final_score,
                              trajectory_sha256=branch["trajectory_sha256"])
                        print(f"completed {name}", flush=True)
                torch.cuda.synchronize()
                after = h.parameter_hash(model)
                if after != before:
                    raise RuntimeError("model parameters changed")
                expected_counts = {"goal_encodes": 1, "feedback_encodes": 12, "predict_samples": 500,
                                   "predict_batches": 40, "shadow_resets": 2, "legacy_resets": 2,
                                   "shadow_commands": 34, "legacy_commands": 34}
                if counts != expected_counts:
                    raise RuntimeError(f"budget mismatch: {counts}")
                event("completed", counts=counts, model_hash=after)
            report = {"schema": "pointmaze_shadow_closed_loop_dev.v1",
                      "scope": "two exposed roots, same-action rendering control, no performance claim",
                      "rows": rows, "counts": counts, "model_before": before, "model_after": after,
                      "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
                      "torch_cuda": torch.version.cuda, "gl_probe": gl, "elapsed_seconds": time.monotonic()-started,
                      "dependencies": dependencies, "journal_sha256": prior.sha(JOURNAL)}
            with OUTPUT.open("x") as out:
                json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
                out.write("\n")
        except BaseException as exc:
            event("failed", error_type=type(exc).__name__, message=str(exc))
            raise
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
