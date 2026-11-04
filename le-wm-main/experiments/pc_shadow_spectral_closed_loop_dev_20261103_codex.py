"""Development-only independent RGB feedback arms; physical execution is opt-in."""
import argparse
import hashlib
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from experiments.pc_pointmaze_durable_dispatch_20261103_codex import (
    DurableJournal, _rgb, _sync_dir, _verify_snapshot, dispatch_reset, save_snapshot,
)
from experiments.pc_pointmaze_shadow_closed_loop_adapter_20261103_codex import RGBClock
from experiments.spectral_reachability.pc_spectral_seeded_feedback_cem_20261103_codex import (
    seeded_cem, spectral_candidates,
)


def decide(arm, evaluate, response, mean, std, noise, synchronize=lambda: None, event=lambda *a, **k: None):
    if arm not in ("baseline", "spectral"):
        raise ValueError("unknown arm")
    prior, lower, upper = ((-mean)/std).repeat(5), ((-1-mean)/std).repeat(5), ((1-mean)/std).repeat(5)
    counts = dict(response_forward_calls=0, reverse_vector_calls=0, svd_calls=0, fd_model_forwards=0)
    structure = None
    if arm == "spectral":
        def counted(x):
            counts["response_forward_calls"] += 1
            event("structure_response_attempt", count=counts["response_forward_calls"])
            value = response(x)
            event("structure_response_returned")
            return value
        synchronize()
        started = time.monotonic()
        event("structure_attempt")
        with torch.enable_grad():
            structure = spectral_candidates(counted, mean, std)
        synchronize()
        if counts["response_forward_calls"] != 1:
            raise RuntimeError("unexpected response calls")
        if structure["structure_counts"] != {"response_forward_calls": 1, "reverse_vector_calls": 2, "svd_calls": 1}:
            raise RuntimeError("unexpected structure counts")
        counts.update(structure["structure_counts"])
        event("structure_returned", counts=counts, seconds=time.monotonic()-started)
    result = seeded_cem(evaluate, prior, lower, upper, noise,
                        None if structure is None else structure["normalized"])
    if result["cem_counts"] != {"samples": 50, "batches": 4}:
        raise RuntimeError("CEM budget mismatch")
    return result, structure, counts


def dispatch_one(clock, action, snapshot, digest, journal, output, arm, phase, tick):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
        raise ValueError("invalid action")
    _verify_snapshot(snapshot, digest, tick, action)
    journal.record("bridge_attempt", bridge=arm, branch=arm, phase=phase, clock=tick,
                   snapshot_path=str(snapshot), snapshot_sha256=digest, action=action.tolist())
    started = time.monotonic()
    image = _rgb(clock.advance(action.copy()))
    if clock.clock != tick:
        raise RuntimeError("clock mismatch")
    path = Path(str(output) + "_rgb.npz")
    digest_rgb = save_snapshot(path, rgb=image, clock=np.asarray(tick))
    journal.record("bridge_returned", bridge=arm, branch=arm, phase=phase, clock=tick,
                   snapshot_sha256=digest, rgb_path=str(path), artifact_sha256=digest_rgb,
                   rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                   seconds=time.monotonic()-started)
    return image.copy()


def run_arm(arm, factory, prefix, planner, output_dir, journal, iterations=5, check_frame=None):
    """Planner sees only its own consecutive RGB, iteration, and no bridge handle."""
    output_dir = Path(output_dir)
    prefix = np.asarray(prefix, dtype=np.float64)
    if arm not in ("baseline", "spectral") or prefix.ndim != 2 or prefix.shape[1] != 2:
        raise ValueError("invalid arm or prefix")
    snapshot = output_dir / f"{arm}_reset.npz"
    digest = save_snapshot(snapshot, clock=np.asarray(0), arm=np.asarray(arm))
    env = dispatch_reset(arm, factory, snapshot, digest, journal, output_dir / f"{arm}_reset", arm)
    with closing(env):
        clock = RGBClock(env)
        frames, history = [env.initial_rgb.copy()], []
        if check_frame:
            check_frame(0, frames[0])
        for tick in range(1, len(prefix)+iterations+1):
            phase = "prefix" if tick <= len(prefix) else "feedback"
            arrays = {}
            if phase == "prefix":
                action = prefix[tick-1].copy()
            else:
                prev, current = clock.pair()
                action, arrays = planner(prev, current, tick-len(prefix)-1)
                action = np.asarray(action, dtype=np.float64)
            snapshot = output_dir / f"{arm}_command{tick}.npz"
            digest = save_snapshot(snapshot, clock=np.asarray(tick), action=action,
                arm=np.asarray(arm), phase=np.asarray(phase), current_rgb=frames[-1],
                previous_rgb=frames[-2] if len(frames) > 1 else np.empty((0,), np.uint8),
                action_history=np.asarray(history, dtype=np.float64).reshape(-1, 2), **arrays)
            image = dispatch_one(clock, action, snapshot, digest, journal,
                                 output_dir / f"{arm}_command{tick}", arm, phase, tick)
            history.append(action.copy())
            frames.append(image)
            if phase == "prefix" and check_frame:
                check_frame(tick, image)
        return clock.pair(), np.asarray(history), np.stack(frames)


def physical_run(arms, output_dir):
    # Imports are deliberately deferred: importing/testing this module never starts a bridge.
    from experiments import pc_pointmaze_shadow_closed_loop_durable_20261103_codex as old
    from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
    from experiments.pc_pointmaze_cp12_renderer_source_audit_20261103_codex import gl_info
    from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge

    prior, h, root = old.prior, old.h, old.ROOT
    for path, digest in old.EXPECTED.items():
        if prior.sha(path) != digest:
            raise ValueError(f"locked dependency changed: {path}")
    specpath = root / "experiments/spectral_reachability/pc_spectral_seeded_feedback_cem_20261103_codex.py"
    if prior.sha(specpath) != "54be2507a20d699805d6b4f9816298424864cdea6e8c08ebb6cdb1836142d26a":
        raise ValueError("spectral module changed")
    source, archive, names = prior.files()
    prior.load_sealed(prior.CONTEXT)
    if any(names[i] != name for i, name in old.SELECTED):
        raise ValueError("roster changed")
    gl = gl_info()
    if gl["gl_vendor"] != "NVIDIA Corporation" or "NVIDIA" not in gl["gl_renderer"]:
        raise RuntimeError("NVIDIA host renderer required")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    _sync_dir(output_dir.parent)
    with DurableJournal(output_dir / "events.jsonl") as journal:
        try:
            protocol = root.parent / "docs/PC_POINTMAZE_SHADOW_SPECTRAL_CLOSED_LOOP_DEV_PROTOCOL_20261103_CODEX.md"
            journal.record("preflight", arms=arms, gl=gl,
                           dependencies={**prior.deps(), **{str(p): prior.sha(p) for p in
                               (*old.EXPECTED, specpath, Path(__file__), protocol)}})
            with np.load(prior.SOURCE.with_suffix(".npz"), allow_pickle=False) as src, \
                 np.load(prior.CONTEXT.with_suffix(".npz"), allow_pickle=False) as oldctx, \
                 np.load(old.V4, allow_pickle=False) as v4, \
                 np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
                 h.lock.gpu_lock(root / "results/e3_p2_host_cuda_cooperative.lock") as wait:
                device = h.require_cuda()
                if device.type != "cuda":
                    raise RuntimeError("CUDA required")
                args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                          config=h.DEFAULT_CONFIG, dinov2_source=root / "external/local_dinov2")
                model, dataset = h.load_model_and_dataset(args, device)
                model.eval()
                before = h.parameter_hash(model)
                if before != source["model_before"]:
                    raise ValueError("model mismatch")
                mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
                if (mean.cpu().tolist() != archive["provenance"]["action_mean"] or
                    std.cpu().tolist() != archive["provenance"]["action_std"] or
                    not bool(torch.isfinite(std).all() and (std > 0).all()) or
                    not np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, atol=1e-7, rtol=0) or
                    not np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, atol=1e-7, rtol=0)):
                    raise ValueError("normalization mismatch")
                weights, _, _ = h.get_encoder()
                pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
                pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
                import json
                camera, historical = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
                metadata, offset = camera["metadata"], historical["body_offset_from_static_scene"]
                goal = obstacle["goal_rgb"]
                goal_state, _ = h.estimate_state(goal, metadata, offset)
                with torch.no_grad():
                    target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
                journal.record("model_loaded", model_hash=before, gpu_lock_wait_seconds=wait,
                               device=torch.cuda.get_device_name(device), goal_encodes=1)
                for index, name in old.SELECTED:
                    prefix = src[name+"_actions"][:12].copy()
                    if prefix.shape != (12, 2) or not np.array_equal(prefix, oldctx[name+"_actions"]):
                        raise ValueError("prefix binding mismatch")
                    start = src["starts"][index].copy()
                    if start.shape != (4,) or not np.isfinite(start).all():
                        raise ValueError("invalid setup")
                    for arm in arms:
                        branch_dir = output_dir / f"{name}_{arm}"
                        branch_dir.mkdir()
                        _sync_dir(output_dir)
                        def check_frame(tick, frame):
                            if old.image_hash(frame) != v4["shadow_hashes"][2*index, tick]:
                                raise RuntimeError("prefix RGB hash mismatch")
                        def encode(previous, current):
                            estimate = h.causal_state(previous, current, metadata, offset, .05)
                            with torch.no_grad():
                                context = h.encode_with_state(model, current, estimate, device)
                                score = h.projected_cost(context["proprio"].reshape(1, 256, 16),
                                                         target, pinv, pos_std)
                            if not np.isfinite(estimate).all() or not bool(torch.isfinite(score).all()):
                                raise ValueError("invalid RGB feedback")
                            return context, estimate, float(score.item())
                        def planner(previous, current, iteration):
                            started = time.monotonic()
                            context, estimate, score = encode(previous, current)
                            seed = 2026110400 + 10*index + iteration
                            noise = torch.randn((3, 16, 10), generator=torch.Generator().manual_seed(seed)).to(device)
                            counts = {"samples": 0, "batches": 0}
                            def event(kind, **fields):
                                journal.record(kind, root=name, arm=arm, iteration=iteration, **fields)
                            def response(action):
                                future = action.reshape(1, 1, 10)
                                pred = unroll_with_action_history(model, context, future[:0], future, 1)["proprio"][-1]
                                return (pred.reshape(256, 16).mean(dim=0) @ pinv.T)[:2] * pos_std
                            def evaluate(candidate):
                                n = len(candidate)
                                event("predict_attempt", samples=n)
                                with torch.no_grad():
                                    future = candidate.unsqueeze(0)
                                    pred = unroll_with_action_history(model, context, future[:0], future, 1)["proprio"][-1]
                                    value = h.projected_cost(pred.reshape(n, 256, 16), target, pinv, pos_std)
                                counts["samples"] += n
                                counts["batches"] += 1
                                event("predict_returned", samples=n)
                                return value
                            result, structure, scounts = decide(arm, evaluate, response, mean, std, noise,
                                                                torch.cuda.synchronize, event)
                            if counts != result["cem_counts"]:
                                raise RuntimeError("meter mismatch")
                            raw = (result["chosen"].reshape(5, 2)*std+mean).clamp(-1, 1)
                            torch.testing.assert_close(((raw-mean)/std).reshape(10), result["chosen"], rtol=1e-5, atol=1e-6)
                            torch.cuda.synchronize()
                            arrays = {"seed": np.asarray(seed), "noise": noise.cpu().numpy(),
                                      "rgb_derived_state": np.asarray(estimate), "rgb_score": np.asarray(score),
                                      "plan_raw": raw.cpu().numpy(), "chosen": result["chosen"].cpu().numpy(),
                                      "final": result["final"].cpu().numpy(),
                                      "final_scores": result["final_scores"].cpu().numpy(),
                                      "winner": np.asarray(result["winner"])}
                            for j, trace in enumerate(result["trace"]):
                                for label, tensor in zip(("candidates", "scores", "elite", "mean", "std"), trace):
                                    arrays[f"cem{j}_{label}"] = tensor.cpu().numpy()
                            if structure:
                                for key in ("raw", "normalized", "jacobian_raw", "singular_values", "direction_raw"):
                                    arrays["structure_"+key] = structure[key].cpu().numpy()
                            event("decision", seed=seed, score=score, cem_counts=counts,
                                  structure_counts=scounts, seconds=time.monotonic()-started)
                            return raw[0].cpu().numpy().astype(np.float64), arrays
                        (previous, current), actions, frames = run_arm(
                            arm, lambda: ShadowRGBBridge(start), prefix, planner, branch_dir,
                            journal, check_frame=check_frame)
                        _, estimate, score = encode(previous, current)
                        path = branch_dir / "trajectory.npz"
                        digest = save_snapshot(path, actions=actions, frames=frames,
                                               final_rgb_derived_state=estimate, final_rgb_score=np.asarray(score))
                        journal.record("branch_completed", root=name, arm=arm, final_rgb_score=score,
                                       trajectory_path=str(path), trajectory_sha256=digest,
                                       feedback_encodes=6, commands=len(actions))
                if h.parameter_hash(model) != before:
                    raise RuntimeError("model parameters changed")
                journal.record("completed", model_hash=before, branches=2*len(arms),
                               equal_total_compute=False, physical_experiment=True)
        except BaseException as exc:
            journal.record("failed", error_type=type(exc).__name__, message=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "spectral", "paired"), default="paired")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-physical", action="store_true")
    args = parser.parse_args()
    if not args.execute_physical:
        parser.error("physical execution requires explicit --execute-physical authorization")
    physical_run(("baseline", "spectral") if args.arm == "paired" else (args.arm,), args.output_dir)


if __name__ == "__main__":
    main()
