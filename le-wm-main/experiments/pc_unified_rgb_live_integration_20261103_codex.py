"""One preregistered live RGB integration diagnostic for unified development v1."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_multidirection_rgb_fixed_terminal_dev_20261103_codex as previous
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import pc_unified_multidirection_dev_v1 as unified

runner = previous.runner
old, h, ROOT = runner.old, runner.h, runner.ROOT
PROTOCOL = ROOT.parent / "docs/PC_UNIFIED_RGB_LIVE_INTEGRATION_PROTOCOL_20261103_CODEX.md"
REPLAY = ROOT / "results/pc_unified_rgb_frozen_model_replay_20261103_codex.json"
PREFIX = ROOT / "results/pc_unified_rgb_live_integration_20261103_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
DECISIONS = [PREFIX.with_name(PREFIX.name + f"_decision{i}.json") for i in range(3)]
IMAGES = PREFIX.with_name(PREFIX.name + "_images.npz")
RECEIPT = PREFIX.with_name(PREFIX.name + "_execution.json")


def digest_context(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def main():
    if any(path.exists() for path in [MANIFEST, *DECISIONS, IMAGES, RECEIPT]):
        raise FileExistsError("prior or incomplete diagnostic exists")
    sha = old.prior.sha
    replay = json.loads(REPLAY.read_text())
    if any(sha(Path(path)) != digest for path, digest in replay["dependencies"].items()):
        raise ValueError("frozen replay dependency mismatch")
    files = {Path(path) for path in replay["dependencies"]}
    files.update((Path(__file__).resolve(), PROTOCOL, REPLAY))
    sources = {str(p): sha(p) for p in sorted(files)}
    baseline = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != baseline["history_sha256"]:
        raise ValueError("history mismatch")
    old.write_json(MANIFEST, {"purpose": "live integration diagnostic, not performance",
                              "root": 0, "dependencies": sources})
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("One exposed root: seal unified decisions before real RGB execution.", flush=True)
    started = time.monotonic()
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != baseline["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != baseline["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (not np.array_equal(mean.cpu().numpy(), baseline["action_mean"])
                or not np.array_equal(std.cpu().numpy(), baseline["action_std"])
                or not bool(torch.isfinite(std).all() and (std > 0).all())):
            raise ValueError("normalizer mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != baseline["goal_sha256"]:
            raise ValueError("goal mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            goal_position = ((target.mean(0) @ pinv.T)[:2] * pos_std).cpu().numpy()
            torch.cuda.synchronize()
        with runner.ShadowRGBBridge(np.asarray([*old.STARTS[0], 0., 0.])) as env:
            warm, warm_times = old.do_actions(env, old.WARMUP)
            frames = [env.initial_rgb.copy(), *warm]
            if ([old.digest_image(x) for x in frames] != baseline["rows"][0]["rgb_sha256"]
                    or not np.array_equal(np.stack(frames), histories["root0_history_rgb"])):
                raise ValueError("warmup history mismatch")
            actions = old.WARMUP.tolist()
            carry, rows = None, []
            for step in range(3):
                ticks = runner.previous.prior.context_ticks(len(frames))
                encode_started = time.monotonic()
                with torch.no_grad():
                    entries = [h.encode_with_state(model, frames[t],
                               h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                               for t in ticks]
                    torch.cuda.synchronize()
                ctx = {key: torch.cat([entry[key] for entry in entries], dim=1) for key in ("visual", "proprio")}
                history_actions = np.asarray(actions[-10:], dtype=float)
                history = ((torch.as_tensor(history_actions, dtype=torch.float32, device=device) - mean)
                           / std).reshape(2, 1, 10)
                encoding_seconds = time.monotonic() - encode_started
                binding = {"root": 0, "step": step, "ticks": ticks,
                           "rgb_sha256": [old.digest_image(frames[t]) for t in ticks],
                           "previous_rgb_sha256": [old.digest_image(frames[t-1]) for t in ticks],
                           "history_actions": history_actions.tolist(), "goal_sha256": old.digest_image(goal),
                           "goal_position": goal_position.tolist(), "model_hash": model_hash,
                           "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT),
                           "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(),
                           "carry": carry}
                meter = runner.base.Meter(unified.CAP)

                def predict(chains):
                    def raw(batch):
                        count, steps, _ = batch.shape
                        packets = ((torch.as_tensor(batch, dtype=torch.float32, device=device) - mean) / std)
                        packets = packets.reshape(count, steps // 5, 10).transpose(0, 1)
                        with torch.no_grad():
                            output = unroll_with_action_history(model, ctx, history.expand(2, count, 10), packets, 3)["proprio"][-1]
                            features = output.reshape(count, 256, 16)
                            position = (features.mean(1) @ pinv.T)[:, :2] * pos_std
                            scores = h.projected_cost(features, target, pinv, pos_std)
                            torch.cuda.synchronize()
                        return position.cpu().numpy(), scores.cpu().numpy()
                    return meter.predict(np.asarray(chains, dtype=np.float32), raw)

                plan_started = time.monotonic()
                compiled = unified.compile_decision(root=0, step=step, target=goal_position, predict=predict,
                                                    context_id=digest_context(binding), carry=carry)
                plan_seconds = time.monotonic() - plan_started
                outer = meter.as_dict()
                for name in ("prediction_calls", "trajectory_samples", "prediction_token_samples"):
                    if outer[name] != compiled["counts"][name]:
                        raise ValueError("nested meter mismatch")
                decision = {"binding": binding, "compiled": compiled, "outer_counts": outer,
                            "encoding_seconds": encoding_seconds, "planning_seconds": plan_seconds,
                            "manifest_sha256": sha(MANIFEST), "full_cost_certified": False}
                old.write_json(DECISIONS[step], decision)
                sealed_ns = time.monotonic_ns()
                observed, times = old.do_actions(env, compiled["dispatched_actions"])
                if not times or times[0][0] < sealed_ns:
                    raise ValueError("execution occurred before decision seal")
                frames.extend(observed)
                actions.extend(compiled["dispatched_actions"])
                rows.append({"step": step, "decision_sha256": sha(DECISIONS[step]), "sealed_ns": sealed_ns,
                             "actions": compiled["dispatched_actions"], "times_ns": times,
                             "rgb_sha256": [old.digest_image(image) for image in observed]})
                carry = compiled["remaining_tail"] if step < 2 else None
                print(f"step={step} calls={outer['prediction_calls']} tokens={outer['prediction_token_samples']} sealed and dispatched", flush=True)
            with IMAGES.open("xb") as stream:
                np.savez_compressed(stream, rgb=np.stack(frames))
            if len(frames) != 31 or h.parameter_hash(model) != model_hash:
                raise ValueError("frame count or frozen parameters changed")
            if sources != {str(p): sha(Path(p)) for p in sources}:
                raise ValueError("dependency drift")
            receipt = {"root": 0, "rows": rows, "warmup_times_ns": warm_times,
                       "executed_actions": actions[15:], "images_sha256": sha(IMAGES),
                       "manifest_sha256": sha(MANIFEST), "model_hash": model_hash,
                       "encoding_calls": 10, "elapsed_seconds": time.monotonic() - started,
                       "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                   "torch": torch.__version__, "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(device)},
                       "full_cost_certified": False}
            old.write_json(RECEIPT, receipt)
    print("Live integration diagnostic sealed.", flush=True)


if __name__ == "__main__":
    main()
