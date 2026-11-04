"""Development native RGB feedback: remaining versus extended time objective."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from experiments.a0_pointmaze_archive_bank_alignment_20261102 import SOURCE, SOURCE_SHA, score
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT,
    load_model_and_dataset, parameter_hash, require_cuda, sha256_file,
)

OUTPUT = ROOT / "results/a0_pointmaze_native_feedback_time_20261102_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_NATIVE_FEEDBACK_TIME_PROTOCOL_20261102.md"
ROOTS = (1880, 1881, 1882, 1883)


def encode(model, image, device):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise ValueError("invalid raw RGB")
    observation = {
        "visual": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None, None].to(device),
        "proprio": torch.zeros(1, 1, 4, device=device),
    }
    with torch.no_grad():
        encoded = model.encode(observation)
    return TensorDict({"visual": encoded["visual"], "proprio": encoded["proprio"]},
                      batch_size=[1, 1], device=device)


def goal_latent(context):
    return context["visual"][0, 0].mean(dim=(0, 1, 2)).detach()


def main():
    if OUTPUT.exists() or sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("output exists or source bank changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    protocol_sha = sha256_file(PROTOCOL)
    bank = np.asarray(json.loads(SOURCE.read_text())["rows"][0]["candidate_bank"][:8], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("candidate bank outside action bounds")
    started = time.monotonic()
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        mean, std = dataset.action_mean.cpu().numpy(), dataset.action_std.cpu().numpy()
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("invalid action normalizer")
        normalized = (bank - mean) / std
        one = np.repeat(normalized[:, None, :], 5, axis=1).reshape(8, 10)
        pairs = np.stack((np.repeat(one, 8, axis=0), np.tile(one, (8, 1))), axis=0)
        one_t, pairs_t = torch.as_tensor(one[None], device=device), torch.as_tensor(pairs, device=device)
        rows = []
        for root in ROOTS:
            path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            images = torch.load(path, map_location="cpu", weights_only=True).numpy()
            initial_state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal = goal_latent(encode(model, images[10], device))
            branches = []
            selected_first = None
            for method in ("remaining", "extended"):
                with NativeRGBBridge(initial_state) as env:
                    ctx0 = encode(model, env.initial_rgb, device)
                    start = time.monotonic()
                    scores0 = score(model, ctx0, goal, pairs_t, 8).reshape(8, 8)
                    first, plan_tail = np.unravel_index(np.argmin(scores0), scores0.shape)
                    if selected_first is not None and first != selected_first:
                        raise RuntimeError("identical initial plans selected different first actions")
                    selected_first = int(first)
                    rgb5 = env.step(np.repeat(bank[first][None], 5, axis=0))
                    ctx5 = encode(model, rgb5, device)
                    if method == "remaining":
                        tail_scores = score(model, ctx5, goal, one_t, 8)
                        tail = int(np.argmin(tail_scores))
                    else:
                        tail_scores = score(model, ctx5, goal, pairs_t, 8).reshape(8, 8)
                        tail = int(np.unravel_index(np.argmin(tail_scores), tail_scores.shape)[0])
                    rgb10 = env.step(np.repeat(bank[tail][None], 5, axis=0))
                    branches.append({"method": method, "first": int(first), "initial_tail": int(plan_tail),
                                     "chosen_tail": tail, "initial_costs": scores0.tolist(),
                                     "feedback_costs": tail_scores.tolist(), "model_encoding_calls": 2,
                                     "prediction_samples": 64 + (8 if method == "remaining" else 64),
                                     "predicted_block_transitions": 128 + (8 if method == "remaining" else 128),
                                     "physical_actions": 10, "execution_and_planning_seconds": time.monotonic() - start,
                                     "rgb5_sha256": hashlib.sha256(rgb5.tobytes()).hexdigest(),
                                     "rgb10_sha256": hashlib.sha256(rgb10.tobytes()).hexdigest()})
            if branches[0]["rgb5_sha256"] != branches[1]["rgb5_sha256"]:
                raise RuntimeError("matched first action did not produce matched feedback")
            # Privileged state is read only after both planner decisions are fixed.
            target = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            sequences = np.stack([np.concatenate((np.repeat(bank[b["first"]][None], 5, axis=0),
                                                  np.repeat(bank[b["chosen_tail"]][None], 5, axis=0)))
                                  for b in branches])
            endpoints = physical.replay_position_endpoints(np.repeat(initial_state[None], 2, axis=0), sequences)
            for branch, endpoint in zip(branches, endpoints):
                branch["physical_position_eval_only"] = endpoint[:2].tolist()
                branch["physical_distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - target))
            rows.append({"root": root, "goal_frame": 10, "goal_rgb_sha256": hashlib.sha256(images[10].tobytes()).hexdigest(),
                         "archive_episode_sha256": sha256_file(path), "branches": branches})
            print(f"completed development root {root}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("frozen model changed")
    report = {"schema": "a0_pointmaze_native_feedback_time.v1",
              "scope": "exposed roots, constant proprio, unequal horizon cost; not P1 or independent evidence",
              "protocol_sha256": protocol_sha, "candidate_source_sha256": SOURCE_SHA,
              "script_sha256": sha256_file(Path(__file__)),
              "bridge_sha256": sha256_file(ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py"),
              "checkpoint_sha256": sha256_file(args.checkpoint), "model_before": before, "model_after": after,
              "states_sha256": sha256_file(args.dataset_root / "states.pth"),
              "actions_sha256": sha256_file(args.dataset_root / "actions.pth"),
              "normalizer_source": "all available archive trajectories; training split unknown",
              "goal_encoder_calls_per_root_shared": 1, "device": torch.cuda.get_device_name(device),
              "camera": "development selected from root 1880 frame 0; lookat=(3,3)",
              "physical_evaluator": physical.position_backend_receipt(),
              "wall_seconds": time.monotonic() - started, "bank": bank.tolist(), "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
