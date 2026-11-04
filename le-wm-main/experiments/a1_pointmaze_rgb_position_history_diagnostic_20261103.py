"""Compare single and actual-history prediction on already executed tails."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import (
    CAMERA, CAMERA_SHA, OUTPUT as SOURCE, SOURCE as PREFIX_SOURCE, SOURCE_SHA as PREFIX_SHA, image_hash,
)
from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import task_mask
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "4a041a1e3e1419a000f97407226817858529f874aa99ab4fdeba6ab5730c45a6"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_POSITION_HISTORY_DIAGNOSTIC_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_rgb_position_history_diagnostic_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def predictions(model, first, second, prefix, tails):
    """Predict identical candidates using cp5 only or cp0/cp5 with real prefix."""
    if prefix.shape != (10,) or tails.ndim != 2 or tails.shape[1] != 10 or tails.shape[0] < 1:
        raise ValueError("invalid normalized action packet shapes")
    n = tails.shape[0]
    with torch.no_grad():
        single = model.unroll(second, act_suffix=tails.unsqueeze(0))["visual"][-1]
        visual = torch.cat((first["visual"], second["visual"]), dim=1).expand(n, -1, -1, -1, -1, -1)
        proprio = torch.cat((first["proprio"], second["proprio"]), dim=1).expand(n, -1, -1, -1)
        actions = torch.stack((prefix.expand(n, -1), tails), dim=1)
        history, _, _ = model.model.forward_pred(visual, model.model.encode_act(actions), proprio)
        history = history[:, -1]
    if single.shape != history.shape or single.shape[0] != n:
        raise ValueError(f"unexpected output shapes: {single.shape}, {history.shape}")
    return single, history


def masked_cost(features, target, mask):
    if features.ndim != 5 or features.shape[1:] != target.shape:
        raise ValueError("incompatible latent shapes")
    errors = ((features - target) ** 2).mean(dim=(1, 4))
    return errors[:, mask].mean(dim=1)


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (PREFIX_SOURCE, PREFIX_SHA), (CAMERA, CAMERA_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    source, prior, camera = (json.loads(p.read_text()) for p in (SOURCE, PREFIX_SOURCE, CAMERA))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (PREFIX_SOURCE.with_suffix(".npz"), prior["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, prior["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", prior["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", prior["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source arrays or model input changed: {path}")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    offset = source["body_offset_from_static_scene"]
    rows, output_arrays, start = [], {}, time.monotonic()
    print("Fixed executed tails: RGB position single-frame versus actual-history prediction.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data, \
            np.load(PREFIX_SOURCE.with_suffix(".npz"), allow_pickle=False) as old, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid action normalizer")
        for row, prior_row in zip(source["rows"], prior["rows"], strict=True):
            root = row["root"]
            if root != prior_row["root"]:
                raise RuntimeError("root mismatch")
            goal_rgb = old[f"r{root}_goal_rgb"]
            goal = encode_with_state(model, goal_rgb, np.zeros(4), device)["visual"][0, 0]
            branches = []
            for i, parent in enumerate(prior_row["branches"]):
                group = row["branches"][i * 3:i * 3 + 3]
                if any(b["initial_method"] != parent["method"] for b in group):
                    raise RuntimeError("prefix method mismatch")
                keys = [f"r{root}_{parent['method']}_{b['policy']}" for b in group]
                histories = [data[f"{key}_history"] for key in keys]
                if not all(np.array_equal(histories[0], h) for h in histories[1:]):
                    raise RuntimeError("history frames differ across same-prefix arms")
                frames = histories[0]
                if image_hash(frames[0]) != image_hash(old[f"r{root}_rgb0"]) or image_hash(frames[-1]) != parent["rgb5_sha256"]:
                    raise RuntimeError("RGB source binding mismatch")
                states = [estimate_state(f, camera["metadata"], offset)[0] for f in (frames[0], frames[-1])]
                first, second = (encode_with_state(model, f, s, device)
                                 for f, s in zip((frames[0], frames[-1]), states, strict=True))
                tail_actions = np.stack([np.asarray(b["tail"], np.float32)[0] for b in group])
                prefix_action = np.asarray(parent["initial_plan"], np.float32)[0]
                prefix = ((torch.as_tensor(prefix_action, device=device) - mean) / std).repeat(5)
                tails = ((torch.as_tensor(tail_actions, device=device) - mean) / std).repeat(1, 5)
                mask, _ = task_mask(frames[-1], goal_rgb)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
                single, history = predictions(model, first, second, prefix, tails)
                with torch.no_grad():
                    actual = torch.stack([encode_with_state(model, data[f"{key}_rgb10"], np.zeros(4), device)["visual"][0, 0]
                                          for key in keys])
                    real_cost = masked_cost(actual, goal, mask_t)
                    costs = {"single": masked_cost(single, goal, mask_t),
                             "history": masked_cost(history, goal, mask_t)}
                    errors = {"single": ((single - actual) ** 2).mean(dim=(1, 2, 3, 4)),
                              "history": ((history - actual) ** 2).mean(dim=(1, 2, 3, 4))}
                key = f"r{root}_{parent['method']}"
                for name, tensor in (("single", single), ("history", history), ("actual", actual)):
                    output_arrays[f"{key}_{name}"] = tensor.cpu().numpy()
                output_arrays[f"{key}_mask"] = mask
                branches.append({"initial_method": parent["method"], "policies": [b["policy"] for b in group],
                                 "prefix_action": prefix_action.tolist(), "tail_actions": tail_actions.tolist(),
                                 "rgb0_sha256": image_hash(frames[0]), "rgb5_sha256": image_hash(frames[-1]),
                                 "rgb10_sha256": [b["rgb10_sha256"] for b in group],
                                 "input0": states[0].tolist(), "input5": states[1].tolist(),
                                 "actual_goal_cost": real_cost.cpu().tolist(),
                                 "predicted_goal_cost": {k: v.cpu().tolist() for k, v in costs.items()},
                                 "prediction_mse": {k: v.cpu().tolist() for k, v in errors.items()},
                                 "selected_index": {k: int(v.argmin().cpu()) for k, v in costs.items()},
                                 "distance_eval_only": [b["distance_eval_only"] for b in group]})
            rows.append({"root": root, "branches": branches})
            print(f"root {root}: fixed-tail history comparison complete", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **output_arrays)
    report = {"schema": "a1_pointmaze_rgb_position_history_diagnostic.v1", "source_sha256": SOURCE_SHA,
              "prefix_source_sha256": PREFIX_SHA, "camera_sha256": CAMERA_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "wall_seconds": time.monotonic() - start,
              "scope": "fixed executed tails on exposed roots; not independent P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
