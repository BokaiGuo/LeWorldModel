"""Host CUDA smoke of AP-compatible PointMaze score and packet mapping."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.a0_pointmaze_predicted_proprio_goal_score_20261103 import score
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA, OUTPUT as SOURCE
from experiments.a1_pointmaze_proprio_score_feedback_cem_20261103 import SOURCE_SHA
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.pc_ap_pointmaze_callback_20261103 import PointMazeAPCallback
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PROTOCOL = ROOT.parent / "docs/PC_AP_POINTMAZE_CALLBACK_SMOKE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_ap_pointmaze_callback_smoke_20261103_DEV01.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, expected in ((SOURCE, SOURCE_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    source, camera, old = (json.loads(path.read_text()) for path in (SOURCE, CAMERA, OLD))
    if sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise RuntimeError("source arrays changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data:
        rgb = data["r1888_rgb0"].copy()
        goal_rgb = data["r1888_goal_rgb"].copy()
        mask = data["r1888_mask"].copy()
        actions = np.clip(data["r1888_initial_noise"][0, :, 0, :], -1, 1).astype(np.float32)
    if actions.shape != (16, 2):
        raise ValueError("unexpected fixed action bank")
    print("Compare the same 16 PointMaze action packets through both score paths.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("checkpoint model identity mismatch")
        current_state, _ = estimate_state(rgb, camera["metadata"], old["body_offset_from_static_scene"])
        goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
        current = encode_with_state(model, rgb, current_state, device)
        goal = encode_with_state(model, goal_rgb, goal_state, device)
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid action standard deviation")
        raw = torch.as_tensor(actions, device=device)
        normalized = ((raw - mean) / std).repeat(1, 5).reshape(1, 16, 1, 10)
        callback = PointMazeAPCallback(model, current, goal["proprio"][0, 0], device)
        full = callback(normalized)
        split = torch.cat((callback(normalized[:, :8]), callback(normalized[:, 8:])), dim=1)
        _, reference = score(model, current, goal, mask, actions[:, None, :], mean, std)
        np.testing.assert_allclose(full.cpu().numpy()[0], reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(full, split, rtol=1e-4, atol=1e-6)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("frozen model changed")
    report = {
        "scope": "known-root CUDA score adapter smoke; no AP controller or physical result",
        "root": 1888, "source_sha256": sha256_file(SOURCE),
        "arrays_sha256": sha256_file(SOURCE.with_suffix(".npz")),
        "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
        "callback_sha256": sha256_file(ROOT / "experiments/spectral_reachability/pc_ap_pointmaze_callback_20261103.py"),
        "model_before": before, "model_after": after, "callback_batches": callback.batches,
        "callback_samples": callback.samples, "reference_samples": len(actions),
        "max_reference_difference": float(np.max(np.abs(full.cpu().numpy()[0] - reference))),
        "max_batch_difference": float((full - split).abs().max().item()),
        "physical_executions": 0,
    }
    raw_report = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw_report)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw_report).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
