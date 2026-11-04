"""Evaluate RGB-derived goal proprio token on sealed development candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_failure_real_rgb_metric_20261103 import (
    ORACLE, ORACLE_SHA, ORIGINAL, ORIGINAL_SHA, PLANS, PLANS_SHA,
)
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import pairwise_ranking
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PREDICTED_PROPRIO_GOAL_SCORE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_predicted_proprio_goal_score_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def score(model, context, goal, mask, proposals, mean, std):
    plans = torch.as_tensor(proposals, dtype=torch.float32, device=mean.device)
    suffix = ((plans - mean) / std).repeat(1, 1, 5).permute(1, 0, 2)
    with torch.no_grad():
        output = model.unroll(context, act_suffix=suffix)
        visual = output["visual"][-1]
        prop = output["proprio"][-1]
        visual_cost = ((visual - goal["visual"][0, 0]) ** 2).mean(dim=(1, 4))
        visual_cost = visual_cost[:, torch.as_tensor(mask, device=mean.device)].mean(dim=1)
        prop_cost = ((prop - goal["proprio"][0, 0]) ** 2).reshape(len(plans), -1).mean(dim=1)
    return visual_cost.cpu().numpy(), prop_cost.cpu().numpy()


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    sources = ((ORACLE, ORACLE_SHA), (PLANS, PLANS_SHA), (ORIGINAL, ORIGINAL_SHA),
               (CAMERA, CAMERA_SHA), (OLD, OLD_SHA))
    for path, expected in sources:
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    oracle, plans, original, camera, old = (json.loads(p.read_text()) for p, _ in sources)
    for path, expected in (
        (ORACLE.with_suffix(".npz"), oracle["arrays_sha256"]),
        (PLANS.with_suffix(".npz"), plans["arrays_sha256"]),
        (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
        (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
        (DEFAULT_DATASET / "actions.pth", original["actions_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, started = {}, [], time.monotonic()
    print("Scoring all 624 fixed candidates using model-predicted proprio and RGB-derived goal.", flush=True)
    with np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(PLANS.with_suffix(".npz"), allow_pickle=False) as inputs, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != plans["model_before"]:
            raise RuntimeError("model identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid action normalizer")
        # Complete all scoring before reading the physical distance arrays.
        for row in oracle["rows"]:
            root, method, phase, n = row["root"], row["method"], row["phase"], row["n"]
            key = f"r{root}_{method}_{phase}"
            goal_rgb = inputs[f"r{root}_goal_rgb"]
            goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            goal = encode_with_state(model, goal_rgb, goal_state, device)
            if phase == "initial":
                current = inputs[f"r{root}_rgb0"]
                current_state, _ = estimate_state(current, camera["metadata"], old["body_offset_from_static_scene"])
            elif phase == "tail":
                history = inputs[f"r{root}_{method}_history"]
                current = history[-1]
                current_state = causal_state(history[-2], history[-1], camera["metadata"],
                                             old["body_offset_from_static_scene"], .05)
            else:
                raise ValueError("unknown phase")
            context = encode_with_state(model, current, current_state, device)
            candidates = bank[f"{key}_candidates"]
            if len(candidates) != n:
                raise RuntimeError("candidate count mismatch")
            visual, prop = score(model, context, goal, inputs[f"r{root}_mask"], candidates, mean, std)
            if not np.isfinite(prop).all() or not np.isfinite(visual).all():
                raise ValueError("nonfinite scores")
            np.testing.assert_allclose(visual, bank[f"{key}_scores"], rtol=1e-4, atol=1e-6)
            arrays[f"{key}_visual"] = visual
            arrays[f"{key}_proprio"] = prop
            arrays[f"{key}_goal_token"] = goal["proprio"].cpu().numpy()
            rows.append({"root": root, "method": method, "phase": phase, "n": n,
                         "goal_rgb_state": goal_state.tolist(), "context_rgb_state": current_state.tolist(),
                         "visual_choice": int(np.argmin(visual)), "proprio_choice": int(np.argmin(prop))})
            print(f"{key}: {n} model predictions scored", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
        for row in rows:
            key = f"r{row['root']}_{row['method']}_{row['phase']}"
            physical = bank[f"{key}_distances"]
            for name in ("visual", "proprio"):
                row[f"{name}_physical_distance"] = float(physical[row[f"{name}_choice"]])
                row[f"{name}_vs_physical"] = pairwise_ranking(arrays[f"{key}_{name}"], physical)
            row["physical_oracle_distance"] = float(np.min(physical))
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_predicted_proprio_goal_score.v1", "sources": {
        str(path.relative_to(ROOT)): digest for path, digest in sources},
        "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
        "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
        "device": torch.cuda.get_device_name(device), "prediction_samples": sum(x["n"] for x in rows),
        "scope": "outcome-exposed development fixed bank; not P1 or prospective control",
        "wall_seconds": time.monotonic() - started, "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
