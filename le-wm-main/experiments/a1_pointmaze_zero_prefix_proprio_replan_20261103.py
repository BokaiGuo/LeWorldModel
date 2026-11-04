"""Execute zero-prefix RGB feedback tails matched to spectral development."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_failure_real_rgb_metric_20261103 import ORIGINAL, ORIGINAL_SHA
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_predicted_proprio_goal_score_20261103 import score
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA, OUTPUT as NOISE
from experiments.a1_pointmaze_proprio_score_feedback_cem_20261103 import SOURCE_SHA as NOISE_SHA
from experiments.a1_pointmaze_proprio_spectral_cem_20261103 import OUTPUT as SOURCE
from experiments.a1_pointmaze_proprio_spectral_common_suffix_20261103 import OUTPUT as COMMON
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "b39bc04c3bceb2f54cc982b32d5517f5ae6be10d8f9ede598ac7496ab3ecda71"
COMMON_SHA = "0f6ea3d1248eff0f79a433fff30002c5ec1df820e282cab5e8225e280a7515f5"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_ZERO_PREFIX_PROPRIO_REPLAN_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_zero_prefix_proprio_replan_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    sources = ((SOURCE, SOURCE_SHA), (NOISE, NOISE_SHA), (COMMON, COMMON_SHA),
               (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA))
    for path, expected in sources:
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    source, noise, common, original, camera, old = (json.loads(p.read_text()) for p, _ in sources)
    for path, expected in ((NOISE.with_suffix(".npz"), noise["arrays_sha256"]),
                           (COMMON.with_suffix(".npz"), common["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", original["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"input changed: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    common_rows = {(r["root"], r["prefix"], r["suffix"]): r for r in common["rows"]}
    arrays, rows, starts, sequences, final_rgb = {}, [], [], [], []
    started = time.monotonic()
    print("Executing zero-prefix real-RGB feedback replanning on eight development roots.", flush=True)
    with np.load(NOISE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid action normalizer")
        if [r["root"] for r in source["rows"]] != list(range(1888, 1896)):
            raise RuntimeError("unexpected roots")
        for row in source["rows"]:
            root = row["root"]
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_rgb = saved[f"r{root}_goal_rgb"]
            goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            goal = encode_with_state(model, goal_rgb, goal_state, device)
            mask, innovations = saved[f"r{root}_mask"], saved[f"r{root}_tail_noise"]
            with NativeRGBBridge(initial) as env:
                if not np.array_equal(env.initial_rgb, saved[f"r{root}_rgb0"]):
                    raise RuntimeError("initial RGB mismatch")
                frames = [env.initial_rgb]
                for _ in range(5):
                    frames.append(env.step(np.zeros((1, 2), np.float32)))
                torch.cuda.synchronize()
                timer = time.monotonic()
                state5 = causal_state(frames[-2], frames[-1], camera["metadata"],
                                      old["body_offset_from_static_scene"], .05)
                feedback = encode_with_state(model, frames[-1], state5, device)
                counts = {"batches": 0, "samples": 0}
                def scorer(proposals):
                    counts["batches"] += 1
                    counts["samples"] += len(proposals)
                    return score(model, feedback, goal, mask, proposals, mean, std)[1]
                tail, cost, trace = cem(innovations, scorer)
                torch.cuda.synchronize()
                plan_seconds = time.monotonic() - timer
                rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                if env.step_count != 10:
                    raise RuntimeError("action clock mismatch")
            if counts != {"batches": 3, "samples": 48}:
                raise RuntimeError("tail sample count mismatch")
            starts.append(initial)
            sequences.append(np.concatenate((np.zeros((5, 2), np.float32),
                                              np.repeat(tail[0][None], 5, axis=0))))
            final_rgb.append(rgb10)
            arrays[f"r{root}_history"], arrays[f"r{root}_rgb10"] = np.stack(frames), rgb10
            rows.append({"root": root, "goal_rgb_state": goal_state.tolist(), "cp5_rgb_state": state5.tolist(),
                         "tail": tail.tolist(), "tail_cost": cost, "tail_trace": trace, "prediction_counts": counts,
                         "feedback_encoding_and_planning_seconds": plan_seconds,
                         "current_rgb_encodes": 1, "goal_rgb_encodes": 1, "physical_steps": 10})
            print(f"root {root}: zero-prefix and replanned tail executed", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    legacy_rgb, legacy_end = replay_legacy(starts, sequences, xml)
    cp5 = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences)[:, :5])
    endpoints = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(legacy_rgb, np.stack(final_rgb)) or not np.array_equal(endpoints[:, :2], legacy_end[:, :2]):
        raise RuntimeError("replay mismatch")
    for i, row in enumerate(rows):
        root = row["root"]
        np.testing.assert_array_equal(cp5[i], common_rows[root, "zero", "zero"]["cp5_state"])
        goal = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
        distance = float(np.linalg.norm(endpoints[i, :2]-goal))
        row.update({"cp5_state_eval_only": cp5[i].tolist(), "goal_position_eval_only": goal.tolist(),
                    "endpoint_eval_only": legacy_end[i].tolist(), "distance_eval_only": distance,
                    "strict_endpoint_success_eval_only": bool(distance < .5),
                    "prior_random_distance": source["rows"][i]["branches"][0]["distance_eval_only"],
                    "prior_spectral_distance": source["rows"][i]["branches"][1]["distance_eval_only"]})
    arrays["sequences"], arrays["endpoints_eval_only"], arrays["cp5_eval_only"] = np.asarray(sequences), legacy_end, cp5
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_zero_prefix_proprio_replan.v1",
              "source_hashes": {str(p.relative_to(ROOT)): h for p, h in sources},
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "device": torch.cuda.get_device_name(device),
              "model_before": before, "model_after": after, "physical_executions": 8, "physical_steps": 80,
              "separate_replay_trajectories": 8, "wall_seconds": time.monotonic()-started, "rows": rows,
              "scope": "exposed roots; matched tail samples not full cost; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
