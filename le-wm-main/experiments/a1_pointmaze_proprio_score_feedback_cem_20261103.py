"""Compare two frozen-model scores in RGB feedback CEM on fixed dev roots."""
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
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import (
    OLD, OLD_SHA, OUTPUT as SOURCE, cem_four,
)
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "9c8e00d5a5d1f1401108bbcfcd5dfda12658a9f7d49468189131a7db461bcb8c"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PROPRIO_SCORE_FEEDBACK_CEM_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_proprio_score_feedback_cem_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA), (OLD, OLD_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    source, original, camera, old = (json.loads(p.read_text()) for p in (SOURCE, ORIGINAL, CAMERA, OLD))
    for path, expected in (
        (SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
        (DEFAULT_CHECKPOINT, original["checkpoint_sha256"]),
        (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
        (DEFAULT_DATASET / "actions.pth", original["actions_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, starts, sequences, final_frames = {}, [], [], [], []
    started = time.monotonic()
    print("Execute visual and proprio score CEM on eight fixed development roots.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid normalizer")
        if [row["root"] for row in source["rows"]] != list(range(1888, 1896)):
            raise RuntimeError("unexpected roots")
        for prior in source["rows"]:
            root = prior["root"]
            path = DEFAULT_DATASET / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(path) != prior["archive_sha256"]:
                raise RuntimeError("archive changed")
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            rgb0, goal_rgb, mask = saved[f"r{root}_rgb0"], saved[f"r{root}_goal_rgb"], saved[f"r{root}_mask"]
            goal_state, _ = estimate_state(goal_rgb, camera["metadata"], old["body_offset_from_static_scene"])
            goal = encode_with_state(model, goal_rgb, goal_state, device)
            initial_state, _ = estimate_state(rgb0, camera["metadata"], old["body_offset_from_static_scene"])
            noise0, noise5 = saved[f"r{root}_initial_noise"], saved[f"r{root}_tail_noise"]
            branches = []
            for arm in ("visual", "proprio"):
                with NativeRGBBridge(initial) as env:
                    if not np.array_equal(env.initial_rgb, rgb0):
                        raise RuntimeError("reset RGB mismatch")
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    context = encode_with_state(model, rgb0, initial_state, device)
                    counts = {"initial_samples": 0, "tail_samples": 0, "initial_batches": 0, "tail_batches": 0}
                    def scorer(context_, phase):
                        def call(proposals):
                            counts[f"{phase}_samples"] += len(proposals)
                            counts[f"{phase}_batches"] += 1
                            scores = score(model, context_, goal, mask, proposals, mean, std)
                            return scores[0 if arm == "visual" else 1]
                        return call
                    plan, cost0, trace0 = cem_four(noise0, scorer(context, "initial"))
                    torch.cuda.synchronize()
                    initial_seconds = time.monotonic() - t0
                    history = [env.initial_rgb]
                    for action in np.repeat(plan[0][None], 5, axis=0):
                        history.append(env.step(action[None]))
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    state5 = causal_state(history[-2], history[-1], camera["metadata"],
                                          old["body_offset_from_static_scene"], .05)
                    feedback = encode_with_state(model, history[-1], state5, device)
                    tail, cost5, trace5 = cem(noise5, scorer(feedback, "tail"))
                    torch.cuda.synchronize()
                    tail_seconds = time.monotonic() - t0
                    rgb10 = env.step(np.repeat(tail[0][None], 5, axis=0))
                    if env.step_count != 10:
                        raise RuntimeError("action clock mismatch")
                if counts != {"initial_samples": 64, "tail_samples": 48, "initial_batches": 4, "tail_batches": 3}:
                    raise RuntimeError("prediction count mismatch")
                if arm == "visual":
                    previous = next(b for b in prior["branches"] if b["method"] == "random_extra")
                    np.testing.assert_array_equal(plan, np.asarray(previous["initial_plan"], np.float32))
                    np.testing.assert_array_equal(tail, np.asarray(previous["tail"], np.float32))
                    np.testing.assert_array_equal(np.stack(history), saved[f"r{root}_random_extra_history"])
                    np.testing.assert_array_equal(rgb10, saved[f"r{root}_random_extra_rgb10"])
                sequence = np.concatenate((np.repeat(plan[0][None], 5, axis=0),
                                           np.repeat(tail[0][None], 5, axis=0)))
                starts.append(initial)
                sequences.append(sequence)
                final_frames.append(rgb10)
                key = f"r{root}_{arm}"
                arrays[f"{key}_history"], arrays[f"{key}_rgb10"] = np.stack(history), rgb10
                branches.append({"arm": arm, "initial_plan": plan.tolist(), "initial_cost": cost0,
                                 "initial_trace": trace0, "cp5_input": state5.tolist(), "tail": tail.tolist(),
                                 "tail_cost": cost5, "tail_trace": trace5, "prediction_counts": counts,
                                 "current_rgb_encodes": 2, "goal_rgb_encodes_per_root_shared": 1,
                                 "initial_encode_and_plan_seconds": initial_seconds,
                                 "feedback_encode_and_plan_seconds": tail_seconds,
                                 "physical_steps": 10, "rgb10_sha256": image_hash(rgb10)})
                print(f"root {root} {arm}: 10 physical steps", flush=True)
            rows.append({"root": root, "goal_rgb_state": goal_state.tolist(), "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    replay_rgb, endpoints = replay_legacy(starts, sequences, xml)
    endpoints2 = physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
    if not np.array_equal(replay_rgb, np.stack(final_frames)) or not np.array_equal(endpoints[:, :2], endpoints2[:, :2]):
        raise RuntimeError("physical replay mismatch")
    for i, row in enumerate(rows):
        goal_eval = dataset.states[row["root"], 10, :2].cpu().numpy().astype(np.float64)
        row["goal_position_eval_only"] = goal_eval.tolist()
        for j, branch in enumerate(row["branches"]):
            endpoint = endpoints[2*i+j]
            distance = float(np.linalg.norm(endpoint[:2] - goal_eval))
            branch.update({"endpoint_eval_only": endpoint.tolist(), "distance_eval_only": distance,
                           "strict_endpoint_success_eval_only": bool(distance < .5)})
            if branch["arm"] == "visual":
                previous = next(b for b in source["rows"][i]["branches"] if b["method"] == "random_extra")
                np.testing.assert_array_equal(endpoint, np.asarray(previous["endpoint_eval_only"]))
    arrays["sequences"], arrays["endpoints_eval_only"] = np.asarray(sequences), endpoints
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_proprio_score_feedback_cem.v1", "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "physical_executions": len(sequences),
              "physical_steps": 10*len(sequences), "separate_replay_trajectories": len(sequences),
              "wall_seconds": time.monotonic()-started, "rows": rows,
              "scope": "exposed development roots; fixed sample budget; not independent P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
