"""Velocity ablation for fixed-prefix, fixed-tail PointMaze predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import mujoco
import numpy as np
import torch

from experiments.a0_pointmaze_fixed_bank_cp5_refresh_20261103 import (
    CAMERA, CAMERA_SHA, NOISE_SHA, NOISE_SOURCE, PLAN_SHA, PLAN_SOURCE, REAL, REAL_SHA,
    SOURCE, SOURCE_SHA, metrics, predict,
)
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import pairwise_ranking
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

PREV = ROOT / "results/a0_pointmaze_fixed_bank_cp5_refresh_20261103_DEV01.json"
PREV_SHA = "36c814bbe1123a64dcccc77c6b0a0e818d96df87f08044b6d8b345589508ef0c"
WORKER = ROOT / "experiments/pointmaze_cp5_velocity_worker_20261103.py"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP5_VELOCITY_SAME_ACTION_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_cp5_velocity_same_action_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    paths = ((SOURCE, SOURCE_SHA), (REAL, REAL_SHA), (PLAN_SOURCE, PLAN_SHA),
             (NOISE_SOURCE, NOISE_SHA), (CAMERA, CAMERA_SHA), (PREV, PREV_SHA))
    for path, expected in paths:
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    source, real, plan, noise, camera, previous = (json.loads(path.read_text()) for path, _ in paths)
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (REAL.with_suffix(".npz"), real["arrays_sha256"]),
                           (NOISE_SOURCE.with_suffix(".npz"), noise["arrays_sha256"]),
                           (PREV.with_suffix(".npz"), previous["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, noise["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", noise["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", noise["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene mismatch")
    if not np.isclose(mujoco.MjModel.from_xml_string(xml).opt.timestep * 5, .05, rtol=0, atol=1e-12):
        raise RuntimeError("frame interval differs from fixed protocol")
    offset = plan["body_offset_from_static_scene"]
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    starts, sequences = [], []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank:
        for row in source["rows"]:
            plans = bank[f"r{row['root']}_{row['method']}_plans"]
            if plans.shape != (48, 2, 2):
                raise ValueError("candidate shape mismatch")
            starts.extend([states[row["root"], 0]] * 48)
            sequences.extend(np.repeat(x, 5, axis=0) for x in plans)
    if len(starts) != 384:
        raise ValueError("incomplete candidate batch")
    started = time.monotonic()
    print("Same-action RGB velocity ablation: replay all 384 sealed candidates.", flush=True)
    with tempfile.TemporaryDirectory(prefix="pc_cp5_velocity_") as tmp:
        request, scene, response = (Path(tmp) / name for name in ("request.npz", "scene.xml", "response.npz"))
        np.savez_compressed(request, sources=np.asarray(starts, np.float64), sequences=np.asarray(sequences, np.float64))
        scene.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(scene), str(response)],
                              capture_output=True, text=True, env=env, timeout=300)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"replay failed: {proc.stderr[-2000:]}")
        print(proc.stdout, flush=True)
        with np.load(response, allow_pickle=False) as data:
            rgb4, rgb5, rgb10, midstate, endpoints = (data[x].copy() for x in
                                                       ("rgb4", "rgb5", "rgb10", "midstate", "endpoints"))
    with np.load(PREV.with_suffix(".npz"), allow_pickle=False) as prior, \
            np.load(REAL.with_suffix(".npz"), allow_pickle=False) as actual:
        if not (np.array_equal(rgb5, prior["rgb5"]) and np.array_equal(rgb10, actual["rgb10"])
                and np.array_equal(endpoints, actual["legacy_endpoints"])):
            raise RuntimeError("cp5/cp10 replay differs from sealed images or full state")
    if rgb4.shape != (384, 224, 224, 3) or midstate.shape != (384, 4):
        raise ValueError("invalid physical output shape")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays = {"rgb4": rgb4, "midstate_privileged": midstate}
    rows = []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(NOISE_SOURCE.with_suffix(".npz"), allow_pickle=False) as inputs, \
            np.load(REAL.with_suffix(".npz"), allow_pickle=False) as actual, \
            np.load(PREV.with_suffix(".npz"), allow_pickle=False) as prior, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != previous["model_before"]:
            raise RuntimeError("model mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid action normalization")
        for group, row in enumerate(source["rows"]):
            root, method = row["root"], row["method"]
            key = f"r{root}_{method}"
            mask = torch.as_tensor(inputs[f"r{root}_initial_mask"], dtype=torch.bool, device=device)
            goal = encode_with_state(model, inputs[f"r{root}_goal_rgb"], np.zeros(4), device)["visual"][0, 0]
            plans = bank[f"{key}_plans"]
            true_rgb_cost, distance = actual[f"{key}_real_cost"], bank[f"{key}_distance"]
            costs = {x: [] for x in ("zero", "causal", "privileged")}
            mses = {x: [] for x in costs}
            state_inputs = {x: [] for x in costs}
            images10 = rgb10[group*48:(group+1)*48]
            with torch.no_grad():
                visual = torch.from_numpy(np.ascontiguousarray(images10.transpose(0, 3, 1, 2))).float()[:, None].to(device)
                actual_z = model.encode({"visual": visual, "proprio": torch.zeros(48, 1, 4, device=device)})["visual"][:, 0]
                for j in range(48):
                    idx = group * 48 + j
                    zero, _ = estimate_state(rgb5[idx], camera["metadata"], offset)
                    causal = causal_state(rgb4[idx], rgb5[idx], camera["metadata"], offset, .05)
                    privileged = zero.copy()
                    privileged[2:] = midstate[idx, 2:]
                    for label, state in (("zero", zero), ("causal", causal), ("privileged", privileged)):
                        state_inputs[label].append(state.copy())
                        context = encode_with_state(model, rgb5[idx], state, device)
                        prediction = predict(model, context, plans[j:j+1, 1:], mean, std)
                        cost, mse = metrics(prediction, actual_z[j:j+1], goal, mask)
                        costs[label].append(float(cost.item()))
                        mses[label].append(float(mse.item()))
            summaries = {}
            for label in costs:
                cost, mse, state = np.asarray(costs[label]), np.asarray(mses[label]), np.asarray(state_inputs[label])
                if not np.isfinite(cost).all() or not np.isfinite(mse).all() or not np.isfinite(state).all():
                    raise ValueError("nonfinite scoring output")
                arrays[f"{key}_{label}_cost"] = cost
                arrays[f"{key}_{label}_mse"] = mse
                arrays[f"{key}_{label}_input"] = state
                choice = int(np.argmin(cost))
                summaries[label] = {"masked_mse_mean": float(mse.mean()),
                                    "vs_real_rgb": pairwise_ranking(cost, true_rgb_cost),
                                    "oracle_choice": choice, "oracle_distance": float(distance[choice]),
                                    "velocity_l2_error_mean": float(np.linalg.norm(state[:, 2:] -
                                                                                    midstate[group*48:(group+1)*48, 2:],
                                                                                    axis=1).mean())}
            if not (np.allclose(costs["zero"], prior[f"{key}_refresh_cost"], rtol=1e-4, atol=1e-6)
                    and np.allclose(mses["zero"], prior[f"{key}_refresh_mse"], rtol=1e-4, atol=1e-6)):
                raise RuntimeError(f"zero velocity failed saved prediction replay for {key}")
            rows.append({"root": root, "method": method, "arms": summaries})
            print(f"{key}: zero, causal and privileged velocity predictions complete", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_cp5_velocity_same_action.v1",
              "previous_sha256": PREV_SHA, "source_sha256": SOURCE_SHA, "real_sha256": REAL_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "worker_sha256": sha256_file(WORKER),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "physical_high_level_steps": 3840, "cp4_cp5_cp10_exact_replay_scope": "cp5 and cp10 exact; cp4 newly observed",
              "cp5_rgb_encodes": 1152, "cp10_rgb_encodes": 384, "goal_rgb_encodes": 8,
              "prediction_samples": 1152, "predicted_packet_transitions": 1152,
              "wall_seconds": time.monotonic()-started,
              "scope": "exposed roots; true cp5 for each candidate is oracle; privileged arm diagnostic only; no P1",
              "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
