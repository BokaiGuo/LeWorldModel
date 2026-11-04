"""Same-action open vs actual-cp5 prediction on previously sealed banks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import pairwise_ranking
from experiments.a0_pointmaze_rgb_initial_real_successor_20261103 import (
    NOISE_SHA, NOISE_SOURCE, PLAN_SHA, PLAN_SOURCE, SOURCE, SOURCE_SHA,
)
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

REAL = ROOT / "results/a0_pointmaze_rgb_initial_real_successor_20261103_DEV01.json"
REAL_SHA = "3331a9c74efb93f163df4799828066f0237f3a767222f3aa454b80c68e63e2fa"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FIXED_BANK_CP5_REFRESH_PROTOCOL_20261103.md"
WORKER = ROOT / "experiments/pointmaze_fixed_bank_cp5_worker_20261103.py"
OUTPUT = ROOT / "results/a0_pointmaze_fixed_bank_cp5_refresh_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def predict(model, context, actions, mean, std):
    plans = torch.as_tensor(actions, dtype=torch.float32, device=mean.device)
    suffix = ((plans - mean) / std).repeat(1, 1, 5).permute(1, 0, 2)
    with torch.no_grad():
        return model.unroll(context, act_suffix=suffix)["visual"][-1]


def metrics(output, actual, goal, mask):
    if output.shape != actual.shape or output.shape[1:] != (1, 16, 16, 384):
        raise ValueError(f"visual mismatch: {output.shape} {actual.shape}")
    pred = output[:, 0, mask, :]
    target = goal[0, mask, :]
    return ((pred - target) ** 2).mean((1, 2)), ((pred - actual[:, 0, mask, :]) ** 2).mean((1, 2))


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (REAL, REAL_SHA), (PLAN_SOURCE, PLAN_SHA),
                           (NOISE_SOURCE, NOISE_SHA), (CAMERA, CAMERA_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"sealed input changed: {path}")
    source, real, plan, noise, camera = (json.loads(p.read_text()) for p in
                                          (SOURCE, REAL, PLAN_SOURCE, NOISE_SOURCE, CAMERA))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (REAL.with_suffix(".npz"), real["arrays_sha256"]),
                           (NOISE_SOURCE.with_suffix(".npz"), noise["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, noise["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", noise["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", noise["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"sealed data changed: {path}")
    offset = plan["body_offset_from_static_scene"]
    # Match the sealed renderer's float32 conversion before casting for MuJoCo.
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    begins, sequences = [], []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank:
        for row in source["rows"]:
            key = f"r{row['root']}_{row['method']}"
            plans = bank[f"{key}_plans"]
            if plans.shape != (48, 2, 2):
                raise ValueError("unexpected candidate bank")
            begins.extend([states[row["root"], 0]] * 48)
            sequences.extend(np.repeat(p, 5, axis=0) for p in plans)
    if len(begins) != 384:
        raise ValueError("unexpected number of plans")
    started = time.monotonic()
    print("Replaying every sealed plan, measuring cp5 and cp10 RGB.", flush=True)
    with tempfile.TemporaryDirectory(prefix="pc_cp5_refresh_") as tmp:
        request, xml_file, response = (Path(tmp) / x for x in ("request.npz", "scene.xml", "response.npz"))
        np.savez_compressed(request, sources=np.asarray(begins, np.float64), sequences=np.asarray(sequences, np.float64))
        xml = official_xml()
        xml_file.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(xml_file), str(response)],
                              capture_output=True, text=True, env=env, timeout=300)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"physical replay failed: {proc.stderr[-2000:]}")
        print(proc.stdout, flush=True)
        with np.load(response, allow_pickle=False) as result:
            mid, end, endpoints = (result[k].copy() for k in ("rgb5", "rgb10", "endpoints"))
    with np.load(REAL.with_suffix(".npz"), allow_pickle=False) as prior:
        if not np.array_equal(end, prior["rgb10"]) or not np.array_equal(endpoints, prior["legacy_endpoints"]):
            print(json.dumps({"rgb_mismatched_frames": int(np.any(end != prior["rgb10"], axis=(1, 2, 3)).sum()),
                              "rgb_max_difference": int(np.max(np.abs(end.astype(np.int16) - prior["rgb10"].astype(np.int16)))),
                              "endpoint_mismatched_rows": int(np.any(endpoints != prior["legacy_endpoints"], axis=1).sum()),
                              "endpoint_max_difference": float(np.max(np.abs(endpoints - prior["legacy_endpoints"])))}), flush=True)
            raise RuntimeError("cp10 image or endpoint differs from sealed replay")
    if mid.shape != (384, 224, 224, 3) or mid.dtype != np.uint8:
        raise ValueError("invalid intermediate images")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays = {"rgb5": mid, "endpoints": endpoints}
    rows = []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(NOISE_SOURCE.with_suffix(".npz"), allow_pickle=False) as inputs, \
            np.load(REAL.with_suffix(".npz"), allow_pickle=False) as real_arrays, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != plan["model_before"]:
            raise RuntimeError("model identity changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid action normalizer")
        for group, row in enumerate(source["rows"]):
            root, method = row["root"], row["method"]
            key = f"r{root}_{method}"
            rgb0, goal_rgb = inputs[f"r{root}_rgb0"], inputs[f"r{root}_goal_rgb"]
            mask = torch.as_tensor(inputs[f"r{root}_initial_mask"], dtype=torch.bool, device=device)
            state0, _ = estimate_state(rgb0, camera["metadata"], offset)
            ctx0 = encode_with_state(model, rgb0, state0, device)
            goal = encode_with_state(model, goal_rgb, np.zeros(4), device)["visual"][0, 0]
            plans = bank[f"{key}_plans"]
            with torch.no_grad():
                open_prediction = predict(model, ctx0, plans, mean, std)
                images = end[group * 48:(group + 1) * 48]
                visuals = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float()[:, None].to(device)
                actual = model.encode({"visual": visuals, "proprio": torch.zeros(48, 1, 4, device=device)})["visual"][:, 0]
                open_cost, open_mse = metrics(open_prediction, actual, goal, mask)
                refreshed_cost, refreshed_mse = [], []
                for j in range(48):
                    image = mid[group * 48 + j]
                    state, _ = estimate_state(image, camera["metadata"], offset)
                    context = encode_with_state(model, image, state, device)
                    prediction = predict(model, context, plans[j:j+1, 1:], mean, std)
                    cost, mse = metrics(prediction, actual[j:j+1], goal, mask)
                    refreshed_cost.append(float(cost.item()))
                    refreshed_mse.append(float(mse.item()))
            open_cost, open_mse = open_cost.cpu().numpy(), open_mse.cpu().numpy()
            refresh, refresh_mse = np.asarray(refreshed_cost), np.asarray(refreshed_mse)
            saved = bank[f"{key}_prediction"]
            if not np.allclose(open_cost, saved, rtol=1e-4, atol=1e-6):
                raise RuntimeError(f"open prediction does not match sealed bank: {key} "
                                   f"maxdiff={np.max(np.abs(open_cost-saved))}")
            real_cost, distance = real_arrays[f"{key}_real_cost"], bank[f"{key}_distance"]
            for label, value in (("open_cost", open_cost), ("open_mse", open_mse),
                                 ("refresh_cost", refresh), ("refresh_mse", refresh_mse)):
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite {key} {label}")
                arrays[f"{key}_{label}"] = value
            oi, ri = int(np.argmin(open_cost)), int(np.argmin(refresh))
            rows.append({"root": root, "method": method,
                         "open_vs_real_rgb": pairwise_ranking(open_cost, real_cost),
                         "refresh_vs_real_rgb": pairwise_ranking(refresh, real_cost),
                         "open_vs_physical": pairwise_ranking(open_cost, distance),
                         "refresh_vs_physical": pairwise_ranking(refresh, distance),
                         "open_masked_mse_mean": float(open_mse.mean()),
                         "refresh_masked_mse_mean": float(refresh_mse.mean()),
                         "open_choice": oi, "refresh_oracle_choice": ri,
                         "open_choice_distance": float(distance[oi]),
                         "refresh_oracle_choice_distance": float(distance[ri]),
                         "max_open_score_abs_error": float(np.max(np.abs(open_cost - saved)))})
            print(f"root {root} {method}: paired predictions completed", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_fixed_bank_cp5_refresh.v1",
              "source_sha256": SOURCE_SHA, "real_sha256": REAL_SHA, "camera_sha256": CAMERA_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "worker_sha256": sha256_file(WORKER),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "model_before": before,
              "model_after": after, "device": torch.cuda.get_device_name(device),
              "cp10_exact_image_and_endpoint_count": 384, "physical_high_level_steps": 3840,
              "cp5_rgb_encodes": 384, "cp10_rgb_encodes": 384, "initial_encodes": 8,
              "goal_encodes": 8, "open_prediction_samples": 384, "refresh_prediction_samples": 384,
              "predicted_packet_transitions": 1152, "wall_seconds": time.monotonic() - started,
              "scope": "exposed adaptive banks; per-candidate true cp5 is physical oracle; no P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
