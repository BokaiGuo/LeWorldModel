"""Separate frozen-model and true-RGB goal scoring on the same sealed banks."""
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
from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import (
    NOISE_SHA, NOISE_SOURCE, OUTPUT as SOURCE, SOURCE as PLAN_SOURCE, SOURCE_SHA as PLAN_SHA, pairwise_ranking,
)
from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import task_mask
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "44f3b133ca6332b9fcb831c6accd50052c239e38a9ef1c5e7ce4657debda2e76"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_INITIAL_REAL_SUCCESSOR_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_rgb_initial_real_successor_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
WORKER = ROOT / "experiments/pointmaze_candidate_rgb_replay_worker_20261103.py"


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (PLAN_SOURCE, PLAN_SHA), (NOISE_SOURCE, NOISE_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"sealed source changed: {path}")
    source, plan_doc, noise_doc = (json.loads(p.read_text()) for p in (SOURCE, PLAN_SOURCE, NOISE_SOURCE))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (NOISE_SOURCE.with_suffix(".npz"), noise_doc["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, noise_doc["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", noise_doc["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", noise_doc["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    begins, sequences, expected_endpoints = [], [], []
    start = time.monotonic()
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank:
        for row in source["rows"]:
            key = f"r{row['root']}_{row['method']}"
            plans = bank[f"{key}_plans"]
            if plans.shape != (48, 2, 2):
                raise ValueError("unexpected candidate shape")
            begins.extend([states[row["root"], 0]] * 48)
            sequences.extend(np.repeat(plan, 5, axis=0) for plan in plans)
            expected_endpoints.extend(bank[f"{key}_endpoints"])
    print("Rendering all 384 sealed candidate terminal RGBs before any analysis.", flush=True)
    with tempfile.TemporaryDirectory(prefix="pc_real_successor_") as tmp:
        directory = Path(tmp)
        request, xml_path, response = (directory / x for x in ("request.npz", "scene.xml", "response.npz"))
        np.savez_compressed(request, sources=np.asarray(begins, np.float64), sequences=np.asarray(sequences, np.float64))
        xml = official_xml()
        xml_path.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(xml_path), str(response)],
                              capture_output=True, text=True, env=env, timeout=300)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"legacy rendering failed: {proc.stderr[-1500:]}")
        print(proc.stdout, flush=True)
        with np.load(response, allow_pickle=False) as payload:
            images, endpoints = payload["rgb10"].copy(), payload["endpoints"].copy()
    expected = np.asarray(expected_endpoints, np.float64)
    if images.shape != (384, 224, 224, 3) or images.dtype != np.uint8 or not np.array_equal(endpoints[:, :2], expected[:, :2]):
        raise RuntimeError("render replay position or RGB shape mismatch")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, output_arrays = [], {"rgb10": images, "legacy_endpoints": endpoints}
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(NOISE_SOURCE.with_suffix(".npz"), allow_pickle=False) as inputs, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != plan_doc["model_before"]:
            raise RuntimeError("model identity changed")
        for i, row in enumerate(source["rows"]):
            root, method = row["root"], row["method"]
            key = f"r{root}_{method}"
            rgb0, rgb_goal = inputs[f"r{root}_rgb0"], inputs[f"r{root}_goal_rgb"]
            mask, _ = task_mask(rgb0, rgb_goal)
            mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
            goal = encode_with_state(model, rgb_goal, np.zeros(4), device)["visual"][0, 0]
            values = []
            with torch.no_grad():
                for batch in np.array_split(images[i*48:(i+1)*48], 3):
                    visual = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).float()[:, None].to(device)
                    observed = {"visual": visual, "proprio": torch.zeros(len(batch), 1, 4, device=device)}
                    z = model.encode(observed)["visual"][:, 0]
                    errors = ((z - goal) ** 2).mean(dim=(1, 4))
                    values.extend(errors[:, mask_t].mean(dim=1).cpu().tolist())
            real = np.asarray(values, np.float64)
            predicted, distance = bank[f"{key}_prediction"], bank[f"{key}_distance"]
            if real.shape != (48,) or not np.isfinite(real).all():
                raise RuntimeError("invalid real-successor scores")
            output_arrays[f"{key}_real_cost"] = real
            rows.append({"root": root, "method": method, "actual_rgb_score": real.tolist(),
                         "model_selected_index": int(np.argmin(predicted)),
                         "actual_rgb_selected_index": int(np.argmin(real)),
                         "physical_oracle_index": int(np.argmin(distance)),
                         "model_choice_distance": float(distance[np.argmin(predicted)]),
                         "real_rgb_choice_distance": float(distance[np.argmin(real)]),
                         "physical_oracle_distance": float(np.min(distance)),
                         "model_vs_real_rgb": pairwise_ranking(predicted, real),
                         "real_rgb_vs_physical": pairwise_ranking(real, distance),
                         "model_vs_physical": pairwise_ranking(predicted, distance)})
            print(f"root {root}: {method} actual RGB scores encoded", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **output_arrays)
    report = {"schema": "a0_pointmaze_rgb_initial_real_successor.v1", "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "worker_sha256": sha256_file(WORKER), "arrays_sha256": sha256_file(ARRAYS),
              "legacy_runtime_sha256": sha256_file(RUNTIME / "lib/libmujoco.so.2.1.2"),
              "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "position_exact_match_count": int(np.sum(np.all(endpoints[:, :2] == expected[:, :2], axis=1))),
              "wall_seconds": time.monotonic() - start,
              "scope": "post-outcome adaptive candidate banks on exposed roots; no P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
