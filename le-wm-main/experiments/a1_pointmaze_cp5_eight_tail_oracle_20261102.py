"""Execute all fixed CP5 tails and decompose prediction/metric/coverage errors."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode, goal_latent
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.json"
SOURCE_SHA = "8f0bba93fa1f093cae4550da317d9b20a6cc018756e37402419f384eabdf6446"
HISTORY = ROOT / "results/a1_pointmaze_cp5_history_diagnostic_20261102_DEV01.json"
HISTORY_SHA = "040d627cd21c6337c244713c464ce91d2afa89d9923ca13d7f601e3d0e2b2cac"
ORIGINAL = ROOT / "results/a1_pointmaze_native_spectral_20261102_DEV01.json"
ORIGINAL_SHA = "2c61566337ae9915807e724913143e067a46ddfa519431de804ee21b452eed4a"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP5_EIGHT_TAIL_ORACLE_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_cp5_eight_tail_oracle_20261102_DEV01.json"
ARRAYS = ROOT / "results/a1_pointmaze_cp5_eight_tail_oracle_20261102_DEV01.npz"


def reversals(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != (8,) or b.shape != (8,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("expected finite eight-action score vectors")
    result = {"reversed": 0, "ties": 0}
    for i, j in itertools.combinations(range(8), 2):
        sign = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        result["reversed"] += int(sign < 0)
        result["ties"] += int(sign == 0)
    return result


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (HISTORY, HISTORY_SHA), (ORIGINAL, ORIGINAL_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"sealed input changed: {path}")
    source, history, original = (json.loads(path.read_text()) for path in (SOURCE, HISTORY, ORIGINAL))
    if (history["source_sha256"] != SOURCE_SHA or source["source_sha256"] != ORIGINAL_SHA
            or sha256_file(DEFAULT_DATASET / "states.pth") != original["states_sha256"]
            or sha256_file(DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]):
        raise RuntimeError("source ancestry, states or checkpoint differs")
    bank = np.asarray(source["bank"], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("invalid action bank")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, started = [], {}, time.monotonic()
    print("Executing every sealed CP5 tail; oracles are evaluation-only.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("model differs from source")
        for original_row, source_row, history_row in zip(original["rows"], source["rows"], history["rows"]):
            root = source_row["root"]
            if root != original_row["root"] or root != history_row["root"]:
                raise RuntimeError("root mismatch")
            archive = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(archive) != original_row["archive_sha256"]:
                raise RuntimeError("archive RGB mismatch")
            images = torch.load(archive, map_location="cpu", weights_only=True).numpy()
            goal = goal_latent(encode(model, images[10], device))
            initial = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            goal_pos = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
            branches = []
            for old, predicted in zip(source_row["branches"], history_row["branches"]):
                method = old["method"]
                if method != predicted["method"]:
                    raise RuntimeError("branch mismatch")
                prefix = np.asarray(old["prefix_action"], dtype=np.float32)
                if prefix.shape != (2,) or not np.isfinite(prefix).all() or np.any(np.abs(prefix) > 1):
                    raise ValueError("bad prefix")
                real_images, real_latents = [], []
                for index, tail in enumerate(bank):
                    with NativeRGBBridge(initial) as env:
                        if image_hash(env.initial_rgb) != original_row["initial_rgb_sha256"]:
                            raise RuntimeError("reset RGB mismatch")
                        rgb5 = env.step(np.repeat(prefix[None], 5, axis=0))
                        if image_hash(rgb5) != old["rgb5_sha256"]:
                            raise RuntimeError("prefix RGB mismatch")
                        rgb10 = env.step(np.repeat(tail[None], 5, axis=0))
                    if index == old["tail_index"] and image_hash(rgb10) != old["rgb10_sha256"]:
                        raise RuntimeError("selected RGB mismatch")
                    real_images.append(rgb10)
                    real_latents.append(goal_latent(encode(model, rgb10, device)).cpu().numpy())
                real_images, real_latents = np.stack(real_images), np.stack(real_latents)
                sequences = np.concatenate((np.repeat(np.repeat(prefix[None, None], 8, axis=0), 5, axis=1),
                                            np.repeat(bank[:, None], 5, axis=1)), axis=1)
                endpoints = physical.replay_position_endpoints(np.repeat(initial[None], 8, axis=0), sequences)
                selected = old["tail_index"]
                if not np.array_equal(endpoints[selected], np.asarray(old["endpoint_eval_only"])):
                    raise RuntimeError("selected physical endpoint mismatch")
                actual_cost = ((real_latents - goal.cpu().numpy()) ** 2).mean(axis=1)
                distances = np.linalg.norm(endpoints[:, :2] - goal_pos, axis=1)
                single = np.asarray(old["tail_costs"])
                history_cost = np.asarray(predicted["history_costs"])
                indices = {"single": selected, "history": int(np.argmin(history_cost)),
                           "real_latent_oracle": int(np.argmin(actual_cost)),
                           "physical_oracle": int(np.argmin(distances))}
                if indices["history"] != predicted["history_choice_not_executed"]:
                    raise RuntimeError("history selection mismatch")
                key = f"r{root}_{method}"
                arrays[f"{key}_rgb10"], arrays[f"{key}_latent10"] = real_images, real_latents
                arrays[f"{key}_endpoint"] = endpoints
                arrays[f"r{root}_goal_latent"] = goal.cpu().numpy()
                branches.append({"method": method, "indices": indices, "single_costs": single.tolist(),
                                 "history_costs": history_cost.tolist(), "actual_latent_costs": actual_cost.tolist(),
                                 "physical_distances": distances.tolist(),
                                 "distance_regret_vs_physical_oracle": {
                                     name: float(distances[index] - distances[indices["physical_oracle"]])
                                     for name, index in indices.items()},
                                 "single_vs_actual": reversals(single, actual_cost),
                                 "history_vs_actual": reversals(history_cost, actual_cost),
                                 "actual_vs_physical": reversals(actual_cost, distances),
                                 "rgb10_sha256": [image_hash(image) for image in real_images]})
                print(f"root {root} {method}: 8 native tails completed", flush=True)
            rows.append({"root": root, "goal_position_eval_only": goal_pos.tolist(), "branches": branches})
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a1_pointmaze_cp5_eight_tail_oracle.v1",
              "scope": "exposed-root post-outcome diagnostic, no independent P1",
              "source_sha256": SOURCE_SHA, "history_sha256": HISTORY_SHA, "original_sha256": ORIGINAL_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "physical_backend": physical.position_backend_receipt(),
              "wall_seconds": time.monotonic() - started, "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
