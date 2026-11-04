"""Replay every sealed A1 candidate and decompose its ranking errors."""
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
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE = ROOT / "results/a1_pointmaze_native_spectral_20261102_DEV01.json"
SOURCE_SHA = "2c61566337ae9915807e724913143e067a46ddfa519431de804ee21b452eed4a"
OUTPUT = ROOT / "results/a1_pointmaze_spectral_rank_decomposition_20261102_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SPECTRAL_RANK_DECOMPOSITION_PROTOCOL_20261102.md"


def pair_margins(predicted, encoded, physical_distance):
    values = [np.asarray(item, dtype=np.float64) for item in (predicted, encoded, physical_distance)]
    if any(item.shape != (5,) or not np.isfinite(item).all() for item in values):
        raise ValueError("expected three finite five-candidate vectors")
    pairs = []
    for i, j in itertools.combinations(range(5), 2):
        margins = [float(item[i] - item[j]) for item in values]
        signs = [int(np.sign(margin)) for margin in margins]
        pairs.append({"i": i, "j": j, "predicted_margin": margins[0],
                      "encoded_margin": margins[1], "physical_margin": margins[2],
                      "predicted_encoded_reverse": signs[0] * signs[1] == -1,
                      "encoded_physical_reverse": signs[1] * signs[2] == -1,
                      "predicted_encoded_tie": 0 in signs[:2],
                      "encoded_physical_tie": 0 in signs[1:]})
    return pairs


def main():
    if OUTPUT.exists() or sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("output exists or sealed source changed")
    old = json.loads(SOURCE.read_text())
    if (old["states_sha256"] != sha256_file(DEFAULT_DATASET / "states.pth")
            or old["checkpoint_sha256"] != sha256_file(DEFAULT_CHECKPOINT)):
        raise RuntimeError("source states or checkpoint changed")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    protocol_sha, started = sha256_file(PROTOCOL), time.monotonic()
    rows = []
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != old["model_before"]:
            raise RuntimeError("frozen model identity differs from source")
        for source_row in old["rows"]:
            root = int(source_row["root"])
            image_path = args.dataset_root / "obses" / f"episode_{root:03d}.pth"
            if sha256_file(image_path) != source_row["archive_sha256"]:
                raise RuntimeError("goal image archive changed")
            images = torch.load(image_path, map_location="cpu", weights_only=True).numpy()
            goal = goal_latent(encode(model, images[10], device))
            initial_state = dataset.states[root, 0].cpu().numpy().astype(np.float64)
            branches = []
            for previous in source_row["branches"]:
                bank = np.asarray(previous["candidate_actions"], dtype=np.float32)
                if bank.shape != (5, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
                    raise ValueError("source action bank malformed")
                rgb_hashes, encoded_costs = [], []
                for action in bank:
                    with NativeRGBBridge(initial_state) as env:
                        if hashlib.sha256(env.initial_rgb.tobytes()).hexdigest() != source_row["initial_rgb_sha256"]:
                            raise RuntimeError("initial RGB changed")
                        image = env.step(np.repeat(action[None], 5, axis=0))
                    rgb_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                    actual_latent = goal_latent(encode(model, image, device))
                    encoded_costs.append(float(torch.mean((actual_latent - goal) ** 2).cpu()))
                actions = np.repeat(bank[:, None, :], 5, axis=1)
                endpoints = physical.replay_position_endpoints(np.repeat(initial_state[None], 5, axis=0), actions)
                goal_position = dataset.states[root, 10, :2].cpu().numpy().astype(np.float64)
                distances = np.linalg.norm(endpoints[:, :2] - goal_position, axis=1)
                selected = int(previous["choice"])
                if rgb_hashes[selected] != previous["rgb5_sha256"]:
                    raise RuntimeError("selected candidate RGB differs from source")
                if not np.array_equal(endpoints[selected, :2], np.asarray(previous["position_eval_only"])):
                    raise RuntimeError("selected candidate physical endpoint differs from source")
                predicted = previous["model_costs"]
                branches.append({"method": previous["method"], "selected": selected,
                                 "actions": bank.tolist(), "predicted_costs": predicted,
                                 "actual_encoded_costs": encoded_costs, "physical_distances": distances.tolist(),
                                 "actual_latent_best": int(np.argmin(encoded_costs)),
                                 "physical_best": int(np.argmin(distances)),
                                 "rgb_sha256": rgb_hashes,
                                 "endpoints_eval_only": endpoints.tolist(),
                                 "pairs": pair_margins(predicted, encoded_costs, distances)})
            rows.append({"root": root, "goal_rgb_sha256": source_row["goal_rgb_sha256"], "branches": branches})
            print(f"completed 15 fixed candidate slots on root {root}", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("frozen model parameters changed")
    report = {"schema": "a1_pointmaze_spectral_rank_decomposition.v1",
              "scope": "post-outcome development diagnostics, shared roots, not independent P1",
              "source_sha256": SOURCE_SHA, "protocol_sha256": protocol_sha,
              "code_sha256": sha256_file(Path(__file__)),
              "bridge_sha256": sha256_file(ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py"),
              "model_before": before, "model_after": after,
              "normalizer": "full archive, training provenance unknown",
              "actual_rgb_encodes": 60, "goal_encodes": 4, "physical_action_slots": 60,
              "device": torch.cuda.get_device_name(device), "wall_seconds": time.monotonic() - started,
              "physical_backend": physical.position_backend_receipt(), "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
