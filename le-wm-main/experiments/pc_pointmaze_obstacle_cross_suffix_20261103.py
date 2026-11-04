"""Physical cross-intervention of fixed obstacle prefixes and suffixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import (
    DATA, GOAL, OUTPUT as SOURCE, ROOTS,
)
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, sha256_file

SOURCE_SHA = "e5955123b5b88797a66e4c703a1c0f5b321a23c8fbead371dc029c8d44a5ff91"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_OBSTACLE_CROSS_SUFFIX_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_obstacle_cross_suffix_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ARMS = ("random", "spectral", "zero")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("source receipt changed")
    source = json.loads(SOURCE.read_text())
    if sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise RuntimeError("source arrays changed")
    if sha256_file(DATA / "states.pth") != source["states_sha256"]:
        raise RuntimeError("source states changed")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu").numpy()
    starts, sequences, ids = [], [], []
    started = time.monotonic()
    print("Execute all fixed obstacle prefix/suffix combinations.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved:
        for root in ROOTS:
            plans = {arm: saved[f"r{root}_{arm}_actions"].copy() for arm in ARMS[:2]}
            prefixes = {arm: plans[arm][:5] for arm in ARMS[:2]}
            suffixes = {arm: plans[arm][5:] for arm in ARMS[:2]}
            prefixes["zero"] = np.zeros((5, 2), np.float32)
            suffixes["zero"] = np.zeros((20, 2), np.float32)
            for prefix in ARMS:
                for suffix in ARMS:
                    seq = np.concatenate((prefixes[prefix], suffixes[suffix]))
                    if seq.shape != (25, 2) or not np.isfinite(seq).all() or (np.abs(seq) > 1).any():
                        raise ValueError("invalid action sequence")
                    ids.append((root, prefix, suffix))
                    starts.append(states[root, 40].astype(np.float64))
                    sequences.append(seq)
        starts, sequences = np.asarray(starts), np.asarray(sequences)
        cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
        cp25 = physical.replay_position_endpoints(starts, sequences)
        for i, (root, prefix, suffix) in enumerate(ids):
            if prefix == suffix and prefix in ARMS[:2]:
                index = 2 * ROOTS.index(root) + ARMS.index(prefix)
                np.testing.assert_array_equal(sequences[i], saved[f"r{root}_{prefix}_actions"])
                np.testing.assert_array_equal(cp25[i, :2], saved["endpoints_eval_only"][index, :2])
                with NativeRGBBridge(starts[i]) as env:
                    for action in sequences[i]:
                        rgb = env.step(action[None])
                    np.testing.assert_array_equal(rgb, saved[f"r{root}_{prefix}_frames"][-1])
    index = {key: i for i, key in enumerate(ids)}
    rows = []
    for i, (root, prefix, suffix) in enumerate(ids):
        zero_end = cp25[index[root, prefix, "zero"], :2]
        dist = float(np.linalg.norm(cp25[i, :2] - GOAL[:2]))
        rows.append({"root": root, "prefix": prefix, "suffix": suffix,
                     "initial_state": starts[i].tolist(), "cp5_state": cp5[i].tolist(),
                     "cp25_state": cp25[i].tolist(),
                     "initial_distance": float(np.linalg.norm(starts[i, :2] - GOAL[:2])),
                     "cp5_distance": float(np.linalg.norm(cp5[i, :2] - GOAL[:2])),
                     "cp25_distance": dist,
                     "suffix_displacement_minus_zero": (cp25[i, :2] - zero_end).tolist(),
                     "suffix_distance_minus_zero": dist - float(np.linalg.norm(zero_end - GOAL[:2]))})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, starts=starts, sequences=sequences, cp5=cp5, cp25=cp25)
    report = {"schema": "pc_pointmaze_obstacle_cross_suffix.v1", "rows": rows,
              "source_sha256": SOURCE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "physical_sequences": len(ids), "original_chains_replayed": 8,
              "new_model_inference": False, "wall_seconds": time.monotonic() - started,
              "scope": "exposed roots, outcomes and parent-optimized actions; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
