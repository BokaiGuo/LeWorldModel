"""Common-suffix physical intervention on sealed development prefixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_failure_real_rgb_metric_20261103 import ORIGINAL, ORIGINAL_SHA
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA
from experiments.a1_pointmaze_proprio_spectral_cem_20261103 import OUTPUT as SOURCE
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, ROOT, sha256_file

SOURCE_SHA = "b39bc04c3bceb2f54cc982b32d5517f5ae6be10d8f9ede598ac7496ab3ecda71"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PROPRIO_SPECTRAL_COMMON_SUFFIX_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_proprio_spectral_common_suffix_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    source, original, camera = (json.loads(path.read_text()) for path in (SOURCE, ORIGINAL, CAMERA))
    if sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise RuntimeError("source arrays changed")
    if sha256_file(DEFAULT_DATASET / "states.pth") != original["states_sha256"]:
        raise RuntimeError("states changed")
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    if [row["root"] for row in source["rows"]] != list(range(1888, 1896)):
        raise RuntimeError("unexpected roots")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    starts, sequences, descriptors = [], [], []
    for row in source["rows"]:
        root = row["root"]
        branches = {branch["arm"]: branch for branch in row["branches"]}
        prefixes = {name: np.asarray(branches[name]["initial_plan"][0], np.float32)
                    for name in ("random", "spectral")}
        suffixes = {name: np.asarray(branches[name]["tail"][0], np.float32)
                    for name in ("random", "spectral")}
        prefixes["zero"] = suffixes["zero"] = np.zeros(2, np.float32)
        for prefix in ("random", "spectral", "zero"):
            for suffix in ("zero", "random", "spectral"):
                sequence = np.concatenate((np.repeat(prefixes[prefix][None], 5, axis=0),
                                           np.repeat(suffixes[suffix][None], 5, axis=0)))
                if not np.isfinite(sequence).all() or np.any(np.abs(sequence) > 1):
                    raise ValueError("invalid action")
                starts.append(states[root, 0].astype(np.float64))
                sequences.append(sequence)
                descriptors.append((root, prefix, suffix))
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    if sequences.shape != (72, 10, 2):
        raise RuntimeError("invalid design size")
    started = time.monotonic()
    print("Executing 72 fixed common-suffix physical interventions.", flush=True)
    rgb10, legacy = replay_legacy(starts, sequences, xml)
    cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
    cp10 = physical.replay_position_endpoints(starts, sequences)
    if not np.array_equal(legacy[:, :2], cp10[:, :2]):
        raise RuntimeError("terminal position backend mismatch")
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved:
        for i, (root, prefix, suffix) in enumerate(descriptors):
            if prefix == suffix and prefix in ("random", "spectral"):
                np.testing.assert_array_equal(rgb10[i], saved[f"r{root}_{prefix}_rgb10"])
                np.testing.assert_array_equal(legacy[i], saved["endpoints_eval_only"][
                    2*(root-1888)+(0 if prefix == "random" else 1)])
    index = {identity: i for i, identity in enumerate(descriptors)}
    rows = []
    for i, (root, prefix, suffix) in enumerate(descriptors):
        goal = states[root, 10, :2].astype(np.float64)
        zero = cp10[index[root, prefix, "zero"], :2]
        end = cp10[i, :2]
        distance = float(np.linalg.norm(end-goal))
        rows.append({"root": root, "prefix": prefix, "suffix": suffix,
                     "initial_position": starts[i, :2].tolist(),
                     "goal_position_eval_only": goal.tolist(), "cp5_state": cp5[i].tolist(),
                     "cp10_state": cp10[i].tolist(),
                     "initial_distance": float(np.linalg.norm(starts[i, :2]-goal)),
                     "cp5_distance": float(np.linalg.norm(cp5[i, :2]-goal)),
                     "cp10_distance": distance, "strict_endpoint_success_eval_only": bool(distance < .5),
                     "suffix_displacement_minus_zero": (end-zero).tolist(),
                     "suffix_distance_change_minus_zero": distance-float(np.linalg.norm(zero-goal))})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, starts=starts, sequences=sequences, rgb10=rgb10,
                            legacy=legacy, cp5=cp5, cp10=cp10)
    report = {"schema": "a1_pointmaze_proprio_spectral_common_suffix.v1", "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "physical_sequences": len(sequences),
              "same_arm_original_rgb_and_endpoint_exact": 16, "cp10_backend_position_exact": 72,
              "wall_seconds": time.monotonic()-started, "rows": rows,
              "scope": "exposed development roots; common suffix intervention; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
