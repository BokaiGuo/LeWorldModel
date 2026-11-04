"""Fixed-action physical prefix/suffix intervention on exposed roots."""
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
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.pc_pointmaze_ap_spectral_cem_20261103 import OUTPUT as SOURCE
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, ROOT, sha256_file

SOURCE_SHA = "1737a2bf2f4501b537e2fdf74da10d80c5d87bd56b9fdda09d85c1ac73954e87"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AP_SPECTRAL_CROSS_SUFFIX_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_ap_spectral_cross_suffix_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    source, original, camera = (json.loads(p.read_text()) for p in (SOURCE, ORIGINAL, CAMERA))
    if sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise RuntimeError("source arrays changed")
    if sha256_file(DEFAULT_DATASET / "states.pth") != original["states_sha256"]:
        raise RuntimeError("state array changed")
    # Match PointMazeDataset's float32 conversion before extracting reset states.
    states = torch.load(DEFAULT_DATASET / "states.pth", weights_only=True, map_location="cpu").float().numpy()
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    starts, sequences, ids = [], [], []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved:
        for root in range(1888, 1896):
            actions = {arm: saved[f"r{root}_{arm}_actions"].copy() for arm in ("random", "spectral")}
            zero = np.zeros((5, 2), np.float32)
            prefixes = {"random": actions["random"][:5], "spectral": actions["spectral"][:5], "zero": zero}
            suffixes = {"random": actions["random"][5:], "spectral": actions["spectral"][5:], "zero": zero}
            for p in ("random", "spectral", "zero"):
                for s in ("random", "spectral", "zero"):
                    seq = np.concatenate((prefixes[p], suffixes[s]))
                    if seq.shape != (10, 2) or not np.isfinite(seq).all() or np.any(np.abs(seq) > 1):
                        raise ValueError("invalid fixed actions")
                    ids.append((root, p, s))
                    starts.append(states[root, 0].astype(np.float64))
                    sequences.append(seq)
        sequences, starts = np.asarray(sequences), np.asarray(starts)
        if sequences.shape != (72, 10, 2):
            raise RuntimeError("invalid intervention matrix")
        print("Execute all 72 fixed prefix/suffix interventions and validate original chains.", flush=True)
        started = time.monotonic()
        rgb, legacy = replay_legacy(starts, sequences, xml)
        cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
        cp10 = physical.replay_position_endpoints(starts, sequences)
        if not np.array_equal(legacy[:, :2], cp10[:, :2]):
            raise RuntimeError("position backend mismatch")
        for i, (root, prefix, suffix) in enumerate(ids):
            if prefix == suffix and prefix in ("random", "spectral"):
                np.testing.assert_array_equal(sequences[i], saved[f"r{root}_{prefix}_actions"])
                np.testing.assert_array_equal(rgb[i], saved[f"r{root}_{prefix}_frames"][-1])
                np.testing.assert_array_equal(legacy[i], saved["endpoints_eval_only"][
                    2*(root-1888) + (prefix == "spectral")])
    index = {key: i for i, key in enumerate(ids)}
    rows = []
    for i, (root, prefix, suffix) in enumerate(ids):
        goal = states[root, 10, :2].astype(np.float64)
        zero_end = cp10[index[root, prefix, "zero"], :2]
        dist = float(np.linalg.norm(cp10[i, :2] - goal))
        rows.append({"root": root, "prefix": prefix, "suffix": suffix,
                     "goal_eval_only": goal.tolist(), "initial_state": starts[i].tolist(),
                     "cp5_state": cp5[i].tolist(), "cp10_state": cp10[i].tolist(),
                     "initial_distance": float(np.linalg.norm(starts[i, :2]-goal)),
                     "cp5_distance": float(np.linalg.norm(cp5[i, :2]-goal)),
                     "cp10_distance": dist,
                     "suffix_displacement_minus_zero": (cp10[i, :2]-zero_end).tolist(),
                     "suffix_distance_change_minus_zero": dist-float(np.linalg.norm(zero_end-goal))})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, starts=starts, sequences=sequences, rgb10=rgb,
                            legacy=legacy, cp5=cp5, cp10=cp10)
    report = {"schema": "pc_pointmaze_ap_spectral_cross_suffix.v1", "rows": rows,
              "source_sha256": sha256_file(SOURCE), "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "physical_sequences": len(ids), "source_chains_reproduced": 16,
              "independent_endpoint_checks": len(ids), "new_model_inference": False,
              "wall_seconds": time.monotonic()-started,
              "scope": "outcome-exposed development intervention; not P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
