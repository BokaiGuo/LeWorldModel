"""Privileged position-backend CEM with a receding, fixed terminal clock."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, ROOT, sha256_file

SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PHYSICAL_CEM_MPC_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_physical_cem_mpc_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def sequences(proposals):
    values = np.asarray(proposals, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2 or values.shape[1] not in (1, 2):
        raise ValueError("invalid proposal shape")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError("invalid action bounds")
    return np.repeat(values, 5, axis=1)


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing to overwrite sealed result")
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("source JSON changed")
    source = json.loads(SOURCE.read_text())
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (DEFAULT_DATASET / "states.pth", source["states_sha256"])):
        if sha256_file(path) != digest:
            raise RuntimeError(f"source input changed: {path}")
    source_arrays = np.load(SOURCE.with_suffix(".npz"), allow_pickle=False)
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float()
    arrays, rows, started = {}, [], time.monotonic()
    print("Physical-CEM MPC upper-bound diagnostic: privileged states and goal, not RGB policy.", flush=True)
    for row in source["rows"]:
        root = row["root"]
        initial = states[root, 0].numpy().astype(np.float64)
        goal = states[root, 10, :2].numpy().astype(np.float64)
        initial_noise = source_arrays[f"r{root}_initial_noise"]
        tail_noise = source_arrays[f"r{root}_tail_noise"]
        count = {"candidate_rollouts": 0, "candidate_high_level_steps": 0}

        def scorer(state, proposals):
            seq = sequences(proposals)
            endpoints = physical.replay_position_endpoints(np.repeat(state[None], len(seq), axis=0), seq)
            count["candidate_rollouts"] += len(seq)
            count["candidate_high_level_steps"] += len(seq) * seq.shape[1]
            return np.linalg.norm(endpoints[:, :2] - goal, axis=1)

        plan, initial_cost, initial_trace = cem(initial_noise, lambda p: scorer(initial, p))
        prefix = sequences(plan[None, :1])[0]
        cp5 = physical.replay_position_endpoints(initial[None], prefix[None])[0]
        tail, tail_cost, tail_trace = cem(tail_noise, lambda p: scorer(cp5, p))
        suffix = sequences(tail[None])[0]
        split = physical.replay_position_endpoints(cp5[None], suffix[None])[0]
        full = physical.replay_position_endpoints(initial[None], np.concatenate((prefix, suffix))[None])[0]
        with NativeRGBBridge(initial) as env:
            if not np.array_equal(env.initial_rgb, source_arrays[f"r{root}_rgb0"]):
                raise RuntimeError("initial RGB mismatch")
            rgb5 = env.step(prefix)
            rgb10 = env.step(suffix)
        arrays[f"r{root}_rgb5"], arrays[f"r{root}_rgb10"] = rgb5, rgb10
        arrays[f"r{root}_cp5_state"], arrays[f"r{root}_full_state"] = cp5, full
        rows.append({"root": root, "initial_state_privileged": initial.tolist(), "goal_position_privileged": goal.tolist(),
                     "initial_plan": plan.tolist(), "initial_cost": initial_cost, "initial_trace": initial_trace,
                     "tail_plan": tail.tolist(), "tail_cost": tail_cost, "tail_trace": tail_trace,
                     "cp5_state_privileged": cp5.tolist(), "split_endpoint": split.tolist(),
                     "full_endpoint": full.tolist(), "split_full_max_abs_difference": float(np.max(np.abs(split-full))),
                     "distance_eval_only": float(np.linalg.norm(full[:2]-goal)),
                     "source_visual_distances": {b["method"]: b["distance_eval_only"] for b in row["branches"]},
                     "rgb5_sha256": image_hash(rgb5), "rgb10_sha256": image_hash(rgb10),
                     "candidate_cost": count, "physical_execution_steps": 10,
                     "verification_replay_steps": 20})
        print(f"root {root}: 96 privileged candidates and native RGB execution completed", flush=True)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_physical_cem_mpc.v1", "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "physical_backend": physical.position_backend_receipt(),
              "wall_seconds": time.monotonic() - started, "rows": rows,
              "scope": "privileged position-backend development oracle; not RGB or independent P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
