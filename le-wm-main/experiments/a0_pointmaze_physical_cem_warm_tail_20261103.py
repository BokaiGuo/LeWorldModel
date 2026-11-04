"""Retain old terminal plan as one CEM proposal on privileged physical MPC."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.a0_pointmaze_physical_cem_mpc_20261103 import sequences
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, sha256_file

SOURCE = ROOT / "results/a0_pointmaze_physical_cem_mpc_20261103_DEV01.json"
SOURCE_SHA = "7fed2f87bfce3119a28a84ee895a408cdebe3f052a69850919e43a800332cd51"
ORIGINAL = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
ORIGINAL_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PHYSICAL_CEM_WARM_TAIL_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_physical_cem_warm_tail_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def warm_cem(noise, retained, scoring):
    noise = np.asarray(noise, np.float32)
    retained = np.asarray(retained, np.float32)
    if noise.shape != (3, 16, 1, 2) or retained.shape != (1, 2):
        raise ValueError("invalid warm CEM inputs")
    if not np.isfinite(noise).all() or not np.isfinite(retained).all() or np.any(np.abs(retained) > 1):
        raise ValueError("invalid values")
    mean, std = np.zeros((1, 2), np.float32), np.full((1, 2), 0.5, np.float32)
    best, best_cost, trace = None, float("inf"), []
    for i in range(3):
        proposals = np.clip(mean + std * noise[i], -1, 1)
        if i == 0:
            proposals[0] = retained
        scores = np.asarray(scoring(proposals), np.float64)
        if scores.shape != (16,) or not np.isfinite(scores).all():
            raise ValueError("invalid scoring result")
        order = np.argsort(scores, kind="stable")
        if scores[order[0]] < best_cost:
            best, best_cost = proposals[order[0]].copy(), float(scores[order[0]])
        elite = proposals[order[:4]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .1)
        trace.append({"proposals": proposals.tolist(), "scores": scores.tolist(),
                      "elite_indices": order[:4].tolist(), "mean": mean.tolist(), "std": std.tolist()})
    return best, best_cost, trace


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (ORIGINAL, ORIGINAL_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError("source changed")
    source, original = json.loads(SOURCE.read_text()), json.loads(ORIGINAL.read_text())
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (ORIGINAL.with_suffix(".npz"), original["arrays_sha256"])):
        if sha256_file(path) != digest:
            raise RuntimeError("source array changed")
    src_arrays = np.load(ORIGINAL.with_suffix(".npz"), allow_pickle=False)
    arrays, rows, started = {}, [], time.monotonic()
    print("Testing old-tail retention under fixed physical CEM budget; privileged development only.", flush=True)
    for row, orig in zip(source["rows"], original["rows"]):
        root = row["root"]
        if root != orig["root"]:
            raise RuntimeError("root mismatch")
        initial = np.asarray(row["initial_state_privileged"])
        cp5 = np.asarray(row["cp5_state_privileged"])
        goal = np.asarray(row["goal_position_privileged"])
        plan = np.asarray(row["initial_plan"], dtype=np.float32)
        calls = {"candidate_rollouts": 0, "candidate_high_level_steps": 0}

        def scoring(proposals):
            seq = sequences(proposals)
            states = physical.replay_position_endpoints(np.repeat(cp5[None], len(seq), axis=0), seq)
            calls["candidate_rollouts"] += len(seq)
            calls["candidate_high_level_steps"] += len(seq) * seq.shape[1]
            return np.linalg.norm(states[:, :2] - goal, axis=1)

        chosen, cost, trace = warm_cem(src_arrays[f"r{root}_tail_noise"], plan[1:2], scoring)
        retained_cost = trace[0]["scores"][0]
        if cost > retained_cost + 1e-12:
            raise RuntimeError("retention failed")
        prefix, tail = sequences(plan[None, :1])[0], sequences(chosen[None])[0]
        split = physical.replay_position_endpoints(cp5[None], tail[None])[0]
        full = physical.replay_position_endpoints(initial[None], np.concatenate((prefix, tail))[None])[0]
        with NativeRGBBridge(initial) as env:
            if not np.array_equal(env.initial_rgb, src_arrays[f"r{root}_rgb0"]):
                raise RuntimeError("initial RGB mismatch")
            rgb5 = env.step(prefix)
            if image_hash(rgb5) != row["rgb5_sha256"]:
                raise RuntimeError("prefix RGB mismatch")
            rgb10 = env.step(tail)
        arrays[f"r{root}_rgb10"] = rgb10
        rows.append({"root": root, "retained_cost": retained_cost, "chosen_tail": chosen.tolist(),
                     "chosen_cost": cost, "original_unwarmed_cost": row["tail_cost"],
                     "full_endpoint": full.tolist(), "split_endpoint": split.tolist(),
                     "split_full_max_abs_difference": float(np.max(np.abs(split-full))),
                     "distance_eval_only": float(np.linalg.norm(full[:2]-goal)),
                     "trace": trace, "candidate_cost": calls, "physical_execution_steps": 10,
                     "rgb10_sha256": image_hash(rgb10)})
        print(f"root {root}: 48 retained-tail candidates and native execution completed", flush=True)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_physical_cem_warm_tail.v1", "source_sha256": SOURCE_SHA,
              "original_sha256": ORIGINAL_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "physical_backend": physical.position_backend_receipt(), "wall_seconds": time.monotonic()-started,
              "scope": "post-outcome privileged physical development; not RGB or independent P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
