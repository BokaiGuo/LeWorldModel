"""Physical oracle decomposition on already sealed RGB-position CEM banks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a1_pointmaze_rgb_state_full_planning_20261103 import (
    OUTPUT as SOURCE, SOURCE as NOISE_SOURCE, SOURCE_SHA as NOISE_SHA,
)
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, ROOT, sha256_file

SOURCE_SHA = "4288b73e675e45526db46698a7abdcd68b619fc235e0a73bc20f91bead6d9b99"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_INITIAL_CANDIDATE_ORACLE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_rgb_initial_candidate_oracle_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def pairwise_ranking(prediction, actual):
    prediction, actual = np.asarray(prediction), np.asarray(actual)
    if prediction.ndim != 1 or prediction.shape != actual.shape or not np.isfinite(prediction).all() or not np.isfinite(actual).all():
        raise ValueError("invalid paired candidate scores")
    i, j = np.triu_indices(len(prediction), 1)
    delta_p, delta_a = prediction[i] - prediction[j], actual[i] - actual[j]
    eligible = (delta_p != 0) & (delta_a != 0)
    return {"comparable": int(eligible.sum()), "reversed": int(np.sum((delta_p * delta_a < 0) & eligible)),
            "prediction_ties": int(np.sum(delta_p == 0)), "outcome_ties": int(np.sum(delta_a == 0))}


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    if sha256_file(SOURCE) != SOURCE_SHA or sha256_file(NOISE_SOURCE) != NOISE_SHA:
        raise RuntimeError("sealed report changed")
    source, noise_doc = (json.loads(p.read_text()) for p in (SOURCE, NOISE_SOURCE))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (NOISE_SOURCE.with_suffix(".npz"), noise_doc["arrays_sha256"]),
                           (DEFAULT_DATASET / "states.pth", noise_doc["states_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    groups, starts, sequences, started = [], [], [], time.monotonic()
    with np.load(NOISE_SOURCE.with_suffix(".npz"), allow_pickle=False) as noise:
        for row in source["rows"]:
            root = row["root"]
            for branch in row["branches"]:
                if branch["policy"] != "rgb_position":
                    continue
                trace = branch["initial_trace"]
                proposals = np.asarray([step["proposals"] for step in trace], np.float32)
                scores = np.asarray([step["scores"] for step in trace], np.float64)
                if proposals.shape != (3, 16, 2, 2) or scores.shape != (3, 16):
                    raise ValueError("unexpected saved CEM trace")
                iterator = iter(trace)
                def replay(candidates):
                    step = next(iterator)
                    np.testing.assert_array_equal(candidates, np.asarray(step["proposals"], np.float32))
                    return np.asarray(step["scores"])
                selected, cost, _ = cem(noise[f"r{root}_initial_noise"], replay, branch["injected"])
                if not np.array_equal(selected, np.asarray(branch["initial_plan"], np.float32)) or cost != branch["initial_cost"]:
                    raise RuntimeError("CEM selection mismatch")
                flat = proposals.reshape(48, 2, 2)
                starts.extend([states[root, 0]] * len(flat))
                sequences.extend([np.repeat(plan, 5, axis=0) for plan in flat])
                groups.append((root, branch, flat, scores.reshape(-1)))
    if len(groups) != 8 or len(sequences) != 384:
        raise RuntimeError("unexpected candidate bank size")
    print("Replaying 384 sealed RGB-position CEM candidates; no selection from outcomes.", flush=True)
    endpoints = physical.replay_position_endpoints(np.asarray(starts, np.float64), np.asarray(sequences, np.float64))
    rows, arrays = [], {}
    for index, (root, branch, plans, prediction) in enumerate(groups):
        ends = endpoints[index * 48:(index + 1) * 48]
        goal = states[root, 10, :2].astype(np.float64)
        distance = np.linalg.norm(ends[:, :2] - goal, axis=1)
        chosen, oracle = int(np.argmin(prediction)), int(np.argmin(distance))
        if not np.array_equal(plans[chosen], np.asarray(branch["initial_plan"], np.float32)):
            raise RuntimeError("selected candidate identity mismatch")
        key = f"r{root}_{branch['method']}"
        arrays[f"{key}_plans"], arrays[f"{key}_prediction"] = plans, prediction
        arrays[f"{key}_endpoints"], arrays[f"{key}_distance"] = ends, distance
        rows.append({"root": root, "method": branch["method"], "goal_position_eval_only": goal.tolist(),
                     "selected_index": chosen, "selected_distance": float(distance[chosen]),
                     "selected_success": bool(distance[chosen] < .5), "oracle_index": oracle,
                     "oracle_distance": float(distance[oracle]), "oracle_success": bool(distance[oracle] < .5),
                     "candidate_success_count": int(np.sum(distance < .5)),
                     "selected_regret": float(distance[chosen] - distance[oracle]),
                     "prediction_physics_rank": pairwise_ranking(prediction, distance),
                     "injected_first_four_distances": distance[:4].tolist() if branch["method"] == "spectral_cem" else None})
        print(f"root {root}: {branch['method']} 48 candidates analyzed", flush=True)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_rgb_initial_candidate_oracle.v1", "source_sha256": SOURCE_SHA,
              "noise_source_sha256": NOISE_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "states_sha256": noise_doc["states_sha256"], "backend": physical.position_backend_receipt(),
              "scope": "post-outcome full open-loop candidate replay on exposed roots; no P1",
              "wall_seconds": time.monotonic() - started, "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
