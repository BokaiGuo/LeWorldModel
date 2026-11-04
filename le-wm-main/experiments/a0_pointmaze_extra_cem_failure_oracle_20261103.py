"""Physical oracle on sealed initial and tail CEM candidates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import pairwise_ranking
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import cem_four
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, ROOT, sha256_file

SOURCE = ROOT / "results/a1_pointmaze_causal_extra_cem_budget_20261103_DEV01.json"
SOURCE_SHA = "9c8e00d5a5d1f1401108bbcfcd5dfda12658a9f7d49468189131a7db461bcb8c"
ORIGINAL = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
ORIGINAL_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_EXTRA_CEM_FAILURE_ORACLE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_extra_cem_failure_oracle_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")
ROOTS = (1889, 1893, 1895)


def replay_trace(trace, noise, algorithm, injected=None):
    entries = iter(trace)
    def scorer(proposals):
        item = next(entries)
        np.testing.assert_array_equal(proposals, np.asarray(item["proposals"], np.float32))
        return np.asarray(item["scores"])
    return algorithm(noise, scorer) if injected is None else algorithm(noise, scorer, injected)


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    if sha256_file(SOURCE) != SOURCE_SHA or sha256_file(ORIGINAL) != ORIGINAL_SHA:
        raise RuntimeError("sealed source changed")
    source, original = json.loads(SOURCE.read_text()), json.loads(ORIGINAL.read_text())
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"input changed: {path}")
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    groups, starts, actions, began = [], [], [], time.monotonic()
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved:
        for row in source["rows"]:
            root = row["root"]
            if root not in ROOTS:
                continue
            for branch in row["branches"]:
                method = branch["method"]
                initial_noise = saved[f"r{root}_initial_noise"]
                if method == "random_extra":
                    plan, cost0, _ = replay_trace(branch["initial_trace"], initial_noise, cem_four)
                else:
                    plan, cost0, _ = replay_trace(branch["initial_trace"], initial_noise[:3], cem, branch["injected"])
                tail, cost5, _ = replay_trace(branch["tail_trace"], saved[f"r{root}_tail_noise"], cem)
                if not (np.array_equal(plan, np.asarray(branch["initial_plan"], np.float32))
                        and np.array_equal(tail, np.asarray(branch["tail"], np.float32))
                        and cost0 == branch["initial_cost"] and cost5 == branch["tail_cost"]):
                    raise RuntimeError("sealed CEM trace did not reproduce")
                for phase in ("initial", "tail"):
                    trace = branch[f"{phase}_trace"]
                    candidates = np.concatenate([np.asarray(x["proposals"], np.float32) for x in trace])
                    scores = np.concatenate([np.asarray(x["scores"], np.float64) for x in trace])
                    expected = (64 if method == "random_extra" else 48) if phase == "initial" else 48
                    if len(candidates) != expected:
                        raise ValueError("candidate bank size mismatch")
                    if phase == "initial":
                        sequences = [np.repeat(p, 5, axis=0) for p in candidates]
                        chosen = plan
                    else:
                        sequences = [np.concatenate((np.repeat(plan[:1], 5, axis=0), np.repeat(p, 5, axis=0)))
                                     for p in candidates]
                        chosen = tail
                    choice = int(np.argmin(scores))
                    if not np.array_equal(candidates[choice], chosen):
                        raise RuntimeError("chosen candidate does not match score argmin")
                    starts.extend([states[root, 0]] * expected)
                    actions.extend(sequences)
                    groups.append({"root": root, "method": method, "phase": phase, "branch": branch,
                                   "candidates": candidates, "scores": scores, "choice": choice, "n": expected})
    if len(groups) != 12 or len(actions) != 624:
        raise RuntimeError("incomplete selected groups")
    print("Replaying 624 saved plans and fixed-prefix tails; no outcome-selected actions.", flush=True)
    endpoints = physical.replay_position_endpoints(np.asarray(starts, np.float64), np.asarray(actions, np.float64))
    rows, arrays, offset = [], {}, 0
    for group in groups:
        root, method, phase, n = (group[x] for x in ("root", "method", "phase", "n"))
        end = endpoints[offset:offset+n]
        offset += n
        target = states[root, 10, :2].astype(np.float64)
        distances = np.linalg.norm(end[:, :2] - target, axis=1)
        choice = group["choice"]
        if phase == "tail" and not np.array_equal(end[choice, :2], np.asarray(group["branch"]["endpoint_eval_only"])[:2]):
            raise RuntimeError("actual selected endpoint does not match sealed execution")
        oracle = int(np.argmin(distances))
        key = f"r{root}_{method}_{phase}"
        arrays[f"{key}_candidates"], arrays[f"{key}_scores"] = group["candidates"], group["scores"]
        arrays[f"{key}_endpoints"], arrays[f"{key}_distances"] = end, distances
        rows.append({"root": root, "method": method, "phase": phase, "n": n,
                     "chosen_index": choice, "chosen_distance": float(distances[choice]),
                     "oracle_index": oracle, "oracle_distance": float(distances[oracle]),
                     "successful_candidates": int(np.sum(distances < .5)),
                     "prediction_vs_physical": pairwise_ranking(group["scores"], distances)})
        print(f"{key}: chosen={distances[choice]:.4f}, oracle={distances[oracle]:.4f}", flush=True)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_extra_cem_failure_oracle.v1", "source_sha256": SOURCE_SHA,
              "original_sha256": ORIGINAL_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "physical_backend": physical.position_backend_receipt(), "physical_sequences": len(actions),
              "wall_seconds": time.monotonic()-began,
              "scope": "post-outcome selected development failures; physical oracle, no P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
