"""Recompute comparable development receipts without new inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OUTPUT as NOISE, cem_four
from experiments.a1_pointmaze_proprio_score_feedback_cem_20261103 import (
    OUTPUT as EXTRA, SOURCE_SHA as NOISE_SHA,
)
from experiments.a1_pointmaze_proprio_spectral_cem_20261103 import OUTPUT as SPECTRAL
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, sha256_file

EXTRA_SHA = "a07f3c73c6cbcc215899d7bbaa6ab1b30ecadf43188b7a882e926c66abdac3df"
SPECTRAL_SHA = "b39bc04c3bceb2f54cc982b32d5517f5ae6be10d8f9ede598ac7496ab3ecda71"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PROPRIO_SPECTRAL_EXTRA_RANDOM_COMPARISON_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_proprio_spectral_extra_random_compare_20261103_DEV01.json"


def replay_cem(noise, recorded, optimizer, injected=None):
    iterator = iter(recorded)

    def scorer(proposals):
        row = next(iterator)
        np.testing.assert_array_equal(proposals, np.asarray(row["proposals"], np.float32))
        return row["scores"]

    return optimizer(noise, scorer) if injected is None else optimizer(noise, scorer, injected)


def main():
    if OUTPUT.exists():
        raise FileExistsError("refusing overwrite")
    sources = ((EXTRA, EXTRA_SHA), (SPECTRAL, SPECTRAL_SHA), (NOISE, NOISE_SHA))
    for path, expected in sources:
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    extra, spectral, noise = (json.loads(path.read_text()) for path, _ in sources)
    for path, data in ((EXTRA, extra), (SPECTRAL, spectral), (NOISE, noise)):
        if sha256_file(path.with_suffix(".npz")) != data["arrays_sha256"]:
            raise RuntimeError(f"arrays changed: {path}")
    if extra["model_before"] != spectral["model_before"] or any(
            data["model_before"] != data["model_after"] for data in (extra, spectral)):
        raise RuntimeError("model identity mismatch")
    expected_roots = list(range(1888, 1896))
    if any([row["root"] for row in data["rows"]] != expected_roots for data in (extra, spectral, noise)):
        raise RuntimeError("root identity mismatch")
    rows = []
    with np.load(EXTRA.with_suffix(".npz"), allow_pickle=False) as random_data, \
            np.load(SPECTRAL.with_suffix(".npz"), allow_pickle=False) as spectral_data, \
            np.load(NOISE.with_suffix(".npz"), allow_pickle=False) as noise_data:
        for i, root in enumerate(expected_roots):
            rrow, srow = extra["rows"][i], spectral["rows"][i]
            np.testing.assert_array_equal(rrow["goal_position_eval_only"], srow["goal_position_eval_only"])
            np.testing.assert_array_equal(rrow["goal_rgb_state"], srow["goal_rgb_state"])
            branches = (next(b for b in rrow["branches"] if b["arm"] == "proprio"),
                        next(b for b in srow["branches"] if b["arm"] == "spectral"))
            np.testing.assert_array_equal(random_data[f"r{root}_proprio_history"][0],
                                          spectral_data[f"r{root}_spectral_history"][0])
            conditions = []
            for j, branch in enumerate(branches):
                initial_noise = noise_data[f"r{root}_initial_noise"]
                injection = None if j == 0 else np.asarray(branch["injected"], np.float32)
                plan, cost, _ = replay_cem(initial_noise if j == 0 else initial_noise[:3],
                                           branch["initial_trace"], cem_four if j == 0 else cem, injection)
                tail, tail_cost, _ = replay_cem(noise_data[f"r{root}_tail_noise"], branch["tail_trace"], cem)
                np.testing.assert_array_equal(plan, np.asarray(branch["initial_plan"], np.float32))
                np.testing.assert_array_equal(tail, np.asarray(branch["tail"], np.float32))
                if (cost, tail_cost) != (branch["initial_cost"], branch["tail_cost"]):
                    raise RuntimeError("CEM score mismatch")
                seq = np.concatenate((np.repeat(plan[:1], 5, axis=0), np.repeat(tail[:1], 5, axis=0)))
                data, source_index = (random_data, 2*i+1) if j == 0 else (spectral_data, 2*i+1)
                np.testing.assert_array_equal(seq, data["sequences"][source_index])
                endpoint = data["endpoints_eval_only"][source_index]
                np.testing.assert_array_equal(endpoint, branch["endpoint_eval_only"])
                dist = float(np.linalg.norm(endpoint[:2]-np.asarray(rrow["goal_position_eval_only"])))
                if dist != branch["distance_eval_only"] or (dist < .5) != branch["strict_endpoint_success_eval_only"]:
                    raise RuntimeError("endpoint mismatch")
                if j == 0:
                    initial_time = branch["initial_encode_and_plan_seconds"]
                    structure_time, jvp, svd = 0., 0, 0
                else:
                    initial_time = branch["initial_encoding_seconds"] + branch["initial_plan_seconds"]
                    structure_time, jvp, svd = branch["structure_seconds"], branch["jvp_calls"], branch["svd_calls"]
                conditions.append({"arm": "random64" if j == 0 else "spectral48",
                                   "initial_samples": branch["prediction_counts"]["initial_samples"],
                                   "tail_samples": branch["prediction_counts"]["tail_samples"],
                                   "jvp_calls": jvp, "svd_calls": svd,
                                   "initial_encode_and_plan_seconds": initial_time,
                                   "structure_seconds": structure_time,
                                   "feedback_encode_and_plan_seconds": branch["feedback_encode_and_plan_seconds"],
                                   "distance_eval_only": dist, "strict_endpoint_success_eval_only": bool(dist < .5)})
            rows.append({"root": root, "conditions": conditions})
    report = {"schema": "a1_pointmaze_proprio_spectral_extra_random_compare.v1",
              "source_hashes": {str(path.relative_to(ROOT)): digest for path, digest in sources},
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "new_model_inference": False, "new_physical_executions": 0,
              "scope": "retrospective exposed development roots; unequal resources; no P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
