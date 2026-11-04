"""Read-only inventory of fixed-action velocity/score cells; no neural inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_VELOCITY_SCORE_CROSSTAB_INVENTORY_20261103_CODEXEXCLUSIVE.md"
INPUTS = {
    "velocity_json": ("results/a0_pointmaze_cp5_velocity_same_action_20261103_DEV01.json",
                      "1a31884a8ab092a04f9d6437a043c54815d1cc0821f212d47bb96a14e8a0569a"),
    "velocity_npz": ("results/a0_pointmaze_cp5_velocity_same_action_20261103_DEV01.npz",
                     "5f72892e58ce6837dfc03088d73641d5e8f0a70d3021033b46daeb8d276b14e8"),
    "score_json": ("results/a0_pointmaze_predicted_proprio_goal_score_20261103_DEV01.json",
                   "a3339218d9f5dc86da737fd773e30f65eaeefbcaf5f6849791b30cf251690d68"),
    "score_npz": ("results/a0_pointmaze_predicted_proprio_goal_score_20261103_DEV01.npz",
                  "7715f7cb9265cad7c4eb875b045ce853c917fe5dd466c529a9226738eb2a136c"),
}


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inventory():
    paths = {}
    for key, (name, expected) in INPUTS.items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"source SHA mismatch: {name}")
        paths[key] = path
    velocity = json.loads(paths["velocity_json"].read_text())
    score = json.loads(paths["score_json"].read_text())
    if (velocity["arrays_sha256"] != INPUTS["velocity_npz"][1]
            or score["arrays_sha256"] != INPUTS["score_npz"][1]
            or velocity["model_before"] != velocity["model_after"]
            or score["model_before"] != score["model_after"]):
        raise ValueError("receipt mismatch")
    groups = [(r["root"], r["method"]) for r in velocity["rows"]]
    expected = [(root, method) for root in (1884, 1885, 1886, 1887)
                for method in ("random_cem", "spectral_cem")]
    if groups != expected:
        raise ValueError("velocity roster differs")
    records = []
    with np.load(paths["velocity_npz"], allow_pickle=False) as vel, \
            np.load(paths["score_npz"], allow_pickle=False) as prop:
        if vel["midstate_privileged"].shape != (384, 4) or vel["rgb4"].shape != (384, 224, 224, 3):
            raise ValueError("velocity input shape")
        for root, method in groups:
            key = f"r{root}_{method}"
            cells = {}
            for mode in ("zero", "causal", "privileged"):
                names = [f"{key}_{mode}_{field}" for field in ("cost", "mse", "input")]
                if any(name not in vel for name in names):
                    raise ValueError("missing saved visual source")
                if [vel[name].shape for name in names] != [(48,), (48,), (48, 4)]:
                    raise ValueError("visual source shape")
                if not all(np.isfinite(vel[name]).all() for name in names):
                    raise ValueError("nonfinite input")
                cells[mode] = {"visual": "saved", "proprio": "not_saved_for_this_bank",
                               "visual_array": names[0], "input_array": names[2]}
            records.append({"root": root, "method": method, "n": 48, "cells": cells})
        score_roots = sorted({row["root"] for row in score["rows"]})
        if set(score_roots) & {root for root, _ in groups}:
            raise ValueError("root overlap needs explicit binding review")
        for row in score["rows"]:
            key = f"r{row['root']}_{row['method']}_{row['phase']}"
            for field in ("visual", "proprio"):
                if f"{key}_{field}" not in prop or prop[f"{key}_{field}"].shape != (row["n"],):
                    raise ValueError("separate score array shape")
    return {"schema": "pc_velocity_score_inventory_v1", "protocol_sha256": sha(PROTOCOL),
            "script_sha256": sha(__file__), "sources": INPUTS, "banks": records,
            "velocity_roots": [1884, 1885, 1886, 1887], "separate_score_roots": score_roots,
            "fixed_candidates": 384, "available_visual_cells": 1152,
            "missing_proprio_cells": 1152, "cross_table_complete": False,
            "missing_reason": "proprio score and predicted proprio token were not saved for velocity banks",
            "can_join_separate_score_bank": False, "neural_inference": False, "physical_execution": False,
            "independent_outcome": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.parent != ROOT / "results" or not output.name.startswith("pc_pointmaze_velocity_score_inventory_"):
        raise ValueError("exclusive results output required")
    result = inventory()
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(output), "sha256": sha(output),
                      "cross_table_complete": result["cross_table_complete"]}))


if __name__ == "__main__":
    main()
