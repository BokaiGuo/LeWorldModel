"""Independently recompute saved score decisions and archive-action physics replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT

NAME = "a0_pointmaze_archive_bank_validate_20261102"
INPUT = ROOT / "results/a0_pointmaze_archive_bank_alignment_20261102_DEV01.json"
INPUT_SHA = "93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291"
DATA = ROOT / "data/point_maze_tail1800_1899"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.parent != ROOT / "results"
            or not args.output.name.startswith(NAME + "_") or args.output.exists()):
        raise ValueError("exclusive output")
    raw = INPUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError("input hash")
    source = json.loads(raw)
    # PointMazeDataset explicitly converts both archives to float32 on load.
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    actions = torch.load(DATA / "actions.pth", map_location="cpu", weights_only=True).float().numpy()
    rows = []
    for row in source["rows"]:
        root = row["root"]
        target = np.asarray(row["goal_position_evaluation_only"])
        ends = np.asarray(row["physical_terminal_positions"])
        costs = np.linalg.norm(ends - target, axis=-1)
        if not np.allclose(costs, row["physical_terminal_costs"], rtol=1e-12, atol=1e-12):
            raise ValueError("saved costs do not match positions")
        model_index = np.unravel_index(np.argmin(row["model_terminal_scores"]), (8, 8))
        oracle_index = np.unravel_index(np.argmin(costs), (8, 8))
        if list(model_index) != row["model_pair"] or list(oracle_index) != row["oracle_pair"]:
            raise ValueError("saved argmin mismatch")
        if not np.isclose(costs[model_index] - costs[oracle_index], row["physical_regret"], atol=1e-12):
            raise ValueError("saved regret mismatch")
        start = states[root, 0].astype(np.float64)
        actual_actions = actions[root, :10].astype(np.float64)
        if not np.array_equal(start, np.asarray(row["state_start_evaluation_only"])):
            raise ValueError("source start mismatch")
        predicted = physical.replay_position_endpoints(start[None], actual_actions[None])[0]
        observed = states[root, 10].astype(np.float64)
        rows.append({
            "root": root, "archive_action_replay_position": predicted[:2].tolist(),
            "archive_frame10_position": observed[:2].tolist(),
            "archive_action_replay_position_error": float(np.linalg.norm(predicted[:2] - observed[:2])),
            "model_pair": list(map(int, model_index)), "oracle_pair": list(map(int, oracle_index)),
            "physical_regret": float(costs[model_index] - costs[oracle_index]),
        })
    result = {
        "schema": NAME + ".v1", "source_sha256": INPUT_SHA,
        "scope": "archive recorded-action position replay; not complete simulator equivalence",
        "rows": rows,
        "max_archive_replay_position_error": max(r["archive_action_replay_position_error"] for r in rows),
        "scientific_admission": False,
    }
    output = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(output)
    print(json.dumps({"sha256": hashlib.sha256(output).hexdigest(), "rows": rows}))


if __name__ == "__main__":
    main()
