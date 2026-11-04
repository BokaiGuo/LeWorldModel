"""Privileged post-outcome suffix representability, never a planning arm."""
import json
from pathlib import Path

import numpy as np

from . import pc_wall_retention_side_20261103 as experiment

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_WALL_RETENTION_COVERAGE_AUDIT_SIDE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_wall_retention_coverage_audit_side_20261103.json"


def witness(task, prefix):
    prefix = np.asarray(prefix)
    padded = np.concatenate((prefix, np.zeros((10, 2))))
    trace, _ = experiment.base.rollout(task["start"], padded[None])
    state = trace[0, 6]
    difference = np.asarray(task["goal"]) - state[:2]
    distance = np.linalg.norm(difference)
    speed = float(distance / 8)
    if speed > .3:
        return {"constructed": False, "reason": "straight witness exceeds speed bound",
                "prefix_state": state.tolist(), "successful_witness": False}
    turn = float((np.arctan2(difference[1], difference[0]) - state[2] + np.pi) % (2 * np.pi) - np.pi)
    suffix = np.array([[0., turn / 2]] * 2 + [[speed, 0.]] * 8)
    actions = np.concatenate((prefix, suffix))
    trace, blocked = experiment.base.rollout(task["start"], actions[None])
    final_distance = float(np.linalg.norm(trace[0, -1, :2] - task["goal"]))
    return {"constructed": True, "prefix_state": state.tolist(), "actions": actions.tolist(),
            "trajectory": trace[0].tolist(), "blocked": blocked[0].tolist(),
            "terminal_distance": final_distance, "successful_witness": final_distance < .05}


def run():
    source = json.loads(experiment.OUTPUT.read_text())
    for path, expected in source["dependencies"].items():
        if experiment.base.digest(Path(path)) != expected:
            raise ValueError(f"source changed: {path}")
    rows = []
    for row in source["rows"]:
        witnesses = [witness(row["task"], prefix) for prefix in row["bank"]]
        indices = [i for i, item in enumerate(witnesses) if item["successful_witness"]]
        rows.append({"index": row["index"], "witnesses": witnesses, "witness_indices": indices,
                     "retained_witness_indices": {name: [i for i in kept if i in indices]
                                                 for name, kept in row["screen"]["retained"].items()},
                     "cem_found_witness_indices": {
                         str(b["cap"]): [i for i in indices if i in b["successful_indices"]]
                         for b in row["budgets"]}})
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = [PROTOCOL, Path(__file__), experiment.OUTPUT, Path(experiment.__file__),
             Path(experiment.base.__file__), Path(experiment.base.physics.__file__)]
    hashes = {str(p): experiment.base.digest(p) for p in paths}
    rows = run()
    if hashes != {str(p): experiment.base.digest(p) for p in paths}:
        raise RuntimeError("source changed during diagnostic")
    with OUTPUT.open("x") as handle:
        json.dump({"scope": "post-outcome privileged representability only, not method performance",
                   "dependencies": hashes, "rows": rows, "P1_admission": False},
                  handle, indent=2, allow_nan=False)
    for row in rows:
        print(row["index"], row["witness_indices"], row["retained_witness_indices"],
              row["cem_found_witness_indices"], flush=True)


if __name__ == "__main__":
    main()
