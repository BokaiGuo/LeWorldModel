"""Post-hoc task feasibility and selected-trajectory collision audit, never an initializer."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_successor_response_composition_e0_dev_20261103_codex as experiment

OUTPUT = experiment.PREFIX.with_name(experiment.PREFIX.name + "_feasibility_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(experiment.PLANS.read_text())
    if source["dependencies"] != experiment.hashes():
        raise ValueError("plan dependency mismatch")
    witnesses = []
    for index, task in enumerate(experiment.wall.tasks()):
        feasible = experiment.wall.feasibility(task)
        for witness in feasible:
            if not witness["within_horizon"] or witness["steps_needed"] > 15:
                continue
            action = np.asarray(witness["actions"])[:15] / experiment.wall.HIGH
            distance = float(np.linalg.norm(experiment.endpoints(task, action[None])[0] - task["goal"]))
            witnesses.append({"root": index, "side": witness["height"],
                              "steps": witness["steps_needed"], "terminal_distance": distance})
    collisions = []
    for row in source["rows"]:
        actions = np.asarray(row["selected"]["chain"])
        padded = np.pad(actions, ((0, 1), (0, 0))) * experiment.wall.HIGH
        trace, blocked = experiment.wall.rollout(row["task"]["start"], padded[None])
        distance = float(np.linalg.norm(trace[0, 15, :2] - row["task"]["goal"]))
        np.testing.assert_allclose(distance, row["selected"]["score"], atol=1e-12, rtol=0)
        collisions.append({"root": row["root"], "budget": row["budget"], "arm": row["arm"],
                           "blocked_steps": int(blocked[0, :15].sum()), "terminal_distance": distance})
    with OUTPUT.open("x") as file:
        json.dump({"plans_sha256": experiment.sha(experiment.PLANS),
                   "diagnostic_sha256": experiment.sha(Path(__file__)),
                   "scope": "post-hoc privileged feasibility and collision audit only",
                   "witnesses": witnesses, "selected": collisions}, file, indent=2, allow_nan=False)
    print(f"audited {len(witnesses)} witness replays and {len(collisions)} selected chains")


if __name__ == "__main__":
    main()
