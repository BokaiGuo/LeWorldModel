"""Post-hoc finite-difference scale and collision audit on sealed successor responses."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_successor_response_composition_e0_dev_20261103_codex as experiment

OUTPUT = experiment.PREFIX.with_name(experiment.PREFIX.name + "_fd_audit.json")


def jacobian(task, prefix, nominal, epsilon):
    directions = np.eye(10).reshape(10, 5, 2)
    positive = np.concatenate((np.broadcast_to(prefix, (10, 5, 2)), nominal + epsilon * directions), axis=1)
    negative = np.concatenate((np.broadcast_to(prefix, (10, 5, 2)), nominal - epsilon * directions), axis=1)
    return ((experiment.endpoints(task, positive) - experiment.endpoints(task, negative)) / (2 * epsilon)).T


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(experiment.PLANS.read_text())
    if source["dependencies"] != experiment.hashes():
        raise ValueError("sealed plan dependency mismatch")
    rows = []
    for row in source["rows"]:
        if row["arm"] != "successor_response" or row["budget"] != 960:
            continue
        for index, evidence in enumerate(row["successor_response"]):
            prefix = np.asarray(row["first_bank"][index]["prefix"])
            for j, item in enumerate(evidence):
                nominal = np.asarray(item["nominal"])
                reference = np.asarray(item["jacobian"])
                np.testing.assert_allclose(jacobian(row["task"], prefix, nominal, .05),
                                           reference, rtol=0, atol=1e-12)
                sensitivities = {}
                for epsilon in (.01, .1):
                    alternative = jacobian(row["task"], prefix, nominal, epsilon)
                    sensitivities[str(epsilon)] = float(
                        np.linalg.norm(alternative - reference) / max(np.linalg.norm(reference), 1e-12))
                actions = np.concatenate((prefix, nominal))[None]
                padded = np.pad(actions, ((0, 0), (0, 6), (0, 0))) * experiment.wall.HIGH
                _, blocked = experiment.wall.rollout(row["task"]["start"], padded)
                rows.append({"root": row["root"], "first": index, "nominal": j,
                             "reference_norm": float(np.linalg.norm(reference)),
                             "relative_difference": sensitivities,
                             "nominal_blocked_steps": int(blocked[0, :10].sum())})
    with OUTPUT.open("x") as file:
        json.dump({"plans_sha256": experiment.sha(experiment.PLANS),
                   "script_sha256": experiment.sha(Path(__file__)),
                   "scope": "post-hoc exposed-task finite-difference sensitivity; no selector changes",
                   "rows": rows}, file, indent=2, allow_nan=False)
    print(f"audited {len(rows)} successor response matrices", flush=True)


if __name__ == "__main__":
    main()
