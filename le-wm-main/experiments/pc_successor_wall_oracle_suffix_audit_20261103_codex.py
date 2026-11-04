"""Privileged witness replay to diagnose suffix optimizer coverage, not planning."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_successor_wall_fixed_prefix_diag_20261103_codex as diagnostic

OUTPUT = diagnostic.PREFIX.with_name(diagnostic.PREFIX.name + "_oracle_suffix_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(diagnostic.PLANS.read_text())
    audit = json.loads(diagnostic.AUDIT.read_text())
    if data["dependencies"] != diagnostic.dependencies() or audit["runs_sha256"] != diagnostic.sha(diagnostic.PLANS):
        raise ValueError("diagnostic hash mismatch")
    rows = []
    for row in data["rows"]:
        options = [item for item in diagnostic.prior.wall.feasibility(row["task"])
                   if item["within_horizon"] and item["steps_needed"] <= 15]
        witness = min(options, key=lambda item: item["steps_needed"])
        action = np.asarray(witness["actions"])[:15] / diagnostic.prior.wall.HIGH
        for length in (5, 10):
            candidate = next(item for item in row["runs"] if item["name"] == f"witness_prefix_{length}")
            np.testing.assert_array_equal(candidate["prefix"], action[:length])
            terminal = float(np.linalg.norm(
                diagnostic.prior.endpoints(row["task"], action[None])[0] - row["task"]["goal"]))
            if terminal > 1e-10:
                raise ValueError("privileged feasible chain failed to replay")
            rows.append({"root": row["root"], "prefix_length": length, "witness_score": terminal,
                         "cem_192": candidate["checkpoints"]["192"]["score"],
                         "cem_384": candidate["checkpoints"]["384"]["score"]})
    with OUTPUT.open("x") as stream:
        json.dump({"runs_sha256": diagnostic.sha(diagnostic.PLANS),
                   "audit_sha256": diagnostic.sha(diagnostic.AUDIT),
                   "script_sha256": diagnostic.sha(Path(__file__)),
                   "scope": "privileged witness replay after diagnostic; never fed to planner",
                   "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} privileged fixed-prefix complete chains", flush=True)


if __name__ == "__main__":
    main()
