"""Post-hoc exact wall coverage and fixed-prefix optimization diagnostic."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as prior

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_EXACT_PREFIX_COVERAGE_CAUSE_DEV_PROTOCOL_20261106_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_prefix_coverage_cause_dev_20261106_codex.json"


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    dependencies = {str(path): prior.sha(path) for path in
                    (Path(__file__), PROTOCOL, prior.OUTPUT, Path(prior.__file__),
                     Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
                     Path(prior.exact.wall.physics.__file__))}
    source = json.loads(prior.OUTPUT.read_text())
    if source["dependencies"] != {path: prior.sha(path) for path in source["dependencies"]}:
        raise ValueError("source dependencies changed")
    rows = []
    print("Diagnostic only: fixed candidate prefixes vs privileged feasible-prefix tail search.", flush=True)
    for row in source["rows"]:
        root, task = row["root"], row["task"]
        pool = np.asarray(row["pool"])
        longer = []
        for i, chain in enumerate(pool):
            result = prior.optimize(task, chain[:5], chain[5:], 384, 261111000 + root * 8 + i)
            previous = row["refined"][i]
            np.testing.assert_array_equal(result["actions"][:36], previous["actions"])
            np.testing.assert_allclose(result["costs"][:36], previous["costs"], atol=1e-12, rtol=0)
            longer.append(result)
        witnesses = [x for x in prior.exact.wall.feasibility(task)
                     if x["within_horizon"] and x["steps_needed"] <= 15]
        if not witnesses:
            raise ValueError("no feasible witness")
        witness = witnesses[0]
        physical = np.asarray(witness["actions"])
        if np.max(np.abs(physical) - prior.exact.wall.HIGH) > 1e-12 or any(witness["blocked"][:15]):
            raise ValueError("illegal or blocked witness")
        normalized = physical[:15] / prior.exact.wall.HIGH
        distance = float(np.linalg.norm(prior.exact.endpoints(task, normalized[None])[0] - task["goal"]))
        if distance >= .05:
            raise ValueError("witness replay failed")
        witness_search = prior.optimize(task, normalized[:5], np.zeros((10, 2)), 384, 261112000 + root)
        if not np.array_equal(np.asarray(witness_search["actions"])[0, 5:], np.zeros((10, 2))):
            raise ValueError("witness tail leaked into initialization")
        rows.append({"root": root, "task": task, "pool_longer": longer,
                     "witness_side": witness["height"], "witness_steps": witness["steps_needed"],
                     "witness_chain": normalized.tolist(), "witness_score": distance,
                     "witness_blocked": witness["blocked"][:15],
                     "witness_prefix_search": witness_search})
        print(f"root {root}: 8 fixed prefixes and feasible witness audited", flush=True)
    if dependencies != {path: prior.sha(path) for path in dependencies}:
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as file:
        json.dump({"dependencies": dependencies, "scope": "post-hoc privileged exact diagnostic",
                   "rows": rows}, file, separators=(",", ":"), allow_nan=False)
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
