"""Recompute action legality, exact costs, crossing and selection for each arm."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_block_chain_exploration_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_exact_block_chain_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record = json.loads(run.OUTPUT.read_text())
    if record["dependencies"] != {p: run.prior.sha(p) for p in record["dependencies"]}:
        raise ValueError("dependency drift")
    summary = {name: {"success": 0, "crossing_candidates": 0, "crossing_roots": 0,
                      "selected_crossing_roots": 0, "scores": []} for name in ("step", "block")}
    for expected_root, (row, task) in enumerate(zip(record["rows"], run.prior.panel.tasks(), strict=True)):
        if row["root"] != expected_root or row["task"] != task:
            raise ValueError("task mismatch")
        for name in summary:
            arm = row["arms"][name]
            chains = np.asarray(arm["actions"])
            if chains.shape != (384, 15, 2) or np.any(np.abs(chains) > 1) or arm["tokens"] != 1152:
                raise ValueError("invalid actions or budget")
            if not np.array_equal(chains[0], np.zeros((15, 2))):
                raise ValueError("zero incumbent missing")
            if name == "block":
                for k in range(3):
                    np.testing.assert_array_equal(chains[:, k*5:(k+1)*5],
                                                  np.repeat(chains[:, k*5:k*5+1], 5, axis=1))
            endpoints = run.prior.exact.endpoints(task, chains)
            costs = np.linalg.norm(endpoints - task["goal"], axis=1)
            np.testing.assert_allclose(costs, arm["costs"], atol=1e-12, rtol=0)
            best = int(np.argmin(costs))
            if arm["best"] != best:
                raise ValueError("selected chain was not scored minimum")
            padded = np.pad(chains, ((0, 0), (0, 1), (0, 0))) * run.prior.exact.wall.HIGH
            trajectory, _ = run.prior.exact.wall.rollout(task["start"], padded)
            crossed = (trajectory[:, 1:16, 0] > 0).any(axis=1)
            np.testing.assert_array_equal(crossed, arm["crossed"])
            entry = summary[name]
            entry["success"] += int(costs[best] < .05)
            entry["crossing_candidates"] += int(crossed.sum())
            entry["crossing_roots"] += int(crossed.any())
            entry["selected_crossing_roots"] += int(crossed[best])
            entry["scores"].append(float(costs[best]))
    for entry in summary.values():
        entry["mean_distance"] = float(np.mean(entry["scores"]))
    a, b = np.asarray(summary["step"]["scores"]), np.asarray(summary["block"]["scores"])
    paired = {"block_better": int(np.sum(b < a - 1e-12)), "step_better": int(np.sum(a < b - 1e-12)),
              "tie": int(np.sum(np.abs(a - b) <= 1e-12)),
              "block_only_success": int(np.sum((b < .05) & (a >= .05))),
              "step_only_success": int(np.sum((a < .05) & (b >= .05)))}
    payload = {"record_sha256": run.prior.sha(run.OUTPUT), "audit_sha256": run.prior.sha(Path(__file__)),
               "summary": summary, "paired": paired}
    with OUTPUT.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({"summary": summary, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
