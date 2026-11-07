"""Numerical stability of archived fixed-probe vector retention."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_vector_budget290_dev_20261107_codex as source
from experiments.pc_input_permissions import file_hash

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RETENTION_NUMERICAL_STRESS_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_retention_numerical_stress_20261107_codex.json"
EPSILON = .000466
N = 256


def select(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape != (8, 8) or not np.isfinite(matrix).all():
        raise ValueError("expected finite 8x8 scores")
    first = int(np.argmin(matrix.min(axis=1)))
    vectors = matrix - matrix.mean(axis=1, keepdims=True)
    distance = np.square(vectors - vectors[first]).sum(axis=1)
    distance[first] = -np.inf
    order = np.argsort(-distance, kind="stable")
    return first, int(order[0]), float(distance[order[0]] - distance[order[1]])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    if plan["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("source pool drift")
    rows = []
    with np.load(source.POOL, allow_pickle=False) as data:
        for root in range(10, 50):
            scores = data[f"r{root}_a_cost"]
            first, second, margin = select(scores)
            if [first, second] != next(r for r in plan["rows"] if r["root"] == root)["decisions"]["vector"]["retained"]:
                raise ValueError("original selection mismatch")
            f32 = select(scores.astype(np.float32))
            rng = np.random.default_rng(27110000 + root)
            perturb = np.stack([select(scores + rng.uniform(-EPSILON, EPSILON, scores.shape))[:2]
                                for _ in range(N)])
            rows.append({"root": root, "original": [first, second], "float32": list(f32[:2]),
                         "squared_distance_margin": margin,
                         "changed_first_draws": int(np.sum(perturb[:, 0] != first)),
                         "changed_second_draws": int(np.sum(perturb[:, 1] != second)),
                         "changed_pair_draws": int(np.sum(np.any(perturb != [first, second], axis=1)))})
    record = {"scope": "posthoc exposed development; synthetic perturbations, not independent tasks",
              "pool_sha256": file_hash(source.POOL), "plan_sha256": file_hash(source.PLAN),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "epsilon": EPSILON, "draws_per_root": N, "rows": rows,
              "summary": {"float32_changed_roots": sum(r["original"] != r["float32"] for r in rows),
                          "perturb_changed_first_roots": sum(r["changed_first_draws"] > 0 for r in rows),
                          "perturb_changed_second_roots": sum(r["changed_second_draws"] > 0 for r in rows),
                          "min_margin": min(r["squared_distance_margin"] for r in rows)}}
    with OUTPUT.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
