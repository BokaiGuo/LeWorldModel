"""Post-hoc fixed-prefix suffix allocation diagnostic on sealed development plans."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_same_policy_exact_mechanism_dev_20261103_codex as previous
from experiments.pc_budget_allocation_20261103_codex import optimize

PROTOCOL = previous.ROOT.parent / "docs/PC_EXACT_SUFFIX_FIXED_CANDIDATE_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = previous.PREFIX.with_name(previous.PREFIX.name + "_fixed_suffix_diag.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(previous.PLANS.read_text())
    if source["dependencies"] != previous.dependencies():
        raise ValueError("sealed plan dependency mismatch")
    hashes = {str(p): previous.sha(p) for p in
              (PROTOCOL, Path(__file__), previous.PLANS, Path(optimize.__code__.co_filename))}
    rows = []
    for root in source["rows"]:
        goal = np.asarray(root["goal"])
        for program in root["methods"]["svd"]["decision"]["programs"]:
            prefix = np.asarray(program["chain"])[:5]
            seed = 2026113000 + 10 * root["root"]
            results = {}
            for budget in (36, 108):
                chain, score, scores = optimize(
                    prefix, 15, budget, np.random.default_rng(seed),
                    lambda actions: np.linalg.norm(previous.rollout(actions)[:, :2] - goal, axis=1))
                results[str(budget)] = {"chain": chain.tolist(), "score": score, "scores": scores}
            np.testing.assert_allclose(results["36"]["scores"], program["scores"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(results["36"]["scores"], results["108"]["scores"][:36], atol=1e-12, rtol=0)
            rows.append({"root": root["root"], "name": program["name"], "prefix": prefix.tolist(),
                         "short": results["36"], "long": results["108"]})
    if any(previous.sha(path) != value for path, value in hashes.items()):
        raise ValueError("dependency changed during diagnostic")
    with OUTPUT.open("x") as stream:
        json.dump({"scope": "post-hoc exposed fixed-prefix diagnostic", "dependencies": hashes,
                   "rows": rows, "PC2_pass": False, "PC3_pass": False}, stream, indent=2, allow_nan=False)
    print(f"{len(rows)} fixed candidate budgets audited", flush=True)


if __name__ == "__main__":
    main()
