"""Deterministically reconstruct saved conditional CEM searches."""
import json

import numpy as np

from experiments import pc_rgb_conditional_temporal_cem_20261107_codex as src
from experiments.pc_input_permissions import file_hash

OUTPUT = src.old.ROOT / "results/pc_rgb_conditional_temporal_cem_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, physical = json.loads(src.PLAN.read_text()), json.loads(src.PHYS.read_text())
    if (plan["sha256"]["pool"] != file_hash(src.POOL)
            or plan["sha256"]["runner"] != file_hash(src.__file__)
            or plan["sha256"]["protocol"] != file_hash(src.PROTOCOL)
            or physical["plan_sha256"] != file_hash(src.PLAN)
            or physical["pool_sha256"] != file_hash(src.POOL)):
        raise ValueError("artifact drift")
    rows = []
    with np.load(src.POOL, allow_pickle=False) as pool:
        for record in plan["rows"]:
            root = record["root"]
            inc, inc_scores = pool[f"r{root}_inc"], pool[f"r{root}_inc_scores"]
            actions, scores = pool[f"r{root}_actions"], pool[f"r{root}_scores"]
            parent = inc[record["parent"]]
            if src.digest(parent) != record["parent_sha256"]:
                raise ValueError("parent mismatch")
            blocks = parent.reshape(5, 5, 2)
            methods = {}
            for i, method in enumerate(src.METHODS):
                remaining = iter(scores[i].reshape(src.ROUNDS, src.N, 5))
                reconstructed, returned_scores, iterations = src.optimize(
                    parent[:10], blocks[2:, 0], lambda _: next(remaining), root, method)
                if not np.array_equal(reconstructed, actions[i]) or not np.array_equal(returned_scores, scores[i]):
                    raise ValueError("candidate reconstruction mismatch")
                if iterations != record["decisions"][method]["iterations"]:
                    raise ValueError("CEM update mismatch")
                chosen, index, value = src.choose(inc, inc_scores, actions[i], scores[i])
                if (index != record["decisions"][method]["index"] or
                        value != record["decisions"][method]["cost"] or
                        src.digest(chosen) != record["decisions"][method]["action_sha256"] or
                        not np.array_equal(chosen, pool[f"r{root}_selected"][i])):
                    raise ValueError("final selection mismatch")
                methods[method] = {"selected_index": index, "best_endpoint": value}
            if not np.array_equal(actions[0, :src.N], actions[1, :src.N]):
                raise ValueError("first-round common random numbers mismatch")
            rows.append({"root": root, "methods": methods,
                         "identical_actions_in_final_selection": np.array_equal(
                             pool[f"r{root}_selected"][0], pool[f"r{root}_selected"][1])})
    with OUTPUT.open("x") as f:
        json.dump({"rows": rows, "roots_verified": len(rows),
                   "identical_final_actions": sum(r["identical_actions_in_final_selection"] for r in rows),
                   "candidate_count_verified": len(rows) * len(src.METHODS) * src.ROUNDS * src.N,
                   "plan_sha256": file_hash(src.PLAN), "physical_sha256": file_hash(src.PHYS),
                   "runner_sha256": file_hash(__file__),
                   "scope": "saved-score deterministic reconstruction; not independent model or environment verification"},
                  f, indent=2, allow_nan=False)
    print(json.dumps({"roots_verified": len(rows),
                      "identical_final_actions": sum(r["identical_actions_in_final_selection"] for r in rows),
                      "candidate_count_verified": len(rows) * len(src.METHODS) * src.ROUNDS * src.N}, indent=2))


if __name__ == "__main__":
    main()
