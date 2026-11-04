"""Recompute the sealed same-chain mechanism interventions independently."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_search_e0_dev_20261104_codex as run

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_mechanism_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.PLANS.read_text())
    audit = json.loads(run.AUDIT.read_text())
    original = json.loads(run.MECHANISM.read_text())
    if (data["dependencies"] != run.dependencies() or audit["plans_sha256"] != run.sha(run.PLANS)
            or original["plans_sha256"] != run.sha(run.PLANS)
            or original["audit_sha256"] != run.sha(run.AUDIT)):
        raise ValueError("source binding mismatch")
    source = {(row["root"], row["arm"]): row for row in data["rows"]}
    verified = []
    for row in original["rows"]:
        root, task = row["root"], row["task"]
        response = source[root, "response"]
        result = response["result"]
        chain, goal = np.asarray(result["selected"]["chain"]), np.asarray(task["goal"])
        if row["chain"] != chain.tolist() or task != response["task"]:
            raise ValueError("selected chain mismatch")
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(result["first_bank"][int(np.argmin(result["prefix_scores"]))]["prefix"]),
                    "random_bridge": np.asarray(source[root, "matched_random"]["result"]["selected"]["chain"])[:5],
                    "lookahead": np.asarray(source[root, "smooth_cem"]["result"]["selected"]["chain"])[:5]}
        suffixes = [np.tile(value, (10, 1)) for value in
                    ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        for name, prefix in prefixes.items():
            saved = row["controls"][name]
            if saved["prefix"] != prefix.tolist():
                raise ValueError("prefix mismatch")

            def predict(actions):
                pos = run.exact.endpoints(task, np.asarray(actions, dtype=np.float32))
                return pos, np.linalg.norm(pos - goal, axis=1)

            distance = float(np.linalg.norm(run.exact.endpoints(task, prefix[None].astype(np.float32))[0] - goal))
            joined = np.stack([np.concatenate((prefix, suffix)) for suffix in suffixes])
            _, common = predict(joined)
            best, score, generations = run.method.cem(np.zeros((10, 2)), prefix, 119,
                np.random.default_rng(2027111400 + root), predict, float(common[0]), 4)
            axes = np.eye(10).reshape(10, 5, 2)
            tails = np.zeros((20, 10, 2))
            tails[0::2, :5], tails[1::2, :5] = .05 * axes, -.05 * axes
            pos, _ = predict(np.concatenate((np.broadcast_to(prefix, (20, 5, 2)), tails), axis=1))
            matrix = ((pos[0::2] - pos[1::2]) / .1).T
            if (distance != saved["prefix_distance"] or not np.array_equal(common, saved["common_distances"])
                    or not np.array_equal(common-common[0], saved["active_changes"])
                    or not np.array_equal(best, saved["own_chain"]) or score != saved["own_distance"]
                    or generations != saved["generations"]
                    or not np.array_equal(matrix, saved["successor_response"])
                    or saved["diagnostic_rollouts"] != 146):
                raise ValueError(f"recomputed mechanism mismatch: {root}/{name}")
        controls, selected = row["controls"], row["controls"]["selected"]
        initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))
        nonprogress = (selected["prefix_distance"] >= initial - 1e-10
                       and np.any(np.asarray(selected["prefix"], dtype=np.float32) != 0))
        comparisons = [name for name in prefixes if name != "selected"]
        active = {name: selected["active_changes"][-1] < controls[name]["active_changes"][-1] - 1e-10
                  for name in comparisons}
        own = {name: selected["own_distance"] < controls[name]["own_distance"] - 1e-10
               for name in comparisons}
        gaps = {name: float(np.linalg.norm(np.asarray(selected["successor_response"])
                                     - np.asarray(controls[name]["successor_response"]))) for name in comparisons}
        if (initial != row["initial_distance"] or bool(nonprogress) != row["nonprogress"]
                or active != row["active_better"] or own != row["own_better"] or gaps != row["response_gap"]
                or row["base_intersection"] != bool(nonprogress and all(active[x] and own[x] for x in ("zero", "greedy")))
                or row["all_intersection"] != bool(nonprogress and all(active[x] and own[x] for x in comparisons))):
            raise ValueError("mechanism comparison mismatch")
        verified.append(root)
    if verified != list(range(24)):
        raise ValueError("incomplete panel")
    run.write(OUTPUT, {"mechanism_sha256": run.sha(run.MECHANISM), "audit_script_sha256": run.sha(Path(__file__)),
                       "verified_roots": verified, "controls_per_root": 5, "rollouts_per_control": 146,
                       "scope": "deterministic reconstruction of prior intervention; not independent replication"})
    print("Reconstructed all 120 prefix interventions and comparisons.", flush=True)


if __name__ == "__main__":
    main()
