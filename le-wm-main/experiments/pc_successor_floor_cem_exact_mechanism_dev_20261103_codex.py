"""Same-final-chain interventions for the sealed floor-CEM compiler."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_successor_floor_cem_exact_dev_20261103_codex as experiment
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize

PROTOCOL = experiment.prior.ROOT.parent / "docs/PC_SUCCESSOR_FLOOR_CEM_EXACT_MECHANISM_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = experiment.PREFIX.with_name(experiment.PREFIX.name + "_mechanism.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (Path(__file__), PROTOCOL, experiment.PLANS, experiment.AUDIT)
    hashes = {str(p): experiment.prior.sha(p) for p in paths}
    source = json.loads(experiment.PLANS.read_text())
    if (source["dependencies"] != experiment.dependencies()
            or json.loads(experiment.AUDIT.read_text())["plans_sha256"] != experiment.prior.sha(experiment.PLANS)):
        raise ValueError("source seal mismatch")
    print("Testing the sealed final selected chain against zero and same-bank greedy.", flush=True)
    rows = []
    for row in source["rows"]:
        if row["arm"] != "successor":
            continue
        task, result, root = row["task"], row["result"], row["root"]
        goal = np.asarray(task["goal"])
        chain = np.asarray(result["selected"]["chain"])
        first = np.asarray([x["prefix"] for x in result["first_bank"]])
        first_scores = np.linalg.norm(experiment.prior.exact.endpoints(task, first) - goal, axis=1)
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)), "greedy": first[np.argmin(first_scores)]}
        tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        tails.append(chain[5:])
        controls = {}
        for name, prefix in prefixes.items():
            prefix_distance = float(np.linalg.norm(experiment.prior.exact.endpoints(task, prefix[None])[0] - goal))
            common_actions = np.asarray([np.concatenate((prefix, tail)) for tail in tails])
            common_scores = np.linalg.norm(experiment.prior.exact.endpoints(task, common_actions) - goal, axis=1)
            trace = []

            def predict(actions):
                values = np.linalg.norm(experiment.prior.exact.endpoints(task, actions) - goal, axis=1)
                trace.append({"actions": actions.tolist(), "scores": values.tolist()})
                return values

            best, score, _ = optimize(prefix, 15, 120, np.random.default_rng(2026121000 + root), predict)
            controls[name] = {"prefix": prefix.tolist(), "prefix_distance": prefix_distance,
                              "common_actions": common_actions.tolist(), "common_scores": common_scores.tolist(),
                              "active_changes": (common_scores - common_scores[0]).tolist(),
                              "own_chain": best.tolist(), "own_score": score, "own_trace": trace}
        initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))
        selected = controls["selected"]
        nonprogress = selected["prefix_distance"] >= initial - 1e-10
        active = all(selected["active_changes"][-1] < controls[k]["active_changes"][-1] - 1e-10
                     for k in ("zero", "greedy"))
        own = all(selected["own_score"] < controls[k]["own_score"] - 1e-10 for k in ("zero", "greedy"))
        rows.append({"root": root, "task": task, "chain": chain.tolist(), "initial_distance": initial,
                     "first_bank_distances": first_scores.tolist(), "controls": controls,
                     "nonprogress": bool(nonprogress), "active_response": bool(active),
                     "own_advantage": bool(own), "limited_witness": bool(nonprogress and active and own)})
    if len(rows) != 24 or hashes != {str(p): experiment.prior.sha(p) for p in paths}:
        raise ValueError("unexpected panel or source drift")
    with OUTPUT.open("x") as stream:
        json.dump({"sources": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Recorded interventions on {len(rows)} final chains.", flush=True)


if __name__ == "__main__":
    main()
