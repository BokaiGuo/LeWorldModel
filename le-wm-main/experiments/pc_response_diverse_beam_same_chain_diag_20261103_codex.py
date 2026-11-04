"""Post-outcome interventions on the sealed diverse-beam final chain."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_diverse_beam_exact_dev_20261103_codex as exp
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize

PROTOCOL = exp.prior.ROOT.parent / "docs/PC_RESPONSE_DIVERSE_BEAM_SAME_CHAIN_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_same_chain_diagnostic.json")
CONTROL_NAMES = ("zero", "greedy", "random_bridge", "lookahead")


def dependencies():
    paths = (Path(__file__), PROTOCOL, exp.PLANS, exp.AUDIT, exp.previous.PLANS, exp.previous.AUDIT)
    return {**exp.dependencies(), **{str(p): exp.prior.sha(p) for p in paths}}


def analyze(row, previous):
    task, result, root = row["task"], row["result"], row["root"]
    if row["arm"] != "response_diverse":
        raise ValueError("unexpected arm")
    goal, chain = np.asarray(task["goal"]), np.asarray(result["selected"]["chain"])
    first = np.asarray([x["prefix"] for x in result["first_bank"]])
    first_scores = np.linalg.norm(exp.prior.exact.endpoints(task, first.astype(np.float32)) - goal, axis=1)
    np.testing.assert_allclose(first_scores, result["prefix_scores"], atol=1e-12, rtol=0)
    prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)), "greedy": first[np.argmin(first_scores)]}
    for arm in ("random_bridge", "lookahead"):
        prior_row = previous[(root, arm)]
        if prior_row["task"] != task:
            raise ValueError("control task mismatch")
        prefixes[arm] = np.asarray(prior_row["result"]["selected"]["chain"])[:5]
    tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
    tails.append(chain[5:])
    controls = {}
    for name, prefix in prefixes.items():
        prefix_end = exp.prior.exact.endpoints(task, prefix[None].astype(np.float32))[0]
        prefix_distance = float(np.linalg.norm(prefix_end - goal))
        common = np.asarray([np.concatenate((prefix, tail)) for tail in tails])
        common_scores = np.linalg.norm(exp.prior.exact.endpoints(task, common.astype(np.float32)) - goal, axis=1)
        trace = []

        def predict(actions):
            model = np.asarray(actions, dtype=np.float32)
            values = np.linalg.norm(exp.prior.exact.endpoints(task, model) - goal, axis=1)
            trace.append({"requested": actions.tolist(), "model": model.tolist(), "scores": values.tolist()})
            return values

        best, score, _ = optimize(prefix, 15, 120, np.random.default_rng(2026121800 + root), predict)
        controls[name] = {"prefix": prefix.tolist(), "prefix_distance": prefix_distance,
                          "common_actions": common.tolist(), "common_scores": common_scores.tolist(),
                          "active_changes": (common_scores - common_scores[0]).tolist(),
                          "own_chain": best.tolist(), "own_score": score, "own_trace": trace}
    initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - goal))
    selected = controls["selected"]
    nonprogress = selected["prefix_distance"] >= initial - 1e-10
    active = {name: bool(selected["active_changes"][-1] < controls[name]["active_changes"][-1] - 1e-10)
              for name in CONTROL_NAMES}
    own = {name: bool(selected["own_score"] < controls[name]["own_score"] - 1e-10) for name in CONTROL_NAMES}
    base = bool(nonprogress and all(active[x] and own[x] for x in ("zero", "greedy")))
    all_controls = bool(nonprogress and all(active[x] and own[x] for x in CONTROL_NAMES))
    return {"root": root, "task": task, "chain": chain.tolist(), "initial_distance": initial,
            "first_scores": first_scores.tolist(), "controls": controls, "nonprogress": bool(nonprogress),
            "active_better": active, "own_better": own, "base_intersection": base,
            "all_controls_intersection": all_controls}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    hashes = dependencies()
    data, old = json.loads(exp.PLANS.read_text()), json.loads(exp.previous.PLANS.read_text())
    if (data["dependencies"] != exp.dependencies()
            or json.loads(exp.AUDIT.read_text())["plans_sha256"] != exp.prior.sha(exp.PLANS)
            or old["dependencies"] != exp.previous.dependencies()
            or json.loads(exp.previous.AUDIT.read_text())["plans_sha256"] != exp.prior.sha(exp.previous.PLANS)):
        raise ValueError("source seal mismatch")
    prior_rows = {(row["root"], row["arm"]): row for row in old["rows"]}
    print("Testing final selected chain with common tails and matched prefix replanning.", flush=True)
    rows = [analyze(row, prior_rows) for row in data["rows"] if row["arm"] == "response_diverse"]
    if len(rows) != 24 or hashes != dependencies():
        raise ValueError("panel or source drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed interventions for {len(rows)} final chains.", flush=True)


if __name__ == "__main__":
    main()
