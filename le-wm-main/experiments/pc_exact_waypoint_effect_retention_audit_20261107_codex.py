"""Independent replay of recorded selection, action trajectories, and scoring."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_waypoint_effect_retention_20261107_codex as experiment

OUTPUT = experiment.ROOT / "results/pc_exact_waypoint_effect_retention_audit_20261107_codex.json"


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    recorded = json.loads(experiment.OUTPUT.read_text())
    if recorded["dependencies"] != {p: experiment.core.sha(p) for p in recorded["dependencies"]}:
        raise ValueError("dependency mismatch")
    source = json.loads(experiment.source.OUTPUT.read_text())
    audits = []
    for row, original in zip(recorded["rows"], source["rows"], strict=True):
        if row["root"] != original["root"] or row["task"] != original["task"]:
            raise ValueError("root mismatch")
        root, task = row["root"], row["task"]
        chains = np.asarray(original["actions"])
        order = np.argsort(original["offsets"], kind="stable")
        pool_ix, a_ix, b_ix = (order[np.arange(start, 384, 48)] for start in (24, 12, 36))
        for key, indices in (("pool_indices", pool_ix), ("a_indices", a_ix), ("b_indices", b_ix)):
            np.testing.assert_array_equal(row[key], indices)
        pool = chains[pool_ix]
        goal = np.asarray(task["goal"])
        def cost(actions):
            actions = np.asarray(actions)
            if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                raise ValueError("illegal actions")
            return np.linalg.norm(experiment.core.exact.endpoints(task, actions) - goal, axis=1)
        np.testing.assert_allclose(cost(chains), original["costs"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(cost(pool[:, :5]), row["immediate"], atol=1e-12, rtol=0)
        for name, ix in (("a_cost", a_ix), ("b_cost", b_ix)):
            tails = chains[ix, 5:]
            cross = np.concatenate((np.repeat(pool[:, None, :5], 8, axis=1),
                                    np.repeat(tails[None], 8, axis=0)), axis=2)
            np.testing.assert_allclose(cost(cross.reshape(-1, 15, 2)).reshape(8, 8),
                                       row[name], atol=1e-12, rtol=0)
        a = np.asarray(row["a_cost"])
        look = a.min(axis=1)
        ordered = np.argsort(look, kind="stable")
        first = int(ordered[0])
        eligible = ordered[1:6]
        centered = a - a.mean(axis=1, keepdims=True)
        rng = np.random.default_rng(2714010 + root)
        shuffled = np.stack([v[rng.permutation(8)] for v in centered])
        def diverse(v):
            return [first, int(eligible[np.argmax(np.linalg.norm(v[eligible] - v[first], axis=1))])]
        expected = {"parent": np.argsort(np.asarray(original["costs"])[pool_ix], kind="stable")[:2].tolist(),
                    "immediate": np.argsort(row["immediate"], kind="stable")[:2].tolist(),
                    "lookahead": ordered[:2].tolist(),
                    "descriptor": diverse(centered), "shuffled": diverse(shuffled)}
        if expected != row["selected"]:
            raise ValueError("selection mismatch")
        for i, trace in enumerate(row["refined"]):
            actions = np.asarray(trace["actions"])
            if actions.shape != (36, 15, 2):
                raise ValueError("refinement count")
            np.testing.assert_array_equal(actions[:, :5], np.broadcast_to(pool[i, :5], (36, 5, 2)))
            np.testing.assert_allclose(cost(actions), trace["costs"], atol=1e-12, rtol=0)
            if trace["best"] != int(np.argmin(trace["costs"])):
                raise ValueError("incorrect refined best")
        incumbent = int(np.argmin(original["costs"]))
        if row["incumbent_index"] != incumbent:
            raise ValueError("incorrect incumbent")
        for arm, indices in expected.items():
            eligible_results = [(original["costs"][incumbent], "incumbent", chains[incumbent])]
            for i in indices:
                t = row["refined"][i]
                eligible_results.append((t["costs"][t["best"]], i, t["actions"][t["best"]]))
            value, origin, chain = min(eligible_results, key=lambda item: item[0])
            outcome = row["outcomes"][arm]
            if origin != outcome["origin"] or not np.array_equal(chain, outcome["chain"]):
                raise ValueError("selection leaked or differs")
            np.testing.assert_allclose(value, outcome["score"], atol=1e-12, rtol=0)
        if row["deployment_full_chain_evaluations"] != 520 or row["deployment_prefix_evaluations"] != 8:
            raise ValueError("budget mismatch")
        audits.append({"root": root, "source_chains": 384, "a_combinations": 64,
                       "b_combinations": 64, "refinement_chains": 288, "passed": True})
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": experiment.core.sha(experiment.OUTPUT),
                   "audit_sha256": experiment.core.sha(Path(__file__)), "roots": audits},
                  stream, separators=(",", ":"))
    print(f"replayed {len(audits)} roots")


if __name__ == "__main__":
    run()
