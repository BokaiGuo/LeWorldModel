"""Independently recompute stored selection, trajectories, and paired metrics."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as run

OUTPUT = run.OUTPUT.with_name(run.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.OUTPUT.read_text())
    if data["dependencies"] != {path: run.sha(path) for path in data["dependencies"]}:
        raise ValueError("dependency drift")
    if len(data["rows"]) != 24:
        raise ValueError("missing task")
    summaries = {arm: [] for arm in run.ARMS}
    mse = {name: [] for name in ("parent", "row_mean", "additive", "descriptor")}
    for row, task in zip(data["rows"], run.panel.tasks(), strict=True):
        root = row["root"]
        if row["task"] != task:
            raise ValueError("task mismatch")
        goal = np.asarray(task["goal"])
        pool, a, b = (np.asarray(row[key]) for key in ("pool", "a", "b"))
        if pool.shape != (8, 15, 2) or np.max(np.abs(pool)) > 1:
            raise ValueError("invalid pool")
        for key, tails in (("a_cost", a), ("b_cost", b)):
            chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, len(tails), 5, 2)),
                                     np.broadcast_to(tails[None], (8, len(tails), 10, 2))), axis=2)
            scores = np.linalg.norm(run.exact.endpoints(task, chains.reshape(-1, 15, 2)) - goal, axis=1)
            np.testing.assert_allclose(scores.reshape(8, -1), row[key], atol=1e-12, rtol=0)
        parent = np.linalg.norm(run.exact.endpoints(task, pool) - goal, axis=1)
        immediate = np.linalg.norm(run.exact.endpoints(task, pool[:, :5]) - goal, axis=1)
        np.testing.assert_allclose(parent, row["parent_cost"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(immediate, row["immediate"], atol=1e-12, rtol=0)
        if run.select(np.asarray(row["a_cost"]), parent, immediate, 261110000 + root) != row["selected"]:
            raise ValueError("selection mismatch")
        search = run.optimize(task, [], np.zeros((15, 2)), 120, 261106000 + root)
        if search != row["nominal_search"]:
            raise ValueError("nominal search cannot be replayed")
        np.testing.assert_array_equal(pool[:4], np.asarray(search["actions"])[row["nominal_indices"]])
        np.testing.assert_array_equal(pool[4:], np.random.default_rng(261107000 + root).uniform(-1, 1, (4, 15, 2)))
        np.testing.assert_array_equal(a, np.random.default_rng(261108000 + root).uniform(-1, 1, (8, 10, 2)))
        np.testing.assert_array_equal(b, np.random.default_rng(261109000 + root).uniform(-1, 1, (16, 10, 2)))
        for i, stored in enumerate(row["refined"]):
            replay = run.optimize(task, pool[i, :5], pool[i, 5:], 36, 261111000 + root * 8 + i)
            if replay != stored:
                raise ValueError("refinement replay mismatch")
            chain = np.asarray(stored["actions"])
            if np.max(np.abs(chain)) > 1:
                raise ValueError("illegal refinement")
            np.testing.assert_array_equal(chain[:, :5], np.broadcast_to(pool[i, :5], (36, 5, 2)))
        for arm, indices in row["selected"].items():
            choice = min(indices, key=lambda i: row["refined"][i]["costs"][row["refined"][i]["best"]])
            result = row["outcomes"][arm]
            if result["prefix_index"] != choice:
                raise ValueError("choice leakage")
            np.testing.assert_array_equal(result["chain"], row["refined"][choice]["actions"][row["refined"][choice]["best"]])
            score = float(np.linalg.norm(run.exact.endpoints(task, np.asarray(result["chain"])[None])[0] - goal))
            np.testing.assert_allclose(score, result["score"], atol=1e-12, rtol=0)
            summaries[arm].append(score)
        predictions = run.heldout_predictions(a, b, np.asarray(row["a_cost"]), parent)
        for name, pred in predictions.items():
            value = float(np.square(pred - row["b_cost"]).mean())
            np.testing.assert_allclose(value, row["heldout_mse"][name], atol=1e-12, rtol=0)
            mse[name].append(value)
        if row["deployment_tokens"] != 788:
            raise ValueError("incorrect cost")
    desc = np.asarray(summaries["descriptor"])
    look = np.asarray(summaries["lookahead"])
    shuffled = np.asarray(summaries["shuffled"])
    wins = int(np.sum(desc < look - 1e-12))
    passed = (float(np.mean(look - desc)) >= .01 and wins >= 14
              and int(np.sum(desc < .05)) >= int(np.sum(look < .05))
              and float(np.mean(shuffled - desc)) >= .01)
    report = {
        "source_sha256": run.sha(run.OUTPUT), "roots": 24, "deployment_tokens_per_arm": 788,
        "arms": {arm: {"mean_score": float(np.mean(values)), "success": int(np.sum(np.asarray(values) < .05))}
                 for arm, values in summaries.items()},
        "descriptor_vs_lookahead_mean_improvement": float(np.mean(look - desc)),
        "descriptor_vs_lookahead_wins": wins,
        "descriptor_vs_shuffled_mean_improvement": float(np.mean(shuffled - desc)),
        "heldout_mse_mean": {name: float(np.mean(values)) for name, values in mse.items()},
        "admission_pass": bool(passed),
    }
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
