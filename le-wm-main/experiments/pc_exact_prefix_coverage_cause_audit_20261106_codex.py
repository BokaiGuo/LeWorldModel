"""Recompute recorded physical outcomes for fixed-prefix coverage diagnostic."""
import json

import numpy as np

from experiments import pc_exact_prefix_coverage_cause_dev_20261106_codex as experiment

OUTPUT = experiment.OUTPUT.with_name(experiment.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(experiment.OUTPUT.read_text())
    prior = json.loads(experiment.prior.OUTPUT.read_text())
    if source["dependencies"] != {path: experiment.prior.sha(path) for path in source["dependencies"]}:
        raise ValueError("dependency drift")
    if len(source["rows"]) != 24:
        raise ValueError("wrong panel")
    short, long, witness_search = [], [], []
    for row, previous in zip(source["rows"], prior["rows"], strict=True):
        task, root = row["task"], row["root"]
        if task != previous["task"] or root != previous["root"]:
            raise ValueError("task mismatch")
        pool = np.asarray(previous["pool"])
        goal = np.asarray(task["goal"])
        for i, result in enumerate(row["pool_longer"]):
            actions = np.asarray(result["actions"])
            if actions.shape != (384, 15, 2) or np.max(np.abs(actions)) > 1 + 1e-12:
                raise ValueError("invalid candidate")
            np.testing.assert_array_equal(actions[:, :5], np.broadcast_to(pool[i, :5], (384, 5, 2)))
            np.testing.assert_array_equal(actions[:36], previous["refined"][i]["actions"])
            scores = np.linalg.norm(experiment.prior.exact.endpoints(task, actions) - goal, axis=1)
            np.testing.assert_allclose(scores, result["costs"], atol=1e-12, rtol=0)
            if result["best"] != int(np.argmin(scores)):
                raise ValueError("best mismatch")
        short.append(min(min(item["costs"][:36]) for item in row["pool_longer"]))
        long.append(min(min(item["costs"]) for item in row["pool_longer"]))
        witness = np.asarray(row["witness_chain"])
        witness_result = row["witness_prefix_search"]
        actions = np.asarray(witness_result["actions"])
        if np.max(np.abs(witness)) > 1 + 1e-12 or any(row["witness_blocked"]):
            raise ValueError("witness invalid")
        np.testing.assert_array_equal(actions[:, :5], np.broadcast_to(witness[:5], (384, 5, 2)))
        np.testing.assert_array_equal(actions[0, 5:], np.zeros((10, 2)))
        scores = np.linalg.norm(experiment.prior.exact.endpoints(task, actions) - goal, axis=1)
        np.testing.assert_allclose(scores, witness_result["costs"], atol=1e-12, rtol=0)
        wscore = float(np.linalg.norm(experiment.prior.exact.endpoints(task, witness[None])[0] - goal))
        if wscore >= .05 or not np.isclose(wscore, row["witness_score"], atol=1e-12, rtol=0):
            raise ValueError("witness replay mismatch")
        witness_search.append(float(scores.min()))
    short, long, witness_search = map(np.asarray, (short, long, witness_search))
    report = {
        "source_sha256": experiment.prior.sha(experiment.OUTPUT),
        "roots": len(short), "feasible_witnesses": len(short),
        "short_pool_success": int((short < .05).sum()), "long_pool_success": int((long < .05).sum()),
        "witness_prefix_search_success": int((witness_search < .05).sum()),
        "long_pool_fail_witness_search_success": int(((long >= .05) & (witness_search < .05)).sum()),
        "short_pool_mean": float(short.mean()), "long_pool_mean": float(long.mean()),
        "witness_prefix_search_mean": float(witness_search.mean()),
        "long_improves": int((long < short - 1e-12).sum()),
        "short": short.tolist(), "long": long.tolist(), "witness_search": witness_search.tolist(),
    }
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
