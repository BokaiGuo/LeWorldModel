"""Reconstruct the sealed equal-tail optimizer and check dispatched actions."""
import json

import numpy as np

from experiments import pc_shared_rgb_equal_tail_replan_20261107_codex as run
from experiments.pc_input_permissions import file_hash


def main():
    plan = json.loads(run.PLAN.read_text())
    physical = json.loads(run.PHYSICAL.read_text())
    source = json.loads(run.base.PLAN.read_text())
    assert physical["plan_sha256"] == file_hash(run.PLAN)
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("source_plan_sha256", run.base.PLAN), ("pool_sha256", run.POOL)):
        assert plan[key] == file_hash(path), key
    assert len(plan["rows"]) == len(physical["rows"]) == len(source["rows"]) == 3
    checked = 0
    with np.load(run.POOL, allow_pickle=False) as pool:
        for row, actual, origin in zip(plan["rows"], physical["rows"], source["rows"], strict=True):
            root = row["root"]
            assert root == actual["root"] == origin["root"]
            tail = np.asarray(origin["common_tail"], dtype=np.float32).reshape(-1, 2)
            prefixes = np.asarray(origin["prefix_candidates"], dtype=np.float32)
            assert row["actual_horizon"] == origin["original_length"] == len(tail) + 5
            for arm in run.base.ARMS:
                chains = pool[f"r{root}_{arm}_chains"]
                scores = pool[f"r{root}_{arm}_scores"]
                entry = row["arms"][arm]
                assert chains.shape == (48, 25, 2) and scores.shape == (48,)
                assert np.isfinite(chains).all() and np.isfinite(scores).all()
                assert np.max(np.abs(chains)) <= 1 and entry["queries"] == 48
                prefix = prefixes[origin["choices"][arm]]
                mean, std, best, bestcost = tail.copy(), np.full_like(tail, .4), tail.copy(), np.inf
                rng = np.random.default_rng(27124000 + root)
                for iteration in range(3):
                    noise = rng.normal(size=(16, *tail.shape))
                    expected = run.make_batch(prefix, mean, std, best, noise)
                    batch = chains[iteration * 16:(iteration + 1) * 16]
                    values = scores[iteration * 16:(iteration + 1) * 16]
                    np.testing.assert_array_equal(batch, expected)
                    index = int(np.argmin(values))
                    if values[index] < bestcost:
                        bestcost = float(values[index])
                        best = batch[index, 5:5 + len(tail)].copy()
                    elite = batch[np.argsort(values, kind="stable")[:4], 5:5 + len(tail)]
                    mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .05)
                choice = int(np.argmin(scores))
                assert choice == entry["choice"] and float(scores[choice]) == entry["score"]
                outcome = actual["outcomes"][arm]
                trace = outcome["trace"]
                assert 0 < len(trace) <= row["actual_horizon"]
                np.testing.assert_array_equal(np.asarray([t["action"] for t in trace]), chains[choice, :len(trace)])
                assert not any(t["terminated"] or t["truncated"] for t in trace[:-1])
                assert outcome["ever"] == any(t["terminated"] for t in trace)
                assert outcome["prefix_success"] == any(t["terminated"] for t in trace[:5])
                if len(trace) < row["actual_horizon"]:
                    assert trace[-1]["terminated"] or trace[-1]["truncated"]
                checked += 1
    print(f"PASS: {checked} arms, {checked * 48} scored candidates, RNG/elite/update/selection and dispatched traces")


if __name__ == "__main__":
    main()
