"""Read-only consistency audit and summary of the sealed development run."""
import json

import numpy as np

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as run
from experiments.pc_input_permissions import file_hash
from experiments.pc_rgb_hold_retention_search_20261107_codex import choose_retained


def main():
    data = json.loads(run.OUTPUT.read_text())
    reference = json.loads(run.REFERENCE.read_text())
    assert data["runner_sha256"] == file_hash(run.__file__)
    assert data["protocol_sha256"] == file_hash(run.PROTOCOL)
    assert data["input_sha256"] == file_hash(run.INPUT)
    assert data["reference_sha256"] == file_hash(run.REFERENCE)
    assert len(data["rows"]) == len(reference["rows"]) == 40
    decisions = steps = 0
    for row, ref in zip(data["rows"], reference["rows"], strict=True):
        assert row["episode"] == ref["episode"]
        for arm in run.ARMS:
            result = row["arms"][arm]
            trace = result["trace"]
            assert result["ever_success"] == any(x["terminated"] for x in trace)
            assert all(x["terminated"] == (x["distance"] < 16) for x in trace)
            reconstructed = []
            for decision in result["decisions"]:
                decisions += 1
                remaining = decision["remaining"]
                assert remaining == 25 - len(reconstructed)
                expected = 300 if arm == "cem300" else 16
                assert decision["queries"] == expected
                assert decision["packet_steps"] == expected * (remaining // 5)
                actions = np.asarray(decision["actions"])
                dispatched = np.asarray(decision["dispatched"])
                assert actions.shape == (remaining, 2)
                assert np.isfinite(actions).all() and np.abs(actions).max() <= 1
                np.testing.assert_array_equal(actions[:len(dispatched)], dispatched)
                reconstructed.extend(decision["dispatched"])
                if arm in ("effect", "lookahead"):
                    if remaining > 5:
                        costs = np.asarray(decision["probe_cost"])
                        assert decision["retained"] == choose_retained(costs, arm, decision["strategy_seed"])
                        all_scores = sum(decision["temporal_scores"], [])
                        choice = decision["choice"]
                        offset = {"parent": 0, "probe": 3, "tail": 12}[choice["source"]]
                        assert offset + choice["index"] == int(np.argmin(all_scores))
                        assert np.isclose(choice["cost"], all_scores[offset + choice["index"]])
                else:
                    scores = [decision["incumbent_cost"]] + sum(
                        (r["costs"] for r in decision["rounds"]), [])
                    assert len(scores) == expected
                    assert decision["chosen"] == int(np.argmin(scores))
                    assert np.isclose(decision["chosen_cost"], min(scores))
            np.testing.assert_array_equal(np.asarray(reconstructed), np.asarray([x["action"] for x in trace]))
            steps += len(trace)
    print(f"audit: {decisions} decisions, {steps} dispatched steps, 40 episodes, 4 arms")
    print(f"reference ever success={sum(any(t['success'] for t in r['trace']) for r in reference['rows'])}/40")
    print(f"initial successes={sum(r['initial_success'] for r in data['rows'])}/40")
    for subset_name, rows in (("all", data["rows"]),
                              ("not_initial", [r for r in data["rows"] if not r["initial_success"]])):
        print(f"\n{subset_name} n={len(rows)}")
        for arm in run.ARMS:
            results = [r["arms"][arm] for r in rows]
            success = sum(r["ever_success"] for r in results)
            queries = sum(sum(d["queries"] for d in r["decisions"]) for r in results)
            packets = sum(sum(d["packet_steps"] for d in r["decisions"]) for r in results)
            planning = sum(sum(d["planning_seconds"] for d in r["decisions"]) for r in results)
            wall = sum(r["wall_seconds"] for r in results)
            print(f"{arm}: success={success}, queries={queries}, packets={packets}, planning={planning:.3f}s, wall={wall:.3f}s")
        for baseline in ("lookahead", "cem16", "cem300"):
            pairs = [(r["arms"]["effect"]["ever_success"], r["arms"][baseline]["ever_success"]) for r in rows]
            print(f"effect vs {baseline}: win={sum(a and not b for a, b in pairs)}, loss={sum(b and not a for a, b in pairs)}")


if __name__ == "__main__":
    main()
