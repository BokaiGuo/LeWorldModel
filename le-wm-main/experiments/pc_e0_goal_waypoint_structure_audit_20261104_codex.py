"""Audit waypoint geometry, CEM prefix choices and post-refinement endpoints."""
import json

import numpy as np

from experiments import pc_e0_goal_waypoint_search_dev_20261104_codex as source


def main():
    out = source.OUT / "waypoint_structure_audit.json"
    if out.exists():
        raise FileExistsError(out)
    raw = json.loads((source.OUT / "result.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    if audit["status"] != "passed" or audit["result_sha256"] != source.prior.previous.sha(source.OUT / "result.json"):
        raise ValueError("source audit not bound")
    rows = []
    for row in raw["rows"]:
        if row["arm"] != "waypoint_cem":
            continue
        if row["status"] != "completed":
            rows.append({"root": row["root"], "status": "failed"})
            continue
        result, goal = row["result"], np.asarray(row["task"]["goal"])
        with np.load(source.OUT / row["trace_file"], allow_pickle=False) as trace:
            origin = trace["positions_1"][0]
            delta = goal - origin
            perpendicular = np.array([-delta[1], delta[0]])
            cursor, subrecords = 2, []
            for index, record in enumerate(result["waypoints"]):
                target = origin + .5 * delta + (-1 if index == 0 else 1) * 1.2 * perpendicular
                np.testing.assert_allclose(record["waypoint"], target, atol=1e-12, rtol=0)
                sizes = [g["size"] for g in record["prefix_generations"]]
                if sum(sizes) != 160:
                    raise ValueError("prefix sample allocation")
                candidate_actions = [trace[f"actions_{cursor}"][0]]
                candidate_scores = [np.linalg.norm(trace[f"positions_{cursor}"][0] - target)]
                cursor += 1
                for size in sizes:
                    a, pos = trace[f"actions_{cursor}"], trace[f"positions_{cursor}"]
                    if len(a) != size:
                        raise ValueError("prefix batch mismatch")
                    candidate_actions.extend(a)
                    candidate_scores.extend(np.linalg.norm(pos - target, axis=1))
                    cursor += 1
                first = int(np.argmin(candidate_scores))
                prefix = np.asarray(record["prefix"], np.float32)
                np.testing.assert_array_equal(prefix, candidate_actions[first])
                np.testing.assert_allclose(record["prefix_waypoint_distance"], candidate_scores[first], atol=1e-12, rtol=0)
                suffix_sizes = [g["size"] for g in record["suffix_generations"]]
                if sum(suffix_sizes) != 220:
                    raise ValueError("suffix sample allocation")
                all_suffix_actions, all_suffix_scores = [], []
                for _ in range(1 + len(suffix_sizes)):
                    a, s = trace[f"actions_{cursor}"], trace[f"scores_{cursor}"]
                    if not np.all(a[:, :5] == prefix):
                        raise ValueError("suffix prefix mismatch")
                    all_suffix_actions.extend(a)
                    all_suffix_scores.extend(s)
                    cursor += 1
                best_index = int(np.argmin(all_suffix_scores))
                np.testing.assert_array_equal(np.asarray(record["suffix_chain"], np.float32),
                                              all_suffix_actions[best_index])
                np.testing.assert_allclose(record["suffix_score"], all_suffix_scores[best_index], atol=1e-12, rtol=0)
                subrecords.append({"prefix_waypoint_distance": candidate_scores[first],
                                   "suffix_best_distance": all_suffix_scores[best_index]})
        chain = np.asarray(row["selected"], np.float32)
        prefix_position = source.prior.previous.exact.endpoints(row["task"], chain[None, :5])[0]
        prefix_distance = float(np.linalg.norm(prefix_position - goal))
        initial_distance = float(np.linalg.norm(np.asarray(row["task"]["start"][:2]) - goal))
        rows.append({"root": row["root"], "status": "completed", "initial_distance": initial_distance,
                     "actual_selected_prefix_distance": prefix_distance, "final_distance": row["score"],
                     "non_immediate_progress": prefix_distance >= initial_distance,
                     "branches": subrecords})
    source.prior.previous.write(out, {"status": "passed", "result_sha256": source.prior.previous.sha(source.OUT / "result.json"),
        "rows": rows, "scope": "same-backend structural audit; prefix distance is not a PC2 mechanism test"})
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
