"""Deterministically replay and summarize sealed matched-prefix searches."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_goal_firstbank_matched_suffix_20261103_codex as run


def main():
    if run.AUDIT.exists():
        raise FileExistsError(run.AUDIT)
    data = json.loads(run.OUTPUT.read_text())
    original = json.loads(run.source.PLANS.read_text())
    if data["dependencies"] != run.dependencies() or original["dependencies"] != run.source.dependencies():
        raise ValueError("dependency mismatch")
    originals = {r["root"]: r for r in original["rows"] if r["arm"] == "goal"}
    if len(data["rows"]) != 24 or len({r["root"] for r in data["rows"]}) != 24:
        raise ValueError("root panel mismatch")
    summaries = {n: {"sets": {name: [] for name in ("retained", "greedy", "random", "oracle")},
                     "retained_contains_oracle": 0, "original_scores": [], "per_root": []}
                 for n in (1, 2, 4)}
    searches = 0
    for row in data["rows"]:
        root, old = row["root"], originals[row["root"]]
        first = old["result"]["first_bank"]
        if row["task"] != old["task"] or len(row["prefixes"]) != len(first):
            raise ValueError("source binding mismatch")
        expected_sets = {"retained": old["result"]["retained"],
                         "greedy": np.argsort(old["result"]["prefix_scores"], kind="stable")[:2].tolist(),
                         "random": np.random.default_rng(2030110300 + root).choice(len(first), size=2, replace=False).tolist()}
        if row["sets"] != expected_sets or row["original_selected_score"] != old["result"]["selected"]["terminal_score"]:
            raise ValueError("set or original score mismatch")
        for i, item in enumerate(row["prefixes"]):
            if item["index"] != i or item["name"] != first[i]["name"] or item["prefix"] != first[i]["prefix"]:
                raise ValueError("candidate mismatch")
            replay = run.run_prefix(row["task"], root, item["prefix"])
            for actual, expected in zip(replay["starts"], item["starts"], strict=True):
                if actual["seed"] != expected["seed"] or actual["batch_sizes"] != expected["batch_sizes"]:
                    raise ValueError("random seed or batching mismatch")
                np.testing.assert_array_equal(actual["scores"], expected["scores"])
                np.testing.assert_array_equal(actual["best_chain"], expected["best_chain"])
                if actual["best_score"] != expected["best_score"] or actual["prediction_tokens"] != 480:
                    raise ValueError("replay or budget mismatch")
                searches += 1
            np.testing.assert_array_equal(replay["cumulative_best"], item["cumulative_best"])
        for n, summary in summaries.items():
            values = [p["cumulative_best"][n - 1] for p in row["prefixes"]]
            oracle = min(values)
            by_set = {key: min(values[i] for i in indices) for key, indices in row["sets"].items()}
            by_set["oracle"] = oracle
            summary["retained_contains_oracle"] += int(any(values[i] == oracle for i in row["sets"]["retained"]))
            summary["original_scores"].append(row["original_selected_score"])
            for key, value in by_set.items():
                summary["sets"][key].append(value)
            summary["per_root"].append({"root": root, "scores": by_set, "original_score": row["original_selected_score"],
                                        "oracle_indices": [i for i, value in enumerate(values) if value == oracle]})
        print(f"replayed root {root}", flush=True)
    for summary in summaries.values():
        oracle = np.asarray(summary["sets"]["oracle"])
        summary["aggregate"] = {}
        for name, scores in summary["sets"].items():
            scores = np.asarray(scores)
            summary["aggregate"][name] = {"mean": float(scores.mean()), "median": float(np.median(scores)),
                                          "successes": int((scores < .05).sum()),
                                          "mean_gap_to_oracle": float((scores - oracle).mean()),
                                          "strictly_above_oracle_roots": int((scores > oracle + 1e-12).sum())}
        del summary["sets"]
    if searches != 24 * 5 * 4:
        raise ValueError("search count mismatch")
    with run.AUDIT.open("x") as stream:
        json.dump({"source_sha256": run.source.sha(run.OUTPUT), "audit_sha256": run.source.sha(Path(__file__)),
                   "searches_replayed": searches, "chains_replayed": searches * run.SAMPLES,
                   "prediction_tokens_replayed": searches * run.SAMPLES * 3, "summary": summaries},
                  stream, indent=2, allow_nan=False)
    print(json.dumps({n: s["aggregate"] for n, s in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
