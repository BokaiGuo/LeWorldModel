"""Summarize repeated uninstrumented end-to-end evaluations."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"


def main():
    output = ROOT / "pc_cost_uninstrumented_summary_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    split = json.loads((ROOT / "pc_cost_split_repeated_summary_20261106_codex.json").read_text())
    runs = {}
    for name in ("cem", "prefix", "cem300x10", "cem300x30"):
        rows = [json.loads((ROOT / f"pc_cost_uninstrumented_{name}_r{i}.json").read_text())
                for i in (1, 2, 3)]
        expected = split["runs"][name]["samples"][0]["successes"]
        expected_episode = json.loads((ROOT / f"pc_cost_split_{name}_r1.run/eval_metrics.json").read_text())["episode_successes"]
        values = [sum(row["episode_successes"]) for row in rows]
        runs[name] = {"evaluation_wall_s": [r["evaluation_wall_s"] for r in rows],
                      "median_evaluation_wall_s": statistics.median(r["evaluation_wall_s"] for r in rows),
                      "process_wall_s": [r["process_wall_s"] for r in rows],
                      "successes": values, "same_success_count_as_instrumented": all(v == expected for v in values),
                      "same_per_episode_outcomes_as_instrumented":
                          all(r["episode_successes"] == expected_episode for r in rows)}
    with output.open("x") as stream:
        json.dump({"runs": runs, "scope": "same ten development tasks and fixed seeds repeated three times",
                   "limitations": "no action trace; video retained; timings have no explicit CUDA synchronization"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
