"""Verified H10 per-step physical measurements; exploratory, never an admission."""
import argparse
import json
import math
import os
from pathlib import Path

from . import codex_p1_three_prefix_h10_analysis_20261030 as analysis
from . import codex_p1_three_prefix_h10_runner_20261030 as runner

NAME = "codex_p1_h10_stepwise_20261030_main"
need = runner.require


def metrics(state, start, goal):
    state, start, goal = map(analysis.vector, (state, start, goal))
    return {
        "pusher_displacement_xy": [state[i] - start[i] for i in (0, 1)],
        "pusher_goal_error": math.hypot(state[0] - goal[0], state[1] - goal[1]),
        **analysis.measure(state, start, goal),
    }


def consume(directory, report_sha256, prepared, prepare_sha256):
    # The original analyzer first verifies the entire fixed denominator and
    # all saved source, transition and image evidence.
    verified = analysis.consume(directory, report_sha256, prepared, prepare_sha256)
    prepare = runner.verify_prepare(prepared, prepare_sha256)["original"]
    root = Path(directory)
    report = runner.native.read(root / "run_report.json", cap=4 * 1024**2)
    shards = []
    for i, anchor in enumerate(prepare["anchors"]):
        path = root / f"anchor_{i:02d}" / "report.json"
        expected = report["shards"][i]["report_sha256"]
        need(runner.runner.physical.sha(path) == expected, "shard digest before stepwise read")
        shard = runner.native.read(path, cap=64 * 1024**2)
        rows = []
        for j, row in enumerate(shard["rows"]):
            need(row["index"] == j and row["status"] ==
                 verified["groups"][i // 3]["probes"][j]["roles"][anchor["role"]]["status"],
                 "stepwise row binding")
            start, goal = anchor["checkpoint"]["state"], anchor["goal"]
            steps = []
            for k, response in enumerate(row["responses"], 1):
                need(response["step"] == k, "noncontiguous physical response")
                steps.append({"step": k, "terminated": response["terminated"],
                              "truncated": response["truncated"],
                              "metrics": metrics(response["state"], start, goal)})
            rows.append({"index": j, "status": row["status"], "steps": steps})
        need(runner.runner.physical.sha(path) == expected, "shard digest after stepwise read")
        shards.append({"anchor": i, "role": anchor["role"],
                       "prefix_checkpoint_vs_root": metrics(anchor["checkpoint"]["state"],
                                                            anchor["root_state"], anchor["goal"]),
                       "root_metrics": metrics(anchor["root_state"], anchor["root_state"], anchor["goal"]),
                       "rows": rows})
    need(runner.runner.physical.sha(root / "run_report.json") == report_sha256,
         "run report drift")
    return {"schema": NAME + ".v1", "run_report_sha256": report_sha256,
            "prepare_sha256": prepare_sha256, "status_counts": verified["status_counts"],
            "shards": shards, "P1_admission": False, "independent_confirmation": False,
            "limitations": "Known roots; no imputation; no contact event inferred from distance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("directory", "report-sha256", "prepared", "prepare-sha256", "output"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == runner.ROOT / "results" and
         output.name.startswith(NAME + "_") and not output.is_symlink(), "exclusive output namespace")
    result = consume(args.directory, args.report_sha256, args.prepared, args.prepare_sha256)
    with output.open("xb") as stream:
        data = runner.native.raw(result)
        need(stream.write(data) == len(data), "output write")
        stream.flush()
        os.fsync(stream.fileno())
    runner.base.paired.fsync_dir(output.parent)
    print(json.dumps({"status": "verified_exploratory", "sha256": runner.runner.physical.sha(output),
                      "P1_admission": False}))


if __name__ == "__main__":
    main()
