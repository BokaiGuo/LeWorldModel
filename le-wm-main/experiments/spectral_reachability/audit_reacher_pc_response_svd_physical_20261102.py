"""Saved-only validation and paired summaries for response-SVD physical panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import audit_reacher_pc_p1_fd_common_physical_v2_20261102_codexexclusive as checks
from experiments.spectral_reachability import run_reacher_pc_p1_fd_common_physical_20261102_codexexclusive as physical


def validate(plan, events, plan_sha):
    checks.require(plan["schema"] == "pc_response_svd_exhaustive_physical_v1"
                   and plan["expected_branches"] == 130, "plan schema")
    checks.require(events and events[0]["kind"] == "start" and events[0]["plan_sha256"] == plan_sha
                   and events[-1]["kind"] == "finished", "journal framing")
    checks.require(not any(event["kind"] == "root_error" for event in events), "root failure")
    roots = [event for event in events if event["kind"] == "root"]
    branches = [event for event in events if event["kind"] == "branch"]
    checks.require(len(roots) == 2 and len(branches) == 130, "branch denominator")
    checks.require([root["seed"] for root in roots] == [r["seed"] for r in plan["roots"]], "root roster")
    paired = []
    counts = {"complete": 0, "stopped": 0, "error": 0}
    for root, observed in zip(plan["roots"], roots, strict=True):
        checks.require(observed["status"] == "verified"
                       and checks.rgb(observed["root_snapshot"]["rendered_goal_rgb"])
                       == root["goal_rgb_sha256"], "root goal")
        rows = [b for b in branches if b["seed"] == root["seed"]]
        checks.require(len(rows) == 65, "per-root denominator")
        outcomes = {}
        prefix_references = {}
        for expected, row in zip(root["branches"], rows, strict=True):
            key = (expected["prefix_index"], expected["tail_index"])
            checks.require((row["prefix_index"], row["tail_index"]) == key, "order mismatch")
            actions = np.asarray(row["raw_actions"], dtype=np.float32)
            checks.require(actions.shape == (15, 2)
                           and np.array_equal(actions, np.asarray(expected["raw_actions"], dtype=np.float32))
                           and physical.digest_actions(actions) == expected["actions_sha256"]
                           == row["actions_sha256"], "action mismatch")
            status = row["status"]
            checks.require(status in counts, "unknown status")
            counts[status] += 1
            trajectory = row["trajectory"]
            steps = row["returned_steps"]
            checks.require(type(steps) is int and 0 <= steps <= 15
                           and type(row["attempted_steps"]) is int
                           and steps <= row["attempted_steps"] <= 15
                           and len(trajectory) in (0, steps + 1), "trajectory length")
            if status != "error":
                checks.require(row["identity"]["reset_rgb_sha256"] == root["root_rgb_sha256"]
                               and row["identity"]["goal_rgb_sha256"] == root["goal_rgb_sha256"]
                               and row["attempted_steps"] == steps, "execution identity")
            if status == "complete":
                checks.require(steps == 15 and row["stop"] is None and len(trajectory) == 16,
                               "incomplete complete branch")
            if status == "stopped":
                checks.require(row["stop"] is not None and row["stop"]["step"] == steps
                               and len(trajectory) == steps + 1, "bad stop")
            for i, point in enumerate(trajectory):
                checks.verify_point(point)
                if i:
                    checks.require(np.array_equal(np.asarray(point["action"], dtype=np.float32), actions[i - 1]),
                                   "executed action mismatch")
                    is_last_stop = status == "stopped" and i == steps
                    checks.require(bool(point["terminated"] or point["truncated"]) == is_last_stop,
                                   "stop flag mismatch")
            if trajectory:
                checks.require(trajectory[0]["rgb"]["sha256"] == root["root_rgb_sha256"],
                               "root RGB mismatch")
            if len(trajectory) >= 6:
                signature = [(p["qpos"], p["qvel"], p["rgb"]["sha256"]) for p in trajectory[:6]]
                if key[0] in prefix_references:
                    checks.require(prefix_references[key[0]] == signature, "same prefix differs by tail")
                else:
                    prefix_references[key[0]] = signature
            outcomes[key] = {
                "prefix": key[0], "tail": key[1], "status": status, "returned_steps": steps,
                "initial": trajectory[0]["qpos_distance"] if trajectory else None,
                "cp5": trajectory[5]["qpos_distance"] if len(trajectory) >= 6 else None,
                "final": trajectory[15]["qpos_distance"] if len(trajectory) == 16 else None,
            }
        for i in range(13):
            for j in range(5):
                current = outcomes[(i, j)]
                neutral = outcomes[(0, j)]
                own_zero = outcomes[(i, 0)]
                base_zero = outcomes[(0, 0)]
                complete = all(x["final"] is not None for x in (current, neutral, own_zero, base_zero))
                paired.append({"seed": root["seed"], **current,
                               "final_minus_neutral": current["final"] - neutral["final"] if complete else None,
                               "adjusted_final_difference":
                                   ((current["final"] - own_zero["final"]) -
                                    (neutral["final"] - base_zero["final"])) if complete else None})
    return {"status": "saved_consistency_known_root_development", "counts": counts,
            "paired_rows": paired, "P1_admission": False, "independent_confirmation": False,
            "scope": "no independent physical replay; 130 branches share two roots"}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha256", "journal", "journal-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan_path, journal_path = Path(args.plan), Path(args.journal)
    checks.require(checks.sha(plan_path) == args.plan_sha256 and checks.sha(journal_path) == args.journal_sha256,
                   "input hash mismatch")
    plan = json.loads(plan_path.read_text())
    with journal_path.open() as stream:
        events = [json.loads(line) for line in stream]
    result = validate(plan, events, args.plan_sha256)
    result.update(plan_sha256=args.plan_sha256, journal_sha256=args.journal_sha256)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "sha256": checks.sha(args.output)}))


if __name__ == "__main__":
    main()
