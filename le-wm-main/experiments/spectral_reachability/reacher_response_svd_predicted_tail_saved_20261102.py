"""Saved-only verification of the known-root predicted-tail physical run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import audit_reacher_pc_p1_fd_common_physical_v2_20261102_codexexclusive as checks
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as journal
from . import reacher_response_svd_predicted_cem_physical_20261102 as physical


def validate(plan, events, plan_sha):
    require = checks.require
    require(plan["schema"] == "response_svd_predicted_tail_physical_v1"
            and plan["expected_branches"] == 4 and len(plan["roots"]) == 2, "plan schema")
    require(events[0] == {"kind": "start", "plan_sha256": plan_sha}
            and events[-1] == {"kind": "finished", "plan_sha256": plan_sha}, "journal framing")
    cursor, results = 1, []
    counts = {"complete": 0, "stopped": 0, "error": 0}
    for root in plan["roots"]:
        require(events[cursor] == {"kind": "root_verified", "seed": root["seed"]}, "root event")
        cursor += 1
        require([b["role"] for b in root["branches"]] == ["selected", "neutral"], "branch roster")
        root_state = None
        for branch in root["branches"]:
            start, row = events[cursor:cursor + 2]
            cursor += 2
            require(start == {"kind": "branch_started", "seed": root["seed"], "role": branch["role"],
                              "actions_sha256": branch["actions_sha256"]}, "start binding")
            require(row["kind"] == "branch_returned" and row["seed"] == root["seed"]
                    and all(row[k] == v for k, v in branch.items()), "branch binding")
            actions = checks.actions(branch["raw_actions"], (15, 2))
            require(checks.action_hash(actions) == branch["actions_sha256"], "action hash")
            outcome = row["outcome"]
            status = outcome["status"]
            require(status in counts, "status")
            counts[status] += 1
            steps, attempted, trajectory = outcome["returned_steps"], outcome["attempted_steps"], outcome["trajectory"]
            require(type(steps) is int and type(attempted) is int and 0 <= steps <= attempted <= 15
                    and len(trajectory) in (0, steps + 1), "trajectory count")
            if status != "error":
                require(outcome["identity"] == {"reset_rgb_sha256": root["root_rgb_sha256"],
                                                "goal_rgb_sha256": root["goal_rgb_sha256"]}
                        and steps == attempted and len(trajectory) == steps + 1, "identity or count")
            if status == "complete":
                require(steps == 15 and outcome["stop"] is None, "complete contract")
            if status == "stopped":
                stop = outcome["stop"]
                require(steps > 0 and stop is not None and stop["step"] == steps
                        and (stop["terminated"] or stop["truncated"]), "stop contract")
            for index, point in enumerate(trajectory):
                image_sha = checks.verify_point(point)
                if index == 0:
                    require(image_sha == root["root_rgb_sha256"], "root RGB")
                    signature = (point["qpos"], point["qvel"], point["target_qpos"])
                    if root_state is None:
                        root_state = signature
                    else:
                        require(signature == root_state, "root state mismatch")
                else:
                    require(np.array_equal(np.asarray(point["action"], np.float32), actions[index - 1])
                            and np.isfinite(point["reward"])
                            and point["target_qpos"] == trajectory[0]["target_qpos"], "point binding")
                    terminated, truncated = point["terminated"], point["truncated"]
                    require(type(terminated) is bool and type(truncated) is bool, "stop flag type")
                    is_stop = bool(terminated or truncated)
                    require(not is_stop or (status == "stopped" and index == steps), "unexpected stop")
                    if status == "stopped" and index == steps:
                        require(terminated == outcome["stop"]["terminated"]
                                and truncated == outcome["stop"]["truncated"], "stop flag binding")
            results.append({"seed": root["seed"], "role": branch["role"], "prefix_index": branch["prefix_index"],
                            "status": status, "attempted_steps": attempted, "returned_steps": steps,
                            "initial": trajectory[0]["qpos_distance"] if trajectory else None,
                            "cp5": trajectory[5]["qpos_distance"] if len(trajectory) > 5 else None,
                            "final": trajectory[15]["qpos_distance"] if len(trajectory) > 15 else None,
                            "stop": outcome["stop"], "actions_sha256": branch["actions_sha256"],
                            "prefix_cpu_decode_max_abs_error": branch["prefix_cpu_decode_max_abs_error"]})
    require(cursor == len(events) - 1 and sum(counts.values()) == 4, "extra or missing events")
    return {"status": "saved_consistency_known_root_development", "counts": counts, "rows": results,
            "independent_confirmation": False, "P1_admission": False,
            "limits": ["saved consistency only; no independent physical replay or simulator attestation",
                       "open-loop predicted tails, not real-RGB feedback replanning",
                       "branch-level journal does not durably record each step as executed",
                       "CPU-decoded predicted tail is not certified bitwise equal to CUDA decoding"]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha256", "journal", "journal-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan_path, journal_path = Path(args.plan), Path(args.journal)
    checks.require(checks.sha(plan_path) == args.plan_sha256 and checks.sha(journal_path) == args.journal_sha256,
                   "external SHA mismatch")
    plan = json.loads(plan_path.read_text())
    checks.require(plan == physical.make_plan(), "plan or source drift")
    result = validate(plan, journal.verify_journal(journal_path), args.plan_sha256)
    result.update(plan_sha256=args.plan_sha256, journal_sha256=args.journal_sha256,
                  consumer_sha256=checks.sha(__file__))
    physical.host.write_new(args.output, result)
    print(json.dumps({"counts": result["counts"], "sha256": checks.sha(args.output)}))


if __name__ == "__main__":
    main()
