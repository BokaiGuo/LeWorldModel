"""Independently validate the sealed fixed-pair physical replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_fixed_pair_rgb_replay_20261102 as replay
from . import audit_reacher_pc_p1_fd_common_physical_v2_20261102_codexexclusive as checks


def validate(plan, events, digest):
    need = checks.require
    need(plan["schema"] == "reacher_fixed_pair_rgb_replay_v1" and plan["expected_branches"] == 52,
         "plan schema")
    need(events[0] == {"kind": "start", "plan_sha256": digest}
         and events[-1] == {"kind": "finished", "plan_sha256": digest}, "journal framing")
    cursor, rows = 1, []
    for root in plan["roots"]:
        need(events[cursor] == {"kind": "root_verified", "seed": root["seed"]}, "root event")
        cursor += 1
        signature = None
        for index, branch in enumerate(root["branches"]):
            actions = checks.actions(branch["actions"], (15, 2))
            need(checks.action_hash(actions) == branch["actions_sha256"], "action digest")
            need(events[cursor] == {"kind": "branch_started", "seed": root["seed"], "index": index,
                                    "actions_sha256": branch["actions_sha256"]}, "branch start")
            cursor += 1
            for step in range(15):
                need(events[cursor] == {"kind": "step_intent", "seed": root["seed"], "index": index,
                                        "step": step, "action": branch["actions"][step]}
                     and events[cursor+1] == {"kind": "step_returned", "seed": root["seed"],
                                              "index": index, "step": step, "terminated": False,
                                              "truncated": False}, "action/stop event")
                cursor += 2
            entry = events[cursor]
            cursor += 1
            need(entry["kind"] == "branch_returned" and entry["seed"] == root["seed"]
                 and entry["index"] == index, "branch result")
            result = entry["outcome"]
            need(result["status"] == "complete" and result["stop"] is None
                 and result["attempted_steps"] == result["returned_steps"] == 15
                 and result["identity"] == {"reset_rgb_sha256": root["root_rgb_sha256"],
                                             "goal_rgb_sha256": root["goal_rgb_sha256"]}
                 and len(result["trajectory"]) == 16, "branch completeness")
            points = result["trajectory"]
            for i, point in enumerate(points):
                rgb = checks.verify_point(point)
                need(point["target_qpos"] == points[0]["target_qpos"], "target drift")
                if i == 0:
                    need(rgb == root["root_rgb_sha256"], "root RGB")
                    current = (point["qpos"], point["qvel"], point["target_qpos"])
                    if signature is None:
                        signature = current
                    need(signature == current, "root state drift")
                else:
                    need(np.array_equal(np.asarray(point["action"], np.float32), actions[i-1])
                         and not point["terminated"] and not point["truncated"]
                         and np.isfinite(point["reward"]), "trajectory action")
                if i == 5:
                    need(rgb == branch["cp5_rgb_sha256"], "cp5 RGB mismatch")
            final = points[15]["qpos_distance"]
            need(abs(final - branch["source_final_distance"]) <= plan["distance_tolerance"],
                 "final distance mismatch")
            rows.append({"seed": root["seed"], "prefix": branch["prefix"], "label": branch["label"],
                         "actions_sha256": branch["actions_sha256"], "cp5_rgb_sha256": branch["cp5_rgb_sha256"],
                         "final_rgb_sha256": checks.rgb(points[15]["rgb"]), "final_distance": final})
    need(cursor == len(events)-1 and len(rows) == 52, "event or branch count")
    return {"schema": "reacher_fixed_pair_rgb_replay_saved_v1", "status": "saved_consistency_complete",
            "branches": len(rows), "rows": rows, "P1_admission": False,
            "limits": ["two exposed roots; not independent confirmation",
                       "saved trace consistency, not independent simulator attestation"]}


def main():
    parser = argparse.ArgumentParser()
    for key in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    plan, journal = Path(args.plan), Path(args.journal)
    checks.require(checks.sha(plan) == args.plan_sha and checks.sha(journal) == args.journal_sha, "input hash")
    data = json.loads(plan.read_text())
    checks.require(data == replay.prepare(), "source plan drift")
    result = validate(data, replay.pair.audit.saved.verify_journal(journal), args.plan_sha)
    result.update(plan_sha256=args.plan_sha, journal_sha256=args.journal_sha,
                  verifier_sha256=checks.sha(__file__))
    replay.host.write_new(args.output, result)
    print(json.dumps({"branches": len(result["rows"]), "sha256": checks.sha(args.output)}))


if __name__ == "__main__":
    main()
