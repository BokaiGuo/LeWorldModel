"""Saved-only verification of the full development physical replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import audit_reacher_pc_p1_fd_common_physical_v2_20261102_codexexclusive as checks
from . import reacher_cem_context_physical_20261102 as experiment


def validate(plan, events, plan_sha):
    need = checks.require
    need(plan["schema"] == "reacher_cem_context_physical_development_v1"
         and plan["expected_branches"] == 48 and len(plan["roots"]) == 2, "plan")
    need(events[0] == {"kind": "start", "plan_sha256": plan_sha}
         and events[-1] == {"kind": "finished", "plan_sha256": plan_sha}, "framing")
    cursor, paired = 1, []
    for root in plan["roots"]:
        need(events[cursor] == {"kind": "root_verified", "seed": root["seed"]}, "root")
        cursor += 1
        root_signature, prefix_signatures, outcomes = None, {}, []
        for index, branch in enumerate(root["branches"]):
            actions = checks.actions(branch["raw_actions"], (15, 2))
            need(checks.action_hash(actions) == branch["actions_sha256"], "action hash")
            need(events[cursor] == {"kind": "branch_started", "seed": root["seed"], "branch_index": index,
                                    "actions_sha256": branch["actions_sha256"]}, "branch start")
            cursor += 1
            for step in range(15):
                intent, returned = events[cursor:cursor + 2]
                cursor += 2
                need(intent == {"kind": "step_intent", "seed": root["seed"], "branch_index": index,
                                "step": step, "action": branch["raw_actions"][step]}
                     and returned == {"kind": "step_returned", "seed": root["seed"], "branch_index": index,
                                      "step": step, "terminated": False, "truncated": False},
                     "step event, stop, or action")
            event = events[cursor]
            cursor += 1
            need(event["kind"] == "branch_returned" and event["seed"] == root["seed"]
                 and event["branch_index"] == index, "branch return")
            outcome = event["outcome"]
            need(outcome["status"] == "complete" and outcome["attempted_steps"] == 15
                 and outcome["returned_steps"] == 15 and outcome["stop"] is None
                 and outcome["identity"] == {"reset_rgb_sha256": root["root_rgb_sha256"],
                                              "goal_rgb_sha256": root["goal_rgb_sha256"]}
                 and len(outcome["trajectory"]) == 16, "incomplete, stopped or invalid branch")
            points = outcome["trajectory"]
            for i, point in enumerate(points):
                digest = checks.verify_point(point)
                if i == 0:
                    need(digest == root["root_rgb_sha256"], "root RGB")
                    signature = (point["qpos"], point["qvel"], point["target_qpos"])
                    if root_signature is None:
                        root_signature = signature
                    need(signature == root_signature, "root state")
                else:
                    need(np.array_equal(np.asarray(point["action"], np.float32), actions[i-1])
                         and point["terminated"] is False and point["truncated"] is False
                         and np.isfinite(point["reward"])
                         and point["target_qpos"] == points[0]["target_qpos"], "point binding")
            prefix_signature = [(point["qpos"], point["qvel"], point["rgb"]["sha256"]) for point in points[:6]]
            prefix = branch["prefix_index"]
            if prefix in prefix_signatures:
                need(prefix_signatures[prefix] == prefix_signature, "same prefix replay")
            else:
                prefix_signatures[prefix] = prefix_signature
            outcomes.append({"context": branch["context"], "initial": points[0]["qpos_distance"],
                             "cp5": points[5]["qpos_distance"], "final": points[15]["qpos_distance"],
                             "actions_sha256": branch["actions_sha256"], "status": outcome["status"]})
            if index % 2:
                prior = root["branches"][index - 1]
                need(prior["context"] == "predicted" and branch["context"] == "observed"
                     and all(prior[k] == branch[k] for k in ("prefix_index", "budget", "cem_seed")),
                     "pair binding")
                predicted, observed = outcomes[-2:]
                paired.append({"seed": root["seed"], "prefix": prefix, "budget": branch["budget"],
                               "cem_seed": branch["cem_seed"], "predicted": predicted, "observed": observed,
                               "observed_minus_predicted": observed["final"] - predicted["final"]})
    need(cursor == len(events) - 1 and len(paired) == 24, "event denominator")
    return {"status": "saved_consistency_complete", "complete_branches": 48, "pairs": paired,
            "independent_confirmation": False, "P1_admission": False,
            "limits": ["known roots and model outcomes already exposed",
                       "offline replay of saved tails, not online feedback or equal-budget baseline",
                       "no independent simulator attestation; saved trajectory consistency only",
                       "CPU-decoded tails not certified bitwise identical to CUDA decoding"]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha256", "journal", "journal-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan_path, journal_path = Path(args.plan), Path(args.journal)
    checks.require(checks.sha(plan_path) == args.plan_sha256 and checks.sha(journal_path) == args.journal_sha256,
                   "input hashes")
    plan = json.loads(plan_path.read_text())
    checks.require(plan == experiment.prepare(), "plan/source drift")
    result = validate(plan, experiment.run.saved.verify_journal(journal_path), args.plan_sha256)
    result.update(plan_sha256=args.plan_sha256, journal_sha256=args.journal_sha256,
                  consumer_sha256=checks.sha(__file__))
    experiment.host.write_new(args.output, result)
    print(json.dumps({"status": result["status"], "complete": result["complete_branches"],
                      "sha256": checks.sha(args.output)}))


if __name__ == "__main__":
    main()
