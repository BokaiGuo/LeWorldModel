"""Verify stored actual-RGB scores against sealed physical action chains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_cem32_same_chain_rgb_20261102 as rgb

host, setup = rgb.host, rgb.setup


def validate(plan, events, physical_plan, physical_events, plan_sha):
    need = host.require
    need(len(events) == 8 and events[0]["kind"] == "start"
         and events[0]["plan_sha256"] == plan_sha
         and events[-1]["kind"] == "finished", "event framing")
    need(events[-1]["model_before"] == events[-1]["model_after"] == plan["model_hash"], "model identity")
    rows = []
    cursor = 1
    for root in physical_plan["roots"]:
        for index, branch in enumerate(root["branches"]):
            event = events[cursor]
            cursor += 1
            matches = [e for e in physical_events if e["kind"] == "branch_returned"
                       and e["seed"] == root["seed"] and e["branch_index"] == index]
            need(len(matches) == 1, "physical branch")
            points = matches[0]["outcome"]["trajectory"]
            need(event["kind"] == "arm" and event["seed"] == root["seed"]
                 and event["role"] == branch["role"] and event["prefix"] == branch["prefix_index"]
                 and event["actions_sha256"] == branch["actions_sha256"]
                 and event["frame_hashes"] == [points[i]["rgb"]["sha256"] for i in (5, 10, 15)]
                 and event["goal_sha256"] == root["goal_rgb_sha256"]
                 and event["physical_final"] == points[15]["qpos_distance"]
                 and event["predicted_cost"] == branch["predicted_terminal_cost"]
                 and event["encoder_calls"] == 2, "arm binding")
            history = np.asarray(event["raw_history"], dtype=np.float32)
            need(history.shape == (3, 10)
                 and np.array_equal(history.reshape(15, 2), np.asarray(branch["raw_actions"], np.float32)),
                 "action history")
            actual = np.asarray(event["actual_coordinate"], dtype=np.float32)
            target = np.asarray(event["goal_coordinate"], dtype=np.float32)
            need(actual.shape == target.shape and actual.size > 0
                 and np.isfinite(actual).all() and np.isfinite(target).all(), "coordinates")
            recomputed = float(np.square(actual - target).mean())
            need(np.isclose(recomputed, event["actual_cost"], rtol=2e-6, atol=1e-7), "latent cost")
            rows.append({key: event[key] for key in
                         ("seed", "role", "prefix", "predicted_cost", "actual_cost", "physical_final")})
    need(cursor == 7 and len(rows) == plan["expected_arms"], "arm denominator")
    return {"status": "saved_consistency_complete", "rows": rows, "encoder_calls_reported": 12,
            "independent_confirmation": False, "P1_admission": False,
            "limits": ["two exposed development roots", "saved data do not attest simulator execution",
                       "CPU-decoded replay tails are not certified bitwise equal to CUDA-scored tails",
                       "model prediction vectors were not saved; only scalar predictions can be compared"]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha256", "journal", "journal-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan_path, journal_path = Path(args.plan), Path(args.journal)
    host.require(setup._sha256_file(plan_path) == args.plan_sha256
                 and setup._sha256_file(journal_path) == args.journal_sha256, "input hashes")
    plan = json.loads(plan_path.read_text())
    host.require(plan == rgb.prepare(), "plan/source drift")
    events = rgb.exp.run.saved.verify_journal(journal_path)
    physical_plan = json.loads(rgb.PLAN.read_text())
    physical_events = rgb.exp.run.saved.verify_journal(rgb.JOURNAL)
    result = validate(plan, events, physical_plan, physical_events, args.plan_sha256)
    result.update(plan_sha256=args.plan_sha256, journal_sha256=args.journal_sha256,
                  consumer_sha256=setup._sha256_file(Path(__file__)))
    host.write_new(args.output, result)
    print(json.dumps({"status": result["status"], "sha256": setup._sha256_file(Path(args.output)),
                      "rows": result["rows"]}))


if __name__ == "__main__":
    main()
