"""Verify saved search decisions and replay each selected two-packet path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import a0_pointmaze_packet_replan_oracle_20261102 as exp
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical


def analyze(report, source, replay):
    need = exp.require
    need(report["schema"] == "a0_pointmaze_packet_replan_oracle_v1"
         and report["P1_admission"] is False and report["source_sha256"] == exp.SOURCE_SHA
         and report["source_receipt_sha256"] == exp.RECEIPT_SHA
         and report["protocol_sha256"] == exp.sha(exp.PROTOCOL)
         and report["runner_sha256"] == exp.sha(exp.__file__)
         and report["physical_backend"] == physical.position_backend_receipt(), "report binding")
    bank = np.asarray(source["protocol"]["bank"], np.float32).astype(np.float64)
    one, _ = exp.action_sequences(bank)
    need(tuple(x["root"] for x in report["rows"]) == exp.ROOTS
         and tuple(x["root"] for x in source["rows"]) == exp.ROOTS, "roster")
    saved = []
    for row, original in zip(report["rows"], source["rows"], strict=True):
        state, goal = np.asarray(row["initial_state"]), np.asarray(row["goal"])
        need(np.array_equal(state, original["state_start_evaluation_only"])
             and np.array_equal(goal, original["goal_position_evaluation_only"]), "source state")
        first = np.asarray(row["first_endpoints"])
        joint = np.asarray(row["joint_endpoints"])
        greedy_next = np.asarray(row["greedy_successor_endpoints"])
        next_first = np.asarray(row["joint_successor_first_endpoints"])
        look = np.asarray(row["joint_successor_lookahead_endpoints"])
        for value, shape in ((first, (8, 4)), (joint, (64, 4)), (greedy_next, (8, 4)),
                             (next_first, (8, 4)), (look, (64, 4))):
            need(value.shape == shape and np.isfinite(value).all(), "endpoint array")
        need(np.allclose(first[:, :2], original["physical_prefix_positions"], atol=1e-10, rtol=0)
             and np.allclose(joint[:, :2].reshape(8, 8, 2), original["physical_terminal_positions"],
                             atol=1e-10, rtol=0), "source endpoint")
        c1 = np.linalg.norm(first[:, :2]-goal, axis=1)
        c2 = np.linalg.norm(joint[:, :2]-goal, axis=1).reshape(8, 8)
        cg = np.linalg.norm(greedy_next[:, :2]-goal, axis=1)
        cl = np.linalg.norm(look[:, :2]-goal, axis=1).reshape(8, 8)
        for field, values in (("first_costs", c1), ("joint_costs", c2), ("greedy_successor_costs", cg),
                              ("joint_successor_lookahead_costs", cl)):
            need(np.allclose(row[field], values, atol=1e-12, rtol=1e-12), "cost arithmetic")
        i, j = np.unravel_index(int(np.argmin(c2)), (8, 8))
        gi, gj = int(np.argmin(c1)), int(np.argmin(cg))
        ni, nj = np.unravel_index(int(np.argmin(cl)), (8, 8))
        policies = row["policies"]
        need(policies["joint_open"]["packets"] == [int(i), int(j)]
             and policies["joint_replan"]["packets"] == [int(i), int(ni)]
             and policies["joint_replan"]["lookahead_second"] == int(nj)
             and policies["greedy_replan"]["packets"] == [gi, gj]
             and policies["zero"]["packets"] == ["zero", "zero"], "selection")
        need(np.array_equal(joint[i*8+j], next_first[j]), "same action reset equivalence")
        final_states = {"joint_open": joint[i*8+j], "joint_replan": next_first[ni],
                        "greedy_replan": greedy_next[gj]}
        for name, entry in policies.items():
            if name == "zero":
                sequence = np.zeros((10, 2), np.float64)
            else:
                a, b = entry["packets"]
                sequence = np.concatenate((one[a], one[b]), axis=0)
                need(np.array_equal(entry["state5"], first[a])
                     and np.array_equal(entry["state10"], final_states[name]), "saved endpoint")
            actual = np.asarray(replay(state[None], sequence[None]))
            need(actual.shape == (1, 4) and np.allclose(actual[0], entry["state10"], atol=1e-10, rtol=0)
                 and np.isclose(np.linalg.norm(actual[0, :2]-goal), entry["position_distance_at_10"],
                                atol=1e-12, rtol=1e-12)
                 and entry["execution_archive_steps"] == 10, "selected path replay")
        saved.append({"root": row["root"], "distances": {
            name: x["position_distance_at_10"] for name, x in policies.items()},
            "joint_open_split_state_max_abs": float(np.max(np.abs(joint[i*8+j]-next_first[j])))})
    return {"schema": "a0_pointmaze_packet_replan_oracle_saved_v1",
            "status": "saved_consistency_and_selected_replay_complete", "roots": saved,
            "selected_path_replays": 16, "P1_admission": False,
            "limits": ["known roots; position-only backend", "candidate library is finite",
                       "no native success or certified full-state generalization",
                       "joint_replan lookahead extends 5 actions beyond common 10-action evaluation"]}


def main():
    p = argparse.ArgumentParser()
    for key in ("report", "report-sha", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    report_path, output = Path(args.report), Path(args.output)
    exp.require(exp.sha(report_path) == args.report_sha and output.is_absolute()
                and output.parent == exp.ROOT / "results"
                and output.name.startswith("a0_pointmaze_packet_replan_oracle_saved_") and not output.exists(),
                "input/output")
    result = analyze(json.loads(report_path.read_text()), exp.build(), physical.replay_position_endpoints)
    result.update(report_sha256=args.report_sha, verifier_sha256=exp.sha(__file__))
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"roots": len(result["roots"]), "sha256": exp.sha(output)}))


if __name__ == "__main__":
    main()
