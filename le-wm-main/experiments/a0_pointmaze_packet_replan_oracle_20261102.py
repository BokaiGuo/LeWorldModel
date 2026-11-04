"""Compare privileged fixed-bank physical packet replanning on known roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/a0_pointmaze_archive_bank_alignment_20261102_DEV01.json"
SOURCE_SHA = "93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291"
RECEIPT = ROOT / "results/a0_pointmaze_archive_bank_validate_20261102_DEV01.json"
RECEIPT_SHA = "60f2bfb51cdcaa58373cf99a3bd1fd530e0e9e96d306134659f662cc6e932d14"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PACKET_REPLAN_ORACLE_PROTOCOL_20261102.md"
ROOTS = (1880, 1881, 1882, 1883)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def action_sequences(bank):
    bank = np.asarray(bank, dtype=np.float64)
    require(bank.shape == (8, 2) and np.isfinite(bank).all() and np.max(np.abs(bank)) <= 1, "bank")
    one = np.repeat(bank[:, None, :], 5, axis=1)
    two = np.concatenate((np.repeat(one, 8, axis=0), np.tile(one, (8, 1, 1))), axis=1)
    return one, two


def evaluate(source, replay):
    bank = np.asarray(source["protocol"]["bank"], dtype=np.float32).astype(np.float64)
    one, two = action_sequences(bank)
    rows = []
    for row in source["rows"]:
        state = np.asarray(row["state_start_evaluation_only"], np.float64)
        target = np.asarray(row["goal_position_evaluation_only"], np.float64)
        require(state.shape == (4,) and target.shape == (2,) and np.isfinite(state).all()
                and np.isfinite(target).all(), "state/target")

        def batch(start, actions):
            states = np.repeat(np.asarray(start)[None], len(actions), axis=0)
            result = np.asarray(replay(states, actions), np.float64)
            require(result.shape == states.shape and np.isfinite(result).all(), "physical endpoint")
            return result

        first = batch(state, one)
        joint = batch(state, two)
        require(np.allclose(first[:, :2], np.asarray(row["physical_prefix_positions"]), atol=1e-10, rtol=0)
                and np.allclose(joint[:, :2].reshape(8, 8, 2),
                                np.asarray(row["physical_terminal_positions"]), atol=1e-10, rtol=0),
                "sealed physical source mismatch")
        cost1 = np.linalg.norm(first[:, :2]-target, axis=1)
        cost2 = np.linalg.norm(joint[:, :2]-target, axis=1).reshape(8, 8)
        greedy = int(np.argmin(cost1))
        joint_i, joint_j = np.unravel_index(int(np.argmin(cost2)), (8, 8))
        greedy_next = batch(first[greedy], one)
        greedy_cost = np.linalg.norm(greedy_next[:, :2]-target, axis=1)
        greedy_j = int(np.argmin(greedy_cost))
        lookahead_next = batch(first[joint_i], two)
        lookahead_cost = np.linalg.norm(lookahead_next[:, :2]-target, axis=1).reshape(8, 8)
        next_i, next_j = np.unravel_index(int(np.argmin(lookahead_cost)), (8, 8))
        # Execute only the selected first packet, not its optimized lookahead tail.
        actual = batch(first[joint_i], one)
        zero = batch(state, np.zeros((1, 10, 2), dtype=np.float64))[0]
        roots = {
            "joint_open": {"packets": [int(joint_i), int(joint_j)], "state5": first[joint_i].tolist(),
                           "state10": joint[joint_i*8+joint_j].tolist(),
                           "search_rollouts": 64, "search_archive_steps": 640},
            "joint_replan": {"packets": [int(joint_i), int(next_i)], "lookahead_second": int(next_j),
                             "state5": first[joint_i].tolist(), "state10": actual[next_i].tolist(),
                             "search_rollouts": 128, "search_archive_steps": 1280},
            "greedy_replan": {"packets": [greedy, greedy_j], "state5": first[greedy].tolist(),
                              "state10": greedy_next[greedy_j].tolist(),
                              "search_rollouts": 16, "search_archive_steps": 80},
            "zero": {"packets": ["zero", "zero"], "state10": zero.tolist(),
                     "search_rollouts": 0, "search_archive_steps": 0},
        }
        for item in roots.values():
            item["position_distance_at_10"] = float(np.linalg.norm(np.asarray(item["state10"])[:2]-target))
            item["execution_archive_steps"] = 10
        rows.append({"root": row["root"], "initial_state": state.tolist(), "goal": target.tolist(),
                     "first_endpoints": first.tolist(), "joint_endpoints": joint.tolist(),
                     "greedy_successor_endpoints": greedy_next.tolist(),
                     "joint_successor_first_endpoints": actual.tolist(),
                     "joint_successor_lookahead_endpoints": lookahead_next.tolist(),
                     "first_costs": cost1.tolist(), "joint_costs": cost2.tolist(),
                     "greedy_successor_costs": greedy_cost.tolist(),
                     "joint_successor_lookahead_costs": lookahead_cost.tolist(), "policies": roots})
        print(json.dumps({"root": row["root"], "distance_at_10":
                          {name: item["position_distance_at_10"] for name, item in roots.items()}}), flush=True)
    return rows


def build():
    require(sha(SOURCE) == SOURCE_SHA and sha(RECEIPT) == RECEIPT_SHA, "source/receipt hashes")
    source = json.loads(SOURCE.read_text())
    require(tuple(r["root"] for r in source["rows"]) == ROOTS, "root order")
    require(source["protocol"]["packet"] == "five repetitions of one physical candidate; action normalized once",
            "packet contract")
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(args.output.is_absolute() and args.output.parent == ROOT / "results"
            and args.output.name.startswith("a0_pointmaze_packet_replan_oracle_") and not args.output.exists(),
            "exclusive output")
    source = build()
    started = time.monotonic()
    print("Evaluating fixed-bank privileged position-only packet replanning on four exposed archive roots.",
          flush=True)
    rows = evaluate(source, physical.replay_position_endpoints)
    result = {"schema": "a0_pointmaze_packet_replan_oracle_v1", "source_sha256": SOURCE_SHA,
              "source_receipt_sha256": RECEIPT_SHA, "protocol_sha256": sha(PROTOCOL),
              "runner_sha256": sha(__file__), "physical_backend": physical.position_backend_receipt(),
              "scope": "four exposed roots; privileged position-only simulator; no neural calls",
              "rows": rows, "wall_seconds": time.monotonic()-started, "P1_admission": False}
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(), "roots": len(rows)}))


if __name__ == "__main__":
    main()
