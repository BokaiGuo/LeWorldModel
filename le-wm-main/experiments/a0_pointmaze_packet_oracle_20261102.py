"""Offline fixed-bank PointMaze oracle at the declared five-action model packet."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import a0_pointmaze_prefix_terminal_oracle_20261102_codex_exclusive as earlier
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

ROOT = Path(__file__).resolve().parents[1]
NAME = "a0_pointmaze_packet_oracle_20261102"
PREVIOUS = ROOT / "results/a0_pointmaze_prefix_terminal_oracle_20261102_codex_exclusive_RESULT01.json"
PREVIOUS_SHA = "002abb3fc1e431ea6273e5c466fb4aa91fdd8866684cee394ddd674292a472d7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluate(inputs, replay):
    n = len(inputs[0]["bank"])
    require(n == 32 and all(len(item["bank"]) == n for item in inputs), "bank")
    rows = []
    # A separate batch per root bounds memory and keeps each complete root.
    for item in inputs:
        bank = item["bank"]
        states1 = np.repeat(item["source_state"][None], n, axis=0)
        sequences1 = np.repeat(bank[:, None, :], 5, axis=1)
        first = np.repeat(bank, n, axis=0)
        second = np.tile(bank, (n, 1))
        sequences2 = np.concatenate((np.repeat(first[:, None], 5, axis=1),
                                     np.repeat(second[:, None], 5, axis=1)), axis=1)
        states2 = np.repeat(item["source_state"][None], n*n, axis=0)
        p = np.asarray(replay(states1, sequences1), dtype=float)
        t = np.asarray(replay(states2, sequences2), dtype=float)
        require(p.shape == states1.shape and t.shape == states2.shape
                and np.isfinite(p).all() and np.isfinite(t).all(), "endpoints")
        p, t = p[:, :2], t[:, :2].reshape(n, n, 2)
        prefix_cost = np.linalg.norm(p-item["target"], axis=-1)
        terminal_cost = np.linalg.norm(t-item["target"], axis=-1)
        prefix_index = int(np.argmin(prefix_cost))
        i, j = np.unravel_index(int(np.argmin(terminal_cost)), terminal_cost.shape)
        tail = int(np.argmin(terminal_cost[prefix_index]))
        rows.append({"root": item["root"], "source_state_privileged": item["source_state"].tolist(),
                     "target": item["target"].tolist(), "candidate_bank": bank.tolist(),
                     "candidate_bank_sha256": item["bank_sha256"],
                     "prefix_positions": p.tolist(), "terminal_positions": t.tolist(),
                     "prefix_selected_index": prefix_index, "terminal_selected_indices": [int(i), int(j)],
                     "prefix_selected_best_terminal_tail_index": tail,
                     "prefix_cost_at_prefix_choice": float(prefix_cost[prefix_index]),
                     "prefix_cost_at_terminal_choice": float(prefix_cost[i]),
                     "terminal_cost_at_terminal_choice": float(terminal_cost[i, j]),
                     "terminal_cost_with_prefix_first_best_tail": float(terminal_cost[prefix_index, tail]),
                     "first_action_differs": bool(i != prefix_index)})
    return rows


def build():
    require(earlier.digest(earlier.SOURCE) == earlier.SOURCE_SHA
            and earlier.digest(PREVIOUS) == PREVIOUS_SHA, "sealed source")
    source = json.loads(earlier.SOURCE.read_bytes())
    inputs = earlier.extract_inputs(source)
    receipt = physical.position_backend_receipt()
    rows = evaluate(inputs, physical.replay_position_endpoints)
    return {"schema": NAME + ".v1",
            "source": {"strict_sha256": earlier.SOURCE_SHA, "previous_sha256": PREVIOUS_SHA,
                       "position_backend_receipt": receipt, "code_sha256": earlier.digest(__file__)},
            "protocol": {"roots": list(earlier.ROOTS), "previously_opened": True,
                         "actions": "one candidate repeated for five archive actions per packet",
                         "mujoco_ticks_per_archive_action": 5, "packet_archive_actions": 5,
                         "horizons_archive_actions": [5, 10],
                         "search": "exact first packet and 32x32 two-packet fixed-bank oracle",
                         "target": "source active waypoint, not global task success",
                         "state": "privileged qpos and qvel; offline position oracle only",
                         "normalizer": "not applicable; no model inference", "fresh_roots": False,
                         "replanning_or_closed_loop": False, "full_state_equivalence": "not certified"},
            "rows": rows, "summary": {"roots": len(rows),
                                      "different_first_action": sum(r["first_action_differs"] for r in rows),
                                      "physical_sequences": len(rows) * (32 + 32*32)},
            "p1_admission": False, "scientific_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent == ROOT / "results"
            and args.output.name.startswith(NAME + "_") and args.output.suffix == ".json", "exclusive output")
    print("Evaluating 5/10 archive-action PointMaze oracle; offline simulator, no neural inference.", flush=True)
    result = build()
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "summary": result["summary"]}))


if __name__ == "__main__":
    main()
