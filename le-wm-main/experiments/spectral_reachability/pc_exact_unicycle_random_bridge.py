"""Matched five-candidate random-turn ablation on the sealed unicycle panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_exact_unicycle_development as base


def random_bridge(goal, seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-base.TURN, base.TURN, size=5)
    candidates = []
    for turn in angles:
        first = np.array([0., turn])
        successor = base.step(np.zeros(3), first)
        response = np.array([np.cos(turn), np.sin(turn)])
        v = np.clip(np.dot(goal - successor[:2], response), -1., 1.)
        program = np.stack([first, np.array([v, 0.])])
        cost = np.linalg.norm(base.step(successor, program[1])[:2] - goal)
        candidates.append((float(cost), program))
    index = int(np.argmin([cost for cost, _ in candidates]))
    return candidates[index][1], {
        "turns": angles.tolist(), "selected_index": index, "transitions": 10,
        "objective_evaluations": 5, "response_evaluations": 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source.read_text())
    if len(source["rows"]) != 64:
        raise ValueError("expected sealed 64-goal panel")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in source["rows"]:
        goal = np.asarray(row["goal"])
        program, ledger = random_bridge(goal, 91200 + row["index"])
        error, _ = base.evaluate(program, goal)
        rows.append({"index": row["index"], "goal": row["goal"], "actions": program.tolist(),
                     "ledger": ledger, "random_bridge_error": error,
                     "fixed_bridge_error": row["methods"]["structured"]["final_error"]})
    diff = np.array([row["fixed_bridge_error"] - row["random_bridge_error"] for row in rows])
    report = {"scope": "known synthetic development roots only; not a learned-model or PC gate",
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summary": {"count": len(rows),
                          "random_mean": float(np.mean([r["random_bridge_error"] for r in rows])),
                          "fixed_mean": float(np.mean([r["fixed_bridge_error"] for r in rows])),
                          "fixed_wins": int(np.sum(diff < -1e-12)),
                          "ties": int(np.sum(np.abs(diff) <= 1e-12)),
                          "random_wins": int(np.sum(diff > 1e-12))}, "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
