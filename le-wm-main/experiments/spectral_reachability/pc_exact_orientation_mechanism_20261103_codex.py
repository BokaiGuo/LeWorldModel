"""Known-dynamics orientation experiment; no RGB or confirmation claims."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_exact_unicycle_development as base
from experiments.spectral_reachability import pc_exact_unicycle_strong_baseline as strong

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_EXACT_ORIENTATION_MECHANISM_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_orientation_mechanism_20261103_codex.json"


def error(state, goal):
    return float(np.linalg.norm(state[:2] - goal))


def restricted(goal):
    start = time.monotonic()
    candidates = []
    for angle in np.linspace(-base.TURN, base.TURN, 9):
        prefix = np.array([0., angle])
        successor = base.step(np.zeros(3), prefix)
        eps = 1e-5
        plus = base.step(successor, [eps, 0.])
        minus = base.step(successor, [-eps, 0.])
        response = (plus[:2] - minus[:2]) / (2 * eps)
        _, singular, vh = np.linalg.svd(response.reshape(2, 1), full_matrices=False)
        if singular[0] <= 1e-12:
            raise ValueError("degenerate translation response")
        direction = float(vh[0, 0])
        magnitude = np.clip(np.dot(goal - successor[:2], response * direction)
                            / (singular[0] ** 2), -1., 1.)
        suffix = np.array([magnitude * direction, 0.])
        chain = np.stack([prefix, suffix])
        final, trace = base.evaluate(chain, goal)
        candidates.append({"chain": chain.tolist(), "error": final,
                           "response": response.tolist(), "trace": trace.tolist()})
    return min(candidates, key=lambda x: x["error"]), {
        "dynamics_transitions": 45, "svd_calls": 9, "objective_evaluations": 9,
        "wall_seconds": time.monotonic() - start}


def cem(goal, seed):
    start = time.monotonic()
    actions, ledger = base.search(goal, seed=seed, cem=True)
    cost, trace = base.evaluate(actions, goal)
    return {"actions": actions.tolist(), "error": cost, "trace": trace.tolist(),
            "ledger": {**ledger, "wall_seconds": time.monotonic() - start}}


def run():
    rows = []
    for i in range(16):
        angle = 2 * np.pi * (i + .37) / 16
        goal = .9 * np.array([np.cos(angle), np.sin(angle)])
        selected, structured_cost = restricted(goal)
        chain = np.asarray(selected["chain"])
        zero_suffix = np.zeros((2, 2))
        zero_prefix_same_suffix = np.stack([np.zeros(2), chain[1]])
        zero_prefix_optimized = np.stack([np.zeros(2), [np.clip(goal[0], -1., 1.), 0.]])
        zero_error, _ = base.evaluate(zero_suffix, goal)
        same_error, _ = base.evaluate(zero_prefix_same_suffix, goal)
        zero_opt_error, _ = base.evaluate(zero_prefix_optimized, goal)
        results = {}
        for budget in (8, 16, 32):
            start = time.monotonic()
            actions, ledger = strong.multistart_gradient(goal, budget)
            elapsed = time.monotonic() - start
            value, trace = base.evaluate(actions, goal)
            results[str(budget)] = {"actions": actions.tolist(), "error": value,
                                    "trace": trace.tolist(), "ledger": {**ledger, "wall_seconds": elapsed}}
        random_controls = [cem(goal, 2026110700 + 4 * i + j) for j in range(4)]
        direct = results["32"]
        choose_structured = selected["error"] < direct["error"] - 1e-12
        actual = selected["chain"] if choose_structured else direct["actions"]
        _, trace = base.evaluate(actual, goal)
        nonprogress = error(trace[1], goal) >= error(trace[0], goal) - 1e-12
        active_selected = selected["error"] - zero_error
        active_zero = same_error - zero_error
        witness = (active_selected < active_zero - 1e-12
                   and selected["error"] < zero_opt_error - 1e-12)
        rows.append({"index": i, "goal": goal.tolist(), "structured": selected,
                     "structured_ledger": structured_cost, "gradient": results,
                     "cem_seeds": random_controls, "zero_suffix_error": zero_error,
                     "common_suffix_zero_prefix_error": same_error,
                     "zero_prefix_optimized_error": zero_opt_error,
                     "active_selected_change": active_selected, "active_zero_change": active_zero,
                     "restricted_witness": witness, "combined_selected": "structured" if choose_structured else "gradient",
                     "combined_actions": actual, "combined_nonprogress": bool(nonprogress),
                     "combined_witness": bool(choose_structured and nonprogress and witness)})
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    print("Exact unicycle: restricted witness vs full-action optimization; not RGB validation.", flush=True)
    rows = run()
    summary = {"roots": len(rows), "restricted_witnesses": sum(x["restricted_witness"] for x in rows),
               "combined_structured_selected": sum(x["combined_selected"] == "structured" for x in rows),
               "combined_witnesses": sum(x["combined_witness"] for x in rows),
               "gradient32_wins": sum(x["gradient"]["32"]["error"] < x["structured"]["error"] - 1e-12 for x in rows),
               "best_cem32_wins": sum(min(c["error"] for c in x["cem_seeds"]) < x["structured"]["error"] - 1e-12 for x in rows)}
    report = {"scope": "synthetic exact dynamics only; diagnostic and development",
              "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
              "gradient_sha256": hashlib.sha256(Path(strong.__file__).read_bytes()).hexdigest(),
              "summary": summary, "rows": rows}
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
