"""Prospectively specified rank-two nominal experiment; development only."""
import json
import time
from pathlib import Path

import numpy as np

from . import pc_response_program_joint_refinement_20261103_codex as joint
from . import d2p_goal_mode_rank_20261102 as modes

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_RANK2_NOMINAL_SIDE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_rank2_nominal_side_20261103.json"
METHODS = ("task", "gain", "ridge", "random", "task_greedy")
CAPS = (64, 128, 256)
DAMPING = 1e-4


def increment(matrix, residual, method):
    if method == "ridge":
        return np.linalg.solve(matrix.T @ matrix + DAMPING * np.eye(2),
                               matrix.T @ residual)
    if method not in ("task", "gain", "task_greedy"):
        raise ValueError("unknown response method")
    ranked = modes.rank_goal_modes(matrix, residual, DAMPING)
    index = int(ranked["order"][0]) if method != "gain" else 0
    return ranked["right_t"][index] * ranked["coefficients"][index]


def generate(goal, index, method):
    if method not in METHODS:
        raise ValueError("unknown method")
    counts = {"nominal": 0, "finite_difference": 0, "chain_score": 0,
              "prefix_score": 0, "svd": 0, "solve": 0}
    bank = []
    if method == "random":
        points = np.random.default_rng(2026111300 + index).uniform(
            [-1., -joint.dynamics.TURN, -1.], [1., joint.dynamics.TURN, 1.], (8, 3))
        for point in points:
            distance, _ = joint.dynamics.evaluate(joint.actions(point), goal)
            counts["chain_score"] += 1
            bank.append({"point": point.tolist(), "terminal_distance": distance})
    else:
        for turn in np.linspace(-joint.dynamics.TURN, joint.dynamics.TURN, 9):
            if abs(turn) < 1e-12:
                continue
            nominal = np.array([.25, -.25])
            end = joint.endpoint(nominal, turn)
            counts["nominal"] += 1
            columns = []
            for unit in np.eye(2):
                columns.append((joint.endpoint(nominal + 1e-5 * unit, turn)
                                - joint.endpoint(nominal - 1e-5 * unit, turn)) / 2e-5)
                counts["finite_difference"] += 2
            matrix = np.stack(columns, axis=1)
            velocity = np.clip(nominal + increment(matrix, goal - end, method), -1., 1.)
            counts["solve" if method == "ridge" else "svd"] += 1
            point = [float(velocity[0]), float(turn), float(velocity[1])]
            distance, _ = joint.dynamics.evaluate(joint.actions(point), goal)
            counts["chain_score"] += 1
            item = {"point": point, "terminal_distance": distance,
                    "response": matrix.tolist(), "nominal_endpoint": end.tolist()}
            if method == "task_greedy":
                state = joint.dynamics.step(np.zeros(3), joint.actions(point)[0])
                item["prefix_distance"] = float(np.linalg.norm(state[:2] - goal))
                counts["prefix_score"] += 1
            bank.append(item)
    return bank, counts


def plan(goal, index, cap, method):
    started = time.monotonic()
    bank, counts = generate(goal, index, method)
    key = "prefix_distance" if method == "task_greedy" else "terminal_distance"
    starts = sorted(bank, key=lambda x: x[key])[:4]
    generation = sum(counts[k] for k in ("nominal", "finite_difference", "chain_score", "prefix_score"))
    result = joint.refine(goal, [x["point"] for x in starts], cap, generation,
                          min(starts, key=lambda x: x["terminal_distance"]))
    result.update(method=method, bank=bank, starts=starts, generation_counts=counts,
                  generation_dynamics_steps=2 * (generation - counts["prefix_score"]) + counts["prefix_score"],
                  planning_seconds=time.monotonic() - started)
    # Diagnostic SVD occurs after selected action and planning time are sealed.
    result["diagnostic_response_singular_values"] = [
        np.linalg.svd(x["response"], compute_uv=False).tolist()
        for x in bank if "response" in x]
    result["diagnostic_svd_calls"] = len(result["diagnostic_response_singular_values"])
    replay, _ = joint.dynamics.evaluate(joint.actions(result["point"]), goal)
    if abs(replay - result["terminal_distance"]) > 1e-10:
        raise AssertionError("selected plan failed replay")
    return result


def summarize(rows):
    summary = {}
    for cap in CAPS:
        panel = {}
        for method in METHODS:
            items = [r for row in rows for r in row["results"]
                     if r["method"] == method and r["cap"] == cap]
            panel[method] = {
                "mean_distance": float(np.mean([x["terminal_distance"] for x in items])),
                "mean_start_distance": float(np.mean([x["pre_refinement_incumbent"]["terminal_distance"] for x in items])),
                "mean_calls": float(np.mean([x["total_calls"] for x in items])),
                "nonprogress": sum(x["intervention"]["controls"]["selected"]["prefix_distance"]
                                  >= x["intervention"]["initial_distance"] - 1e-12 for x in items),
                "limited_witnesses": sum(x["intervention"]["limited_witness"] for x in items)}
        summary[str(cap)] = panel
    return summary


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    print("Rank-two nominal response vs full ridge/random; exact exposed development only.", flush=True)
    rows = []
    for index in range(16):
        angle = 2 * np.pi * (index + .37) / 16
        goal = .9 * np.array([np.cos(angle), np.sin(angle)])
        results = []
        for cap in CAPS:
            arms = {m: plan(goal, index, cap, m) for m in METHODS}
            for arm in arms.values():
                arm["intervention"] = joint.intervention(goal, arm["point"], arms["task_greedy"]["point"])
            results.extend(arms.values())
        rows.append({"index": index, "goal": goal.tolist(), "results": results})
        print(f"completed root {index}", flush=True)
    report = {"scope": "exact-model exposed development only",
              "P1_admission": False, "full_cost_certified": False,
              "dependencies": {str(p): joint.digest(p) for p in (
                  PROTOCOL, Path(__file__), Path(joint.__file__), Path(joint.dynamics.__file__),
                  Path(joint.strong.__file__), Path(modes.__file__))},
              "rows": rows, "summary": summarize(rows)}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
